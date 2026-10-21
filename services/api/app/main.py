"""DeFaCo Studio API (v0.1): real-time market data for the demo.

    .venv/bin/uvicorn app.main:app --app-dir services/api --port 8000

REST
  GET /api/market/status            trading-session status + data-source health
  GET /api/realtime/quotes?codes=   batched quotes (Tencent -> Sina)
  GET /api/realtime/minute/{code}   today's 1-minute series (Tencent -> pytdx -> AkShare)
  GET /api/kline/{code}?years=3     forward-adjusted daily K-line up to today
  GET  /api/assistant/status        LLM configured? model names (never the key)
  POST /api/assistant/chat          {messages, context} -> SSE: tool-using Q&A agent
  POST /api/assistant/report        {code, context}     -> SSE: bull/bear/editor stock report
  POST /api/portfolio/optimize      {model, objective, max_weight, industry_cap, universe?, asof?}
                                    -> live min-variance portfolio (shape of portfolio.json models[...])
  GET  /api/portfolio/options       risk models / objectives and whether the research panel is loaded
WebSocket
  /ws/quotes   send {"subscribe": ["600519.SH", ...]}; receives {"type": "quotes", ...}
               every 3 s while the market is open (30 s otherwise)
"""
from __future__ import annotations

import asyncio
import time
import json
import logging
from contextlib import asynccontextmanager

from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .assistant.agent import guarded, run_chat, run_report
from .assistant.llm import ChatClient
from .assistant.tools import Toolbox
from .backtest import BacktestService, public, returns_csv
from .factors import FactorService
from .backtest_report import build_report
from .monitor import Monitor, MonitorError
from .portfolio import MODELS, OBJECTIVES, OptimizeError, PortfolioEngine
from .realtime import INDICES, MarketData, session_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")
md: MarketData | None = None
llm: ChatClient | None = None
pf: PortfolioEngine | None = None             # research panel, loaded once in the background
pf_task: asyncio.Future | None = None
mon: Monitor | None = None
bt: BacktestService | None = None
fs: FactorService | None = None


def _load_portfolio_engine() -> None:
    global pf
    from quantkit.config import load_settings
    pf = PortfolioEngine.from_store(load_settings().data_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global md, llm, pf_task, mon, bt, fs
    md = MarketData()
    llm = ChatClient()
    pf_task = asyncio.ensure_future(asyncio.to_thread(_load_portfolio_engine))  # quotes are served meanwhile
    mon = Monitor(md, lambda: pf)
    from quantkit.config import load_settings
    bt = BacktestService(load_settings().data_dir)
    fs = FactorService(load_settings().data_dir)
    mon_task = asyncio.create_task(mon.loop())
    yield
    mon_task.cancel()
    await mon.http.aclose()
    await md.http.aclose()
    await llm.aclose()


app = FastAPI(title="DeFaCo Studio API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                   allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["*"])


@app.get("/api/market/status")
async def market_status():
    return {"session": session_status(), "sources": md.health,
            "disclaimer": "免费公开行情接口，可能存在延迟；仅供研究与教学"}


@app.get("/api/realtime/quotes")
async def quotes(codes: str = ""):
    wanted = [c.strip().upper() for c in codes.split(",") if c.strip()][:300]
    return {"session": session_status(), "quotes": await md.get_quotes(wanted)}


@app.get("/api/realtime/minute/{code}")
async def minute(code: str):
    return await md.get_minutes(code.upper())


@app.get("/api/kline/{code}")
async def kline(code: str, years: int = 3):
    return await md.get_kline(code.upper(), max(1, min(years, 10)))


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] | None = None


class ReportRequest(BaseModel):
    code: str
    context: dict[str, Any] | None = None


def _sse(gen):
    async def body():
        async for ev in guarded(gen):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/assistant/status")
async def assistant_status():
    c = llm.cfg
    return {"ready": c.ready, "enabled": c.enabled, "has_key": bool(c.api_key), "model": c.model, "model_pro": c.model_pro,
            "provider": c.label, "provider_id": c.provider, "problem": c.problem}


@app.post("/api/assistant/chat")
async def assistant_chat(req: ChatRequest):
    return _sse(run_chat(llm, Toolbox(md, pf, req.context, fs), req.messages, req.context))


@app.post("/api/assistant/report")
async def assistant_report(req: ReportRequest):
    return _sse(run_report(llm, Toolbox(md, pf, req.context), req.code, req.context))


class OptimizeRequest(BaseModel):
    model: str = "pca_factor"
    objective: str = "minvar"
    max_weight: float = Field(0.10, gt=0, le=1)
    industry_cap: float | None = Field(0.35, gt=0, le=1)
    universe: list[str] | None = Field(None, max_length=1000)
    asof: str | None = None
    risk_aversion: float = Field(5.0, gt=0, le=1000)   # mean-variance only
    compare: bool = True                               # characteristic exposures vs the NLS portfolio


# ----------------------------------------------------------------------------- live backtests (F8)
class BacktestStrategy(BaseModel):
    kind: str = Field("opt", pattern="^(opt|ew|vw)$")
    model: str = "lw_nonlinear"
    objective: str = "minvar"
    max_weight: float = Field(0.10, gt=0, le=1)
    industry_cap: float | None = Field(0.35, gt=0, le=1)
    risk_aversion: float = Field(5.0, gt=0, le=1000)
    label: str | None = Field(None, max_length=40)


class BacktestExecution(BaseModel):
    timing: str = Field("next_open", pattern="^(next_open|close)$")
    aum: float = Field(1e8, gt=1e4, le=1e12)
    commission_bps: float = Field(2.5, ge=0, le=50)
    min_commission: float = Field(5.0, ge=0, le=100)
    impact_y: float = Field(0.5, ge=0, le=5)
    participation_cap: float | None = Field(0.10, gt=0, le=1)


class BacktestRequest(BaseModel):
    strategies: list[BacktestStrategy] = Field(default_factory=lambda: [BacktestStrategy()], min_length=1, max_length=6)
    include_ew: bool = True
    start: str = "2016-01-01"
    end: str = "2025-12-31"
    freq: str = Field("M", pattern="^(M|W|Q)$")
    top_n: int = Field(100, ge=20, le=300)
    universe: list[str] | None = Field(None, max_length=500)
    execution: BacktestExecution = Field(default_factory=BacktestExecution)


def _job(job_id: str):
    from fastapi import HTTPException
    job = bt.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="回测任务不存在或已过期")
    return job


@app.post("/api/backtest/run")
async def backtest_run(req: BacktestRequest):
    for s in req.strategies:
        if s.kind == "opt" and (s.model not in MODELS or s.objective not in OBJECTIVES):
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail=f"未知模型或目标：{s.model} / {s.objective}")
    job = bt.submit(req.model_dump())
    return {"job": job.id, "status": job.status}


@app.get("/api/backtest/jobs/{job_id}")
async def backtest_job(job_id: str):
    job = _job(job_id)
    out = {"job": job.id, "status": job.status, "progress": round(job.progress, 4), "message": job.message,
           "error": job.error, "elapsed": round((job.finished or time.time()) - job.started, 1)}
    if job.status == "done":
        out["result"] = public(job.result)
    return out


@app.get("/api/backtest/jobs/{job_id}/report.html")
async def backtest_report(job_id: str):
    from fastapi import HTTPException
    from fastapi.responses import HTMLResponse
    job = _job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail="回测尚未完成")
    page = await asyncio.to_thread(build_report, job.result)
    return HTMLResponse(page, headers={"Content-Disposition": f'attachment; filename="defaco_backtest_{job.id}.html"'})


@app.get("/api/backtest/jobs/{job_id}/returns.csv")
async def backtest_csv(job_id: str):
    from fastapi import HTTPException
    from fastapi.responses import Response
    job = _job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail="回测尚未完成")
    return Response("\ufeff" + returns_csv(job.result), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="defaco_returns_{job.id}.csv"'})


# ----------------------------------------------------------------------------- factor research (F4)
class FactorCorrRequest(BaseModel):
    names: list[str] = Field(..., min_length=2, max_length=30)
    universe: str = "csi800"
    neutral: str = "industry_size"


def _fs_call(fn, *a):
    from fastapi import HTTPException
    try:
        return fn(*a)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未知因子 {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="因子数据未生成：先运行 scripts/build_alpha_factors.py") from exc


@app.get("/api/factors/summary")
async def factors_summary(universe: str = "csi800", neutral: str = "industry_size", h: int = 1):
    return await asyncio.to_thread(_fs_call, fs.summary, universe, neutral, h)


@app.get("/api/factors/detail/{name}")
async def factors_detail(name: str, universe: str = "csi800", neutral: str = "industry_size", h: int = 1):
    return await asyncio.to_thread(_fs_call, fs.detail, name, universe, neutral, h)


@app.post("/api/factors/corr")
async def factors_corr(req: FactorCorrRequest):
    return await asyncio.to_thread(_fs_call, fs.corr, req.names, req.universe, req.neutral)


# ----------------------------------------------------------------------------- monitor & push (F10)
class MonitorRun(BaseModel):
    rules: list[str] | None = None
    push: bool = True
    force: bool = False


class MonitorTest(BaseModel):
    channel: str | None = None


def _mon_error(exc: MonitorError):
    from fastapi import HTTPException
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/monitor/config")
async def monitor_config():
    return mon.public_config()


@app.put("/api/monitor/config")
async def monitor_update(patch: dict[str, Any]):
    try:
        return mon.update_config(patch)
    except (MonitorError, ValueError) as exc:
        raise _mon_error(MonitorError(str(exc))) from exc


@app.post("/api/monitor/state")
async def monitor_state(client: dict[str, Any]):
    """The browser's lists (holdings, watchlist, pool, 组合构建 settings) for the scheduled rules."""
    mon.set_client(client)
    return {"ok": True, "updated": mon.state["client"]["updated"]}


@app.post("/api/monitor/run")
async def monitor_run(req: MonitorRun):
    events = await mon.run(req.rules, push=req.push, force=req.force)
    return {"events": events, "last_run": mon.last_run}


@app.post("/api/monitor/test")
async def monitor_test(req: MonitorTest):
    title = "DeFaCo Studio 测试推送"
    body = "这是一条测试消息：渠道配置正确时，你会在这里收到组合风险、偏离、区制切换和涨跌停预警。"
    res = await mon.push(title, body, "info", only=req.channel)
    if not res:
        raise _mon_error(MonitorError("没有已启用并填写 webhook 的渠道"))
    return {"results": res}


@app.get("/api/monitor/events")
async def monitor_events(limit: int = 50):
    return {"events": mon.state["events"][: max(1, min(limit, 300))]}


@app.delete("/api/monitor/events")
async def monitor_clear_events():
    mon.state["events"] = []
    mon.save()
    return {"ok": True}


@app.get("/api/portfolio/options")
async def portfolio_options():
    ready = pf is not None
    from quantkit.riskmodels.defaco import default_model_file
    try:
        defaco = default_model_file().exists()
    except FileNotFoundError:
        defaco = False
    return {"ready": ready, "models": MODELS, "objectives": OBJECTIVES, "implemented_objectives": ["minvar", "rp", "hrp", "mv"],
            "defaco": defaco}


@app.post("/api/portfolio/optimize")
async def portfolio_optimize(req: OptimizeRequest):
    if pf is None and pf_task is not None:
        try:  # first request right after startup: wait for the panel
            await asyncio.wait_for(asyncio.shield(pf_task), timeout=60)
        except Exception as e:  # noqa: BLE001  (data missing, or still loading after 60 s)
            log.exception("portfolio engine unavailable")
            raise HTTPException(503, f"研究数据未就绪：{e!r}") from e
    if pf is None:
        raise HTTPException(503, "组合引擎未启动")
    try:
        return await asyncio.to_thread(pf.optimize, **req.model_dump())
    except OptimizeError as e:
        raise HTTPException(e.status, str(e)) from e


class Holding(BaseModel):
    code: str = Field(max_length=12)
    shares: int = Field(ge=0)


class RebalanceRequest(OptimizeRequest):
    compare: bool = False
    holdings: list[Holding] = Field(default_factory=list, max_length=500)
    cash: float = Field(0.0, ge=0)
    prices: dict[str, float] | None = None          # live quotes from the page; the last close otherwise
    frozen_buy: list[str] = Field(default_factory=list, max_length=1000)    # limit-up / suspended
    frozen_sell: list[str] = Field(default_factory=list, max_length=1000)   # limit-down / suspended


@app.post("/api/portfolio/rebalance")
async def portfolio_rebalance(req: RebalanceRequest):
    """Whole-lot order list for the 调仓 page (see app.portfolio.PortfolioEngine.rebalance)."""
    if pf is None and pf_task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(pf_task), timeout=60)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"研究数据未就绪：{e!r}") from e
    if pf is None:
        raise HTTPException(503, "组合引擎未启动")
    body = req.model_dump()
    try:
        return await asyncio.to_thread(pf.rebalance, **body)
    except OptimizeError as e:
        raise HTTPException(e.status, str(e)) from e


@app.websocket("/ws/quotes")
async def ws_quotes(ws: WebSocket):
    await ws.accept()
    subs: set[str] = set(INDICES)

    async def reader():
        while True:
            msg = json.loads(await ws.receive_text())
            if "subscribe" in msg:
                new = {c.upper() for c in msg["subscribe"][:300]} - subs
                subs.update(new)
                if new:  # answer new subscriptions immediately instead of at the next tick
                    data = await md.get_quotes(sorted(new))
                    await ws.send_text(json.dumps({"type": "quotes", "session": session_status(), "quotes": data}, ensure_ascii=False))
            if "unsubscribe" in msg:
                subs.difference_update(c.upper() for c in msg["unsubscribe"])

    task = asyncio.create_task(reader())
    try:
        while True:
            st = session_status()
            data = await md.get_quotes(sorted(subs))
            await ws.send_text(json.dumps({"type": "quotes", "session": st, "quotes": data}, ensure_ascii=False))
            await asyncio.sleep(3 if st["code"] in ("trading", "auction") else 30)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
