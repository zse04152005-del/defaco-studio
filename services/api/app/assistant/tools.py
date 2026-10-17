"""Read-only tools the assistant may call. Every number it reports must come from one of these.

Sources: the research engine's exported results (web/public/demo-data, data as of 2025-12-31),
the live quote layer (Tencent / Sina, possibly delayed) and the research write-ups.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

from ..realtime import INDICES, MarketData, session_status

ROOT = Path(__file__).resolve().parents[4]
DEMO = ROOT / "web" / "public" / "demo-data"
FACTOR_CN = {
    "size": "市值", "mom_12_2": "12-2 月动量", "mom_6_1": "6-1 月动量", "rev_1m": "1 月反转（上月收益）",
    "vol_1m": "1 月波动率", "vol_12m": "12 月波动率", "maxret_1m": "上月最大日收益", "turn_1m": "1 月换手",
    "turn_6m": "6 月换手", "abturn": "异常换手", "illiq": "非流动性", "beta_12m": "12 月 β", "ivol_3m": "3 月特质波动",
    "high52": "距 52 周高点", "ep": "盈利收益率 E/P", "bm": "账面市值比 B/M", "sp": "营收市值比 S/P",
    "cfp": "现金流市值比", "roe": "ROE", "gross_margin": "毛利率", "revenue_yoy": "营收同比", "profit_yoy": "利润同比",
    "asset_growth": "资产增长", "leverage": "杠杆",
}
# what a HIGH percentile means (so the model cannot read e.g. a low illiquidity rank as "illiquid")
FACTOR_HIGH = {
    "size": "市值大", "mom_12_2": "过去一年（剔除近月）涨得多", "mom_6_1": "过去半年（剔除近月）涨得多",
    "rev_1m": "上个月涨得多", "vol_1m": "近 1 月波动大", "vol_12m": "近 1 年波动大", "maxret_1m": "上月出现过大涨日",
    "turn_1m": "近 1 月换手高（交易活跃）", "turn_6m": "近半年换手高", "abturn": "近期换手相对自身历史异常放大",
    "illiq": "流动性差（单位成交额引起的价格冲击大）", "beta_12m": "对市场敏感（β 高）", "ivol_3m": "特质波动大",
    "high52": "价格接近 52 周高点", "ep": "市盈率低（便宜）", "bm": "市净率低（便宜）", "sp": "市销率低（便宜）",
    "cfp": "市现率低（便宜）", "roe": "ROE 高", "gross_margin": "毛利率高", "revenue_yoy": "营收增长快",
    "profit_yoy": "利润增长快", "asset_growth": "资产扩张快", "leverage": "负债率高",
}
INDEX_NAMES = {"000300.SH": "沪深300", "000905.SH": "中证500", "000001.SH": "上证指数", "399001.SZ": "深证成指", "399006.SZ": "创业板指"}


def r(x: Any, n: int = 4) -> Any:
    """Round floats for compact tool output (keeps the numbers the model can cite)."""
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, n)
    if isinstance(x, dict):
        return {k: r(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [r(v, n) for v in x]
    return x


@lru_cache(maxsize=None)
def _json(name: str) -> dict:
    return json.loads((DEMO / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _securities() -> pd.DataFrame:
    return pd.DataFrame(_json("securities.json")["items"]).set_index("code", drop=False)


@lru_cache(maxsize=1)
def _screener() -> pd.DataFrame:
    return pd.DataFrame(_json("screener.json")["items"]).set_index("code", drop=False)


@lru_cache(maxsize=1)
def _all_securities() -> pd.DataFrame:
    """Every listed A-share (securities_all.json): code, name, industry; research universe first."""
    path = DEMO / "securities_all.json"
    if not path.exists():
        return _securities()[["code", "name", "industry"]]
    d = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(d["rows"], columns=d["fields"])
    df = df[df["board"] != "index"].sort_values("research", ascending=False)
    return df.drop_duplicates("code").set_index("code", drop=False)[["code", "name", "industry"]]


def _resolve(code_or_name: str) -> str | None:
    """Code or name -> canonical code; the research universe is tried before the full A-share list."""
    q = code_or_name.strip().upper()
    for sec in (_securities(), _all_securities()):
        if q in sec.index:
            return q
        num = q.split(".")[0]
        hit = sec[sec["code"].str.startswith(num)] if num.isdigit() else sec[sec["name"] == code_or_name.strip()]
        if len(hit) == 0 and not num.isdigit():
            hit = sec[sec["name"].str.contains(code_or_name.strip(), regex=False)]
        if len(hit):
            return hit.index[0]
    return None


# ----------------------------------------------------------------------------- technicals
def technicals(bars: list[list]) -> dict[str, Any]:
    """Summary indicators from daily bars [t, o, h, l, c, vol, ...] (oldest first)."""
    df = pd.DataFrame([b[:6] for b in bars], columns=["t", "o", "h", "l", "c", "v"]).astype({"c": float, "h": float, "l": float})
    c = df["c"]
    ret = c.pct_change()
    out: dict[str, Any] = {"last_date": str(df["t"].iloc[-1])[:10], "close": float(c.iloc[-1])}
    for n in (5, 20, 60, 250):
        if len(c) > n:
            out[f"ret_{n}d"] = float(c.iloc[-1] / c.iloc[-1 - n] - 1)
    for n in (5, 20, 60):
        if len(c) >= n:
            out[f"ma{n}"] = float(c.rolling(n).mean().iloc[-1])
    out["above_ma20"] = bool(c.iloc[-1] > out.get("ma20", np.inf))
    out["above_ma60"] = bool(c.iloc[-1] > out.get("ma60", np.inf))
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / 14).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    out["rsi14"] = float(100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1])) if dn.iloc[-1] > 0 else 100.0
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    out["macd_dif"], out["macd_dea"] = float(dif.iloc[-1]), float(dea.iloc[-1])
    out["macd_state"] = "金叉区域（DIF>DEA）" if dif.iloc[-1] > dea.iloc[-1] else "死叉区域（DIF<DEA）"
    last250 = df.tail(250)
    hi, lo = float(last250["h"].max()), float(last250["l"].min())
    out["high_250d"], out["low_250d"] = hi, lo
    out["pos_in_250d_range"] = (float(c.iloc[-1]) - lo) / (hi - lo) if hi > lo else None
    out["ann_vol_60d"] = float(ret.tail(60).std() * np.sqrt(252))
    nav = c.tail(250)
    out["max_drawdown_250d"] = float((nav / nav.cummax() - 1).min())
    return out


# ----------------------------------------------------------------------------- tool implementations
class Toolbox:
    """Read-only tools. ``ctx`` is the browser's request context (page, lists); ``pf`` the live
    portfolio engine (None until the research panel is loaded)."""

    def __init__(self, md: MarketData, pf: Any = None, ctx: dict | None = None, fs: Any = None):
        self.md = md
        self.pf = pf
        self.ctx = ctx or {}
        self.fs = fs            # FactorService (factor research), optional

    async def factor_research(self, name: str | None = None, neutral: str = "industry_size", universe: str = "csi800", top: int = 10) -> dict:
        import asyncio
        if self.fs is None:
            return {"error": "因子研究服务不可用"}
        if name:
            d = await asyncio.to_thread(self.fs.detail, name, universe, neutral, 1)
            return {"factor": d["name"], "family": d["family"], "formula": d["formula"], "desc": d["desc"], "stats": d["stats"],
                    "quantile_mean_monthly": d["quantile_mean"], "ic_decay_by_month": d["decay"], "yearly_rank_ic": d["yearly_ic"],
                    "note": "月末截面、未来 1 个月收益；stats.t 为 IC 均值 t 值；ls_* 为多空组合（未扣成本）；turnover 为头部组月换手"}
        s = await asyncio.to_thread(self.fs.summary, universe, neutral, 1)
        rows = sorted(s["rows"], key=lambda r: -abs(r.get("icir") or 0))[: max(1, min(top, 30))]
        keep = ("name", "family", "desc", "rank_ic", "icir", "t", "ls_ann", "ls_sharpe", "turnover")
        return {"universe": s["universe"], "neutral": s["neutral"], "period": f"{s['start']} → {s['end']}", "n_factors": s["n_factors"],
                "top_by_abs_icir": [{k: r.get(k) for k in keep} for r in rows],
                "note": "Rank IC 为因子与下月收益的截面秩相关；方向由 IC 符号决定；同时检验很多因子时建议以 |t|>3 为门槛"}

    # --- the user's lists, holdings and live portfolios (analysis.py) and charts (charts.py)
    async def analyze_stocks(self, source: str = "watchlist", codes: list[str] | None = None) -> dict:
        from .analysis import analyze_stocks
        return await analyze_stocks(self, source, codes)

    async def analyze_holdings(self, model: str = "lw_nonlinear") -> dict:
        from .analysis import analyze_holdings
        return await analyze_holdings(self, model)

    async def get_live_portfolio(self, model: str | None = None, max_weight: float | None = None,
                                 industry_cap: float | None = None, use_pool: bool | None = None) -> dict:
        from .analysis import get_live_portfolio
        return await get_live_portfolio(self, model, max_weight, industry_cap, use_pool)

    async def make_chart(self, kind: str, **kw: Any) -> dict:
        from .charts import ChartError, build_chart
        try:
            option, summary = await build_chart(self, kind, **kw)
        except ChartError as exc:
            return {"error": str(exc)}
        return {"kind": kind, "title": (option.get("title") or {}).get("text", ""), "summary": summary, "_chart": option,
                "note": "图表已在界面中显示给用户；回答中可写“见图 N”，并引用 summary 中的关键数字"}

    async def get_market_overview(self) -> dict:
        ov = _json("overview.json")
        idx = {}
        for sym, series in ov["indices"].items():
            s = pd.Series({d: v for d, v in series})
            name = {"sh000300": "沪深300", "sh000905": "中证500"}.get(sym, sym)
            idx[name] = {"close": float(s.iloc[-1]), "ret_20d": float(s.iloc[-1] / s.iloc[-21] - 1),
                         "ret_60d": float(s.iloc[-1] / s.iloc[-61] - 1), "ret_250d": float(s.iloc[-1] / s.iloc[-251] - 1)}
        inds = sorted(ov["industries"], key=lambda x: x["ret"])
        regime = ov["vol_regime"][-1]
        out = {"research_asof": ov["asof"], "indices_at_asof": idx,
               "breadth_at_asof": {k: ov["breadth"][k] for k in ("up", "down", "flat")},
               "best_industries_1d": [{"industry": x["ind"], "ret": x["ret"]} for x in inds[-5:][::-1]],
               "worst_industries_1d": [{"industry": x["ind"], "ret": x["ret"]} for x in inds[:5]],
               "vol_regime_latest": {"date": regime[0], "ann_vol_20d": regime[1],
                                     "high_vol_threshold": ov["vol_threshold"],
                                     "state": "高波动" if regime[1] > ov["vol_threshold"] else "低波动"}}
        try:
            q = await self.md.get_quotes(list(INDEX_NAMES))
            out["live_indices"] = {INDEX_NAMES[k]: {"price": v.get("price"), "pct": v.get("pct"), "time": v.get("time")}
                                   for k, v in q.items()}
            out["session"] = session_status()
        except Exception:  # noqa: BLE001
            out["live_indices"] = "实时行情暂不可用"
        return out

    async def search_security(self, query: str) -> dict:
        sec = _securities()
        q = query.strip()
        hit = sec[sec["code"].str.contains(q.upper(), regex=False) | sec["name"].str.contains(q, regex=False)].head(8)
        return {"matches": hit[["code", "name", "industry", "mcap"]].to_dict("records"),
                "note": "mcap 为流通市值（亿元，截至研究数据日）"}

    async def get_realtime_quotes(self, codes: list[str]) -> dict:
        codes = [c for c in (_resolve(x) or x for x in codes[:20])]
        q = await self.md.get_quotes(codes)
        keep = ("name", "price", "pct", "change", "open", "high", "low", "prev_close", "volume", "amount",
                "turnover", "pe", "pb", "limit_up", "limit_down", "time", "source")
        return {"session": session_status(), "quotes": {c: {k: v.get(k) for k in keep} for c, v in q.items()},
                "note": "免费公开行情，可能有延迟；pct、turnover 为小数（0.01 = 1%）；volume 单位为股，amount 单位为元"}

    async def get_stock_analysis(self, code: str) -> dict:
        c = _resolve(code)
        if not c:
            return {"error": f"找不到证券 {code}，可先调用 search_security"}
        sec = _securities().loc[c]
        out: dict[str, Any] = {"code": c, "name": sec["name"], "industry": sec["industry"], "float_mcap_yi": sec["mcap"]}
        bars, src = None, None
        try:
            k = await self.md.get_kline(c, 2)
            if k.get("bars"):
                bars, src = k["bars"], f"{k.get('source', 'tencent')} 前复权日线（至今日）"
        except Exception:  # noqa: BLE001
            pass
        if bars is None and (DEMO / "stocks" / f"{c}.json").exists():
            bars, src = _json(f"stocks/{c}.json")["bars"], "研究数据日线（至 2025-12-31）"
        if bars:
            out["technicals"] = technicals(bars)
            out["technicals_source"] = src
        scr = _screener()
        if c in scr.index:
            row = scr.loc[c]
            prof = {f: float(row[f]) for f in FACTOR_CN if f in row and pd.notna(row[f])}
            out["factor_percentiles"] = {f"{FACTOR_CN[f]}（高分位={FACTOR_HIGH[f]}）": v for f, v in prof.items()}
            out["factor_note"] = (f"截面分位数（0=最低，1=最高），截至 {_json('screener.json')['asof']}，在 {len(scr)} 只股票中排名；"
                                  "括号内是“分位高”的含义，分位低则含义相反（如非流动性分位低 = 流动性好）")
            peers = scr[scr["industry"] == sec["industry"]].sort_values("mcap", ascending=False).head(6)
            out["industry_peers"] = peers[["code", "name", "mcap", "ret_1m"]].to_dict("records")
        held = {}
        for m, p in _json("portfolio.json")["models"].items():
            w = {x["code"]: x for x in p["weights"]}
            if c in w:
                held[p["label"]] = {"weight": w[c]["w"], "risk_contribution": w[c]["rc"]}
        out["in_model_portfolios"] = held or "不在任何模型组合中（研究股票池为沪深300 市值前 100）"
        return out

    async def get_portfolio(self, model: str = "pca_factor") -> dict:
        P = _json("portfolio.json")
        if model not in P["models"]:
            return {"error": f"未知模型 {model}", "available": {k: v["label"] for k, v in P["models"].items()}}
        m = P["models"][model]
        return {"model": model, "label": m["label"], "asof": P["asof"], "universe_size": P["universe_size"],
                "objective": "多头最小方差，单股上限 10%", "n_holdings": m["n_holdings"], "hhi": m["hhi"],
                "risk": {k: v for k, v in m["risk"].items() if k != "exposures"},
                "risk_note": "total_vol 为年化预期波动；factor_share/idio_share 为因子/特质风险占比",
                "top_holdings": m["weights"][:12],
                "holdings_note": "w = 权重，rc = 风险贡献占比",
                "industry_weights": dict(sorted(m["industry_weights"].items(), key=lambda kv: -kv[1])),
                "model_explain": m.get("explain", {}),
                "available_models": {k: v["label"] for k, v in P["models"].items()}}

    async def compare_portfolios(self, model_a: str, model_b: str) -> dict:
        P = _json("portfolio.json")["models"]
        if model_a not in P or model_b not in P:
            return {"error": "未知模型", "available": {k: v["label"] for k, v in P.items()}}
        a, b = P[model_a], P[model_b]
        inds = set(a["industry_weights"]) | set(b["industry_weights"])
        dind = sorted(({"industry": i, "a": a["industry_weights"].get(i, 0.0), "b": b["industry_weights"].get(i, 0.0),
                        "diff": a["industry_weights"].get(i, 0.0) - b["industry_weights"].get(i, 0.0)} for i in inds),
                      key=lambda x: -abs(x["diff"]))
        wa, wb = {x["code"]: x for x in a["weights"]}, {x["code"]: x for x in b["weights"]}
        dst = sorted(({"code": c, "name": (wa.get(c) or wb.get(c))["name"], "a": wa.get(c, {}).get("w", 0.0),
                       "b": wb.get(c, {}).get("w", 0.0)} for c in set(wa) | set(wb)),
                     key=lambda x: -abs(x["a"] - x["b"]))
        return {"a": a["label"], "b": b["label"], "vol_a": a["risk"]["total_vol"], "vol_b": b["risk"]["total_vol"],
                "n_holdings_a": a["n_holdings"], "n_holdings_b": b["n_holdings"],
                "industry_diff_top": dind[:8], "stock_diff_top": dst[:10]}

    async def get_backtest(self, models: list[str] | None = None) -> dict:
        B = _json("backtest.json")
        res = B["results"]
        models = [m for m in (models or list(res)) if m in res]
        out = {}
        for m in models:
            x = res[m]
            mon = pd.Series(x["monthly"])
            yearly = (1 + mon).groupby(mon.index.str[:4]).prod() - 1
            out[m] = {"label": x["label"], "metrics": x["metrics"], "yearly_returns": yearly.to_dict(),
                      "worst_months": mon.nsmallest(3).to_dict(), "best_months": mon.nlargest(3).to_dict()}
        return {"period": f"{B['start']} → {B['end']}", "results": out,
                "note": "月末调仓、扣除 A 股真实交易成本；ann_* 为年化；sharpe 为年化 Sharpe；turnover 为月均单边换手"}

    async def screen_stocks(self, filters: dict | None = None, industries: list[str] | None = None,
                            sort_by: str = "mcap", top_n: int = 15) -> dict:
        df = _screener()
        rev = {v: k for k, v in FACTOR_CN.items()}
        for f, rng in (filters or {}).items():
            f = rev.get(f, f)
            if f in df and isinstance(rng, (list, tuple)) and len(rng) == 2:
                df = df[(df[f] >= rng[0]) & (df[f] <= rng[1])]
        if industries:
            df = df[df["industry"].isin(industries)]
        sort_by = rev.get(sort_by, sort_by)
        df = df.sort_values(sort_by if sort_by in df else "mcap", ascending=False)
        cols = ["code", "name", "industry", "mcap", "ret_1m"] + [f for f in (filters or {}) if f in df][:6]
        return {"asof": _json("screener.json")["asof"], "n_hits": int(len(df)), "top": df.head(min(top_n, 30))[cols].to_dict("records"),
                "note": "因子值为截面分位数 0–1；ret_1m 为近 1 月收益；mcap 为流通市值（亿元）",
                "available_factors": {f: f"{FACTOR_CN[f]}（高分位={FACTOR_HIGH[f]}）" for f in FACTOR_CN}}

    async def get_research_findings(self, topic: str = "summary") -> dict:
        # stage reports live in the thesis folder next to defaco-studio ($DEFACO_PAPER overrides)
        rep = Path(os.environ.get("DEFACO_PAPER") or ROOT.parent) / "00_项目管理" / "阶段报告"
        files = {"m0": rep / "M0_复现结论.md", "m1": rep / "M1_结论.md", "m2": rep / "M2_结论.md", "m3": rep / "M3_结论.md", "m4": rep / "M4_结论.md"}
        if topic in files and files[topic].exists():
            return {"topic": topic, "text": files[topic].read_text(encoding="utf-8")[:7000]}
        if rep.exists():
            return {"topic": "summary", "available": [k for k, f in files.items() if f.exists()],
                    "hint": "topic='m0' / 'm1' / 'm2' / 'm3' / 'm4' 可获取各阶段完整结论"}
        return {"topic": "summary",
                "method": "DeFaCo：特征驱动载荷 + 因子协方差 + 行业块结构残差 + 可微组合层 + 决策导向、换手感知训练（毕业论文模型）",
                "status": "论文实验结论尚未公开（将在论文答辩后发布）；本安装中没有阶段报告。请只介绍方法思路，不要给出任何实验数字。"}


ToolFn = Callable[..., Awaitable[dict]]

SCHEMAS: list[dict] = [
    {"name": "get_market_overview", "description": "市场总览：沪深300/中证500 涨跌、涨跌家数、行业涨跌前后五、波动区制，以及实时指数行情。",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "search_security", "description": "按名称或代码模糊搜索 A 股证券，返回代码、行业、流通市值。用户只给名称时先调用它。",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_realtime_quotes", "description": "实时行情（腾讯/新浪，可能延迟）：价格、涨跌幅、成交额、换手、PE/PB、涨跌停价。",
     "parameters": {"type": "object", "properties": {"codes": {"type": "array", "items": {"type": "string"},
                                                              "description": "如 600519.SH、000001.SZ，或股票名称"}},
                    "required": ["codes"]}},
    {"name": "get_stock_analysis", "description": "个股分析数据：技术指标（收益、均线、RSI、MACD、波动、回撤、52 周区间位置）、24 个因子截面分位、同行业可比公司、在各模型组合中的权重。",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"name": "get_portfolio", "description": "某个风险模型下的最小方差组合：持仓、风险贡献、行业权重、预期波动、因子/特质风险占比（DeFaCo 另有行业块风险占比）。model 可选 defaco（论文模型 DeFaCo M4）/ pca_factor / lw_nonlinear / lw_linear / sample / ewma。",
     "parameters": {"type": "object", "properties": {"model": {"type": "string", "default": "pca_factor"}}}},
    {"name": "compare_portfolios", "description": "比较两个模型组合的行业权重与个股权重差异，用于回答“为什么减配/增配某行业”。模型名同 get_portfolio（含 defaco）。",
     "parameters": {"type": "object", "properties": {"model_a": {"type": "string"}, "model_b": {"type": "string"}},
                    "required": ["model_a", "model_b"]}},
    {"name": "get_backtest", "description": "2016–2025 滚动样本外回测绩效：年化收益/波动、Sharpe、最大回撤、换手、逐年收益、最好/最差月份。models 可选 EW / lw_nonlinear / pca_factor / sample。",
     "parameters": {"type": "object", "properties": {"models": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "screen_stocks", "description": "因子选股：filters 形如 {\"ep\": [0.7, 1], \"vol_12m\": [0, 0.3]}（分位数区间 0–1），可限定行业，按某因子或 mcap 降序取前 top_n。",
     "parameters": {"type": "object", "properties": {
         "filters": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "number"}}},
         "industries": {"type": "array", "items": {"type": "string"}}, "sort_by": {"type": "string"},
         "top_n": {"type": "integer"}}}},
    {"name": "analyze_stocks", "description": "批量分析用户的自选股 / 候选池 / 持仓股票 / 指定股票：实时价、近 20/60/250 日收益、波动、回撤、RSI、8 个关键因子分位、在模型组合中的权重、行业分布、两两相关性和分散效果。",
     "parameters": {"type": "object", "properties": {
         "source": {"type": "string", "enum": ["watchlist", "pool", "holdings", "codes"], "description": "watchlist=自选，pool=候选池，holdings=持仓，codes=用 codes 参数指定"},
         "codes": {"type": "array", "items": {"type": "string"}}}, "required": ["source"]}},
    {"name": "analyze_holdings", "description": "分析用户在「持仓调仓」页录入的持仓：按市值的权重与集中度、风险模型估计的年化波动和每只股票的风险贡献、行业权重，以及同一批股票的最小方差组合作为风险对照。",
     "parameters": {"type": "object", "properties": {"model": {"type": "string", "enum": ["defaco", "lw_nonlinear", "lw_linear", "pca_factor", "sample", "ewma"]}}}},
    {"name": "get_live_portfolio", "description": "按用户当前「组合构建」页参数（或传入的参数）实时求解多头最小方差组合；use_pool=true 时用候选池作为股票池。model 可选 defaco（论文模型 DeFaCo M4，结果含模型年份、κ、行业块风险和特征暴露）/ pca_factor / lw_nonlinear / lw_linear / sample / ewma。",
     "parameters": {"type": "object", "properties": {"model": {"type": "string"}, "max_weight": {"type": "number"},
                                                    "industry_cap": {"type": "number"}, "use_pool": {"type": "boolean"}}}},
    {"name": "make_chart", "description": "生成图表并直接显示给用户。kind：price_compare（多股累计收益走势）、kline（单股 K 线）、factor_radar（因子雷达，≤4 只）、correlation（相关性热力图）、industry_pie（行业分布）、weights_bar（权重与风险贡献）、backtest_nav（回测净值）、bar/line（自定义，数值必须来自工具结果）。股票可用 codes 或 source（watchlist/pool/holdings/portfolio）指定。",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["price_compare", "kline", "factor_radar", "correlation", "industry_pie", "weights_bar", "backtest_nav", "bar", "line"]},
         "codes": {"type": "array", "items": {"type": "string"}},
         "source": {"type": "string", "enum": ["watchlist", "pool", "holdings", "portfolio"]},
         "days": {"type": "integer", "description": "回看交易日数，默认 250"},
         "model": {"type": "string"}, "models": {"type": "array", "items": {"type": "string"}},
         "title": {"type": "string"},
         "labels": {"type": "array", "items": {"type": "string"}},
         "series": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "values": {"type": "array", "items": {"type": "number"}}}}},
         "unit": {"type": "string"}}, "required": ["kind"]}},
    {"name": "factor_research", "description": "因子研究：不传 name 时返回 135 个因子（自建 24 个 + Qlib Alpha158 / WorldQuant Alpha101 / 国泰君安 191）按 |ICIR| 排序的前 top 个；传 name（如 turn_1m、WQ040、CORD20）返回该因子的 IC、t 值、多空收益、分层收益、IC 衰减、逐年 IC。",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "neutral": {"type": "string", "enum": ["none", "industry", "industry_size"]},
                                                    "universe": {"type": "string", "enum": ["csi800", "csi300"]}, "top": {"type": "integer"}}}},
    {"name": "get_research_findings", "description": "本项目论文研究（DeFaCo 消融阶梯 M0–M4）的结论与数字。topic: summary / m0 / m1 / m2 / m3 / m4。",
     "parameters": {"type": "object", "properties": {"topic": {"type": "string", "enum": ["summary", "m0", "m1", "m2", "m3", "m4"]}}}},
]
SOURCE_LABEL = {
    "get_market_overview": "市场总览", "search_security": "证券检索", "get_realtime_quotes": "实时行情",
    "get_stock_analysis": "个股分析", "get_portfolio": "组合引擎", "compare_portfolios": "组合对比",
    "get_backtest": "回测引擎", "screen_stocks": "因子选股", "get_research_findings": "研究结论",
    "factor_research": "因子研究", "analyze_stocks": "批量分析", "analyze_holdings": "持仓分析", "get_live_portfolio": "实时组合", "make_chart": "图表",
}


def openai_tools() -> list[dict]:
    return [{"type": "function", "function": s} for s in SCHEMAS]


async def call_tool(box: Toolbox, name: str, args: dict) -> dict:
    fn = getattr(box, name, None)
    if fn is None or name not in SOURCE_LABEL:
        return {"error": f"未知工具 {name}"}
    try:
        return r(await fn(**args))
    except TypeError as exc:
        return {"error": f"参数错误：{exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
