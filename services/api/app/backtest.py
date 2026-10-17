"""Live backtests (F8): any risk model x objective on the research data, with a realistic
A-share execution model (quantkit.backtest.simulator) and a statistics-first report.

    POST /api/backtest/run         -> {"job": id}        (runs in a worker thread)
    GET  /api/backtest/jobs/{id}   -> status, progress, result
    GET  /api/backtest/jobs/{id}/report.html | returns.csv

The full research panel (from 2012, enough for a 250-day window + the 1-year listing rule before
2014) is loaded on the first request and kept in memory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantkit.backtest.simulator import CASH, Execution, simulate
from quantkit.data.store import DataStore
from quantkit.metrics.report import deflate, full_metrics
from quantkit.pipeline import load_panel, month_ends
from quantkit.universe import UniverseRule, eligible

from .portfolio import MODELS, OBJECTIVES, PortfolioEngine, _RecentBars

log = logging.getLogger("api.backtest")
FREQ_CN = {"M": "月末", "W": "周末", "Q": "季末"}


@dataclass
class StrategySpec:
    kind: str = "opt"                 # "opt" | "ew" | "vw"
    model: str = "lw_nonlinear"
    objective: str = "minvar"
    max_weight: float = 0.10
    industry_cap: float | None = 0.35
    risk_aversion: float = 5.0
    label: str | None = None

    def name(self) -> str:
        if self.label:
            return self.label
        if self.kind == "ew":
            return "等权 EW"
        if self.kind == "vw":
            return "市值加权 VW"
        return f"{MODELS.get(self.model, self.model)} · {OBJECTIVES.get(self.objective, self.objective)}"


class _OptStrategy:
    def __init__(self, eng: PortfolioEngine, spec: StrategySpec, universe_fn):
        self.eng, self.spec, self.universe_fn, self.name = eng, spec, universe_fn, spec.name()

    def target_weights(self, ctx) -> pd.Series:
        codes = self.universe_fn(ctx.asof)
        if self.spec.kind == "ew":
            return pd.Series(1.0 / len(codes), index=codes)
        if self.spec.kind == "vw":
            m = self.eng.panel.float_mcap.loc[ctx.asof, codes].fillna(0.0)
            return m / m.sum()
        r = self.eng.optimize(self.spec.model, self.spec.objective, self.spec.max_weight, self.spec.industry_cap,
                              universe=codes, asof=str(ctx.asof.date()), risk_aversion=self.spec.risk_aversion)
        return pd.Series({x["code"]: x["w"] for x in r["weights"]})


@dataclass
class Job:
    id: str
    params: dict
    status: str = "queued"            # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    result: dict | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None


class BacktestService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._eng: PortfolioEngine | None = None
        self._rf: pd.Series | None = None
        self._lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.cache: dict[str, str] = {}      # params hash -> job id

    # ------------------------------------------------------------------ data
    def engine(self) -> PortfolioEngine:
        with self._lock:
            if self._eng is None:
                t0 = time.perf_counter()
                store = _RecentBars(self.data_dir, pd.Timestamp("2012-01-01"))
                panel = load_panel(store)
                names = {}
                if panel.members is not None and "name" in panel.members:
                    m = panel.members.sort_values("snapshot")
                    names = m.drop_duplicates("code", keep="last").set_index("code")["name"].to_dict()
                self._eng = PortfolioEngine(panel=panel, names=names, dataset=self.data_dir.name, data_dir=self.data_dir)
                rf = DataStore(self.data_dir).read("curated", "rf_shibor3m.parquet")
                s = rf.set_index(pd.to_datetime(rf["date"]))["rate"].sort_index()
                self._rf = (s.reindex(panel.dates, method="ffill").fillna(s.iloc[0]) / 100 / 252)
                log.info("backtest panel loaded: %s in %.1fs", panel.ret.shape, time.perf_counter() - t0)
            return self._eng

    @property
    def ready(self) -> bool:
        return self._eng is not None

    # ------------------------------------------------------------------ jobs
    def submit(self, params: dict) -> Job:
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        if key in self.cache and self.jobs.get(self.cache[key]) and self.jobs[self.cache[key]].status in ("running", "done", "queued"):
            return self.jobs[self.cache[key]]
        job = Job(id=uuid.uuid4().hex[:12], params=params)
        self.jobs[job.id] = job
        self.cache[key] = job.id
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        # keep the most recent 30 jobs
        for jid in list(self.jobs)[:-30]:
            self.jobs.pop(jid, None)
        return job

    def _run(self, job: Job) -> None:
        job.status = "running"
        try:
            job.message = "加载研究数据…"
            eng = self.engine()
            job.result = self.run(eng, job.params, job)
            job.status, job.progress, job.message = "done", 1.0, "完成"
        except Exception as exc:  # noqa: BLE001
            log.exception("backtest failed")
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        job.finished = time.time()

    # ------------------------------------------------------------------ the backtest
    def run(self, eng: PortfolioEngine, p: dict, job: Job | None = None) -> dict:
        panel = eng.panel
        start = max(pd.Timestamp(p.get("start", "2016-01-01")), panel.dates[260 + 252])
        end = min(pd.Timestamp(p.get("end", "2025-12-31")), panel.dates[-1])
        freq = p.get("freq", "M")
        me = pd.Series(panel.dates[(panel.dates >= start) & (panel.dates <= end)])
        key = {"M": me.dt.to_period("M"), "W": me.dt.to_period("W"), "Q": me.dt.to_period("Q")}[freq]
        rebal = list(me.groupby(key).max())
        if len(rebal) < 3:
            raise ValueError("回测区间太短")
        top_n = int(p.get("top_n", 100))
        pool = [c for c in (p.get("universe") or []) if c in panel.codes]
        rule = UniverseRule(indices=("hs300",), top_n_by_mcap=top_n)

        def universe(t):
            if pool:
                ok, _ = eng.universe(t, pool)
                return ok
            return eligible(panel, t, rule)

        ex_p = p.get("execution", {})
        ex = Execution(timing=ex_p.get("timing", "next_open"), aum=float(ex_p.get("aum", 1e8)),
                       commission_bps=float(ex_p.get("commission_bps", 2.5)), min_commission=float(ex_p.get("min_commission", 5.0)),
                       impact_y=float(ex_p.get("impact_y", 0.5)),
                       participation_cap=ex_p.get("participation_cap", 0.10))
        specs = [StrategySpec(**{k: v for k, v in s.items() if k in StrategySpec.__dataclass_fields__}) for s in p.get("strategies", [])][:6]
        if p.get("include_ew", True):
            specs.append(StrategySpec(kind="ew"))
        bench = panel.market_ret.loc[rebal[0]:end].fillna(0.0) if panel.market_ret is not None else None

        results, metrics = {}, {}
        n = len(specs)
        for i, spec in enumerate(specs):
            strat = _OptStrategy(eng, spec, universe)

            def prog(done, total, i=i):
                if job:
                    job.progress = (i + done / total) / n
                    job.message = f"{spec.name()}：调仓 {done}/{total}"
            sim = simulate(panel, strat, rebal, ex, end=end, rf_daily=self._rf, on_rebalance=prog)
            r = sim.returns.loc[rebal[0]:end]
            m = full_metrics(r, bench, self._rf, sim.turnover, sim.costs)
            hold = [w.drop(CASH, errors="ignore") for w in sim.weights.values()]
            m["avg_holdings"] = float(np.mean([(h > 1e-4).sum() for h in hold])) if hold else 0
            m["avg_cash"] = float(np.mean([w.get(CASH, 0.0) for w in sim.weights.values()])) if hold else 0
            reasons = pd.Series([v for d in sim.blocked.values() for v in d.values()]).value_counts().to_dict()
            m["blocked_orders"] = reasons
            m["unfilled_avg"] = float(sim.unfilled.mean()) if len(sim.unfilled) else 0.0
            metrics[spec.name()] = m
            nav = (1 + r).cumprod()
            last = sim.weights[max(sim.weights)].drop(CASH, errors="ignore").sort_values(ascending=False)
            results[spec.name()] = {
                "spec": spec.__dict__, "nav": [[str(d.date()), round(float(v), 5)] for d, v in nav.items()],
                "drawdown": [[str(d.date()), round(float(v), 5)] for d, v in (nav / nav.cummax() - 1).items()],
                "monthly": {str(k): round(float(v), 5) for k, v in ((1 + r).groupby(r.index.to_period("M")).prod() - 1).items()},
                "turnover": [[str(d.date()), round(float(v), 4)] for d, v in sim.turnover.items()],
                "costs": {str(d.date()): {k: round(float(x), 7) for k, x in row.items()} for d, row in sim.costs.iterrows()},
                "last_weights": [{"code": c, "name": eng.names.get(c, c), "w": round(float(v), 5)} for c, v in last.head(15).items()],
                "_returns": r,
            }
        deflate(metrics)
        for k, m in metrics.items():
            results[k]["metrics"] = {a: b for a, b in m.items() if not a.startswith("_")}
        out = {"params": p, "rebalances": len(rebal), "start": str(rebal[0].date()), "end": str(end.date()),
               "freq": FREQ_CN.get(freq, freq), "execution": ex.__dict__, "dataset": eng.dataset,
               "universe": f"候选池 {len(pool)} 只" if pool else f"沪深300 历史成分股流通市值前 {top_n}",
               "strategies": results}
        if bench is not None:
            bnav = (1 + bench.loc[rebal[0]:end]).cumprod()
            out["benchmark"] = {"name": "沪深300", "nav": [[str(d.date()), round(float(v), 5)] for d, v in bnav.items()]}
        return out


def public(result: dict) -> dict:
    """Result without the raw return series (kept server-side for reports)."""
    return {**result, "strategies": {k: {a: b for a, b in v.items() if a != "_returns"} for k, v in result["strategies"].items()}}


def returns_csv(result: dict) -> str:
    df = pd.DataFrame({k: v["_returns"] for k, v in result["strategies"].items()})
    if "benchmark" in result:
        b = pd.Series({pd.Timestamp(d): v for d, v in result["benchmark"]["nav"]})
        df["沪深300（净值）"] = b
    df.index.name = "date"
    return df.to_csv(float_format="%.6f")
