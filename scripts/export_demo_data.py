"""Export static JSON for the web demo (until the FastAPI service exists).

    DEFACO_DATA=data_smoke python scripts/export_demo_data.py

Writes web/public/demo-data/*.json. Every file carries a ``meta`` block stating the dataset,
so the UI can show whether numbers come from the smoke (survivorship-biased) data.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantkit.backtest.costs import CostModel
from quantkit.backtest.engine import RebalanceContext, run_backtest
from quantkit.backtest.strategies import EqualWeight, MinVariance
from quantkit.config import load_settings
from quantkit.data.store import DataStore
from quantkit.data.sw import sw_name
from quantkit.features.characteristics import ALL_CHARS
from quantkit.metrics.performance import monthly_sd_sharpe, performance, weight_stats
from quantkit.pipeline import load_panel, month_ends
from quantkit.riskmodels import build
from quantkit.universe import UniverseRule, eligible

log = logging.getLogger("export")
MODELS = {"lw_nonlinear": "非线性收缩 NLS", "lw_linear": "线性收缩 LW", "pca_factor": "PCA 因子模型",
          "sample": "样本协方差", "ewma": "EWMA"}


def clean(x):
    """JSON-safe conversion (NaN -> None, numpy -> python)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    return x


def dump(out: Path, name: str, obj) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, separators=(",", ":"))
    log.info("wrote %s (%.0f KB)", name, (out / f"{name}.json").stat().st_size / 1024)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = load_settings()
    store = DataStore(cfg.data_dir)
    out = cfg.root / "web" / "public" / "demo-data"
    smoke = "smoke" in cfg.data_dir.name
    meta = {"dataset": cfg.data_dir.name, "survivorship_biased": smoke,
            "note": "演示数据：当前沪深300成分股回溯，存在幸存者偏差，仅用于界面演示" if smoke else "研究数据集（历史成分股）"}

    p = load_panel(store)
    asof = p.dates[-1]
    members = p.members
    names = members.drop_duplicates("code").set_index("code")["name"].to_dict() if "name" in members else {}
    codes = list(p.codes)
    ind = p.industry_at(asof, codes)
    ind_name = ind.map(sw_name)

    # ------------------------------------------------------------------ securities list
    mcap = p.float_mcap.loc[asof]
    close = p.close.loc[asof]
    win = p.window(asof, 21)
    ret_1m = (1 + p.ret.loc[win].fillna(0)).prod() - 1
    secs = [{"code": c, "name": names.get(c, c), "industry": ind_name[c], "mcap": float(mcap.get(c, np.nan)) / 1e8,
             "close": float(close.get(c, np.nan)), "ret_1m": float(ret_1m.get(c, np.nan)),
             "board": "STAR" if c.startswith("688") else "main"} for c in codes]
    dump(out, "securities", {"meta": meta, "asof": asof, "items": secs})

    # screener table: latest month-end percentiles of every characteristic
    cn = store.read("curated", "characteristics_norm.parquet")
    cn = cn[cn["date"] == cn["date"].max()].set_index("code")
    rows = []
    for c in codes:
        if c not in cn.index:
            continue
        rows.append({"code": c, "name": names.get(c, c), "industry": ind_name[c], "mcap": float(mcap.get(c, np.nan)) / 1e8,
                     "ret_1m": float(ret_1m.get(c, np.nan)),
                     **{k: round(float(cn.loc[c, k]) + 0.5, 4) for k in ALL_CHARS}})
    dump(out, "screener", {"meta": meta, "asof": pd.Timestamp(cn["date"].iloc[0]).strftime("%Y-%m-%d"),
                           "factors": ALL_CHARS, "items": rows})

    # ------------------------------------------------------------------ market overview
    idx = store.read("curated", "index_daily.parquet")
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx[idx["date"] >= asof - pd.DateOffset(years=2)]
    indices = {s: g[["date", "close"]].values.tolist() for s, g in idx.groupby("symbol")}
    last_ret = p.ret.loc[asof]
    amount = p.amount.loc[asof]
    sector = pd.DataFrame({"ret": last_ret, "amount": amount, "ind": ind_name}).dropna()
    by_ind = sector.groupby("ind").apply(lambda g: pd.Series({
        "ret": np.average(g["ret"], weights=g["amount"]), "amount": g["amount"].sum() / 1e8, "n": len(g)}),
        include_groups=False).reset_index()
    breadth = {"up": int((last_ret > 0).sum()), "down": int((last_ret < 0).sum()), "flat": int((last_ret == 0).sum()),
               "hist": np.histogram(last_ret.dropna().clip(-0.1, 0.1), bins=20, range=(-0.1, 0.1))[0].tolist()}
    rv = p.market_ret.rolling(21).std() * np.sqrt(252)
    regime = rv.dropna().iloc[-500:]
    dump(out, "overview", {"meta": meta, "asof": asof,
                           "indices": {k: [[d.strftime("%Y-%m-%d"), v] for d, v in vals] for k, vals in indices.items()},
                           "industries": by_ind.to_dict("records"), "breadth": breadth,
                           "vol_regime": [[d.strftime("%Y-%m-%d"), v] for d, v in regime.items()],
                           "vol_threshold": float(rv.quantile(0.75))})

    # ------------------------------------------------------------------ per-stock K-lines
    bars = store.read("curated", "bars.parquet")
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars[bars["date"] >= asof - pd.DateOffset(years=3)]
    trading = p.tradestatus.loc[asof] > 0
    top = mcap[trading].dropna().sort_values(ascending=False).index[:80]
    chars = store.read("curated", "characteristics_norm.parquet")
    chars = chars[chars["date"] == chars["date"].max()].set_index("code")
    for c in top:
        b = bars[bars["code"] == c].sort_values("date")
        b = b[b["tradestatus"] > 0]
        # forward-adjusted prices from total returns, anchored at the latest close
        adj = (1 + b["ret"].fillna(0)).cumprod()
        f = b["close"].iloc[-1] / adj.iloc[-1] * adj / b["close"]
        k = pd.DataFrame({"t": b["date"].dt.strftime("%Y-%m-%d"),
                          "o": (b["open"] * f).round(3), "h": (b["high"] * f).round(3),
                          "l": (b["low"] * f).round(3), "c": (b["close"] * f).round(3),
                          "v": b["volume"], "a": b["amount"]})
        prof = {ch: float(chars.loc[c, ch]) if c in chars.index else 0.0 for ch in ALL_CHARS}
        dump(out / "stocks", c, {"meta": meta, "code": c, "name": names.get(c, c), "industry": ind_name[c],
                                 "mcap": float(mcap[c]) / 1e8, "bars": k.values.tolist(), "profile": prof})

    # ------------------------------------------------------------------ portfolio build (latest month-end)
    rule = UniverseRule(indices=tuple(members["index"].unique()), top_n_by_mcap=100)
    me = month_ends(p.dates)
    t = me[-1]
    uni = eligible(p, t, rule)
    ctx = RebalanceContext(asof=t, panel=p, drift_weights=pd.Series(dtype=float), suspended=set(), no_buy=set(), no_sell=set())
    port = {}
    for m, label in MODELS.items():
        strat = MinVariance(lambda m=m: build(m), rule, max_weight=0.10, keep_covariance=True)
        w = strat.target_weights(ctx)
        cov = strat.last_diagnostics["cov"]
        S = cov.as_frame()
        wv = w.reindex(S.index).fillna(0.0)
        tot = float(wv @ S.values @ wv)
        mrc = (S.values @ wv.values) * wv.values / tot          # risk contributions (sum to 1)
        rc = pd.Series(mrc, index=S.index)
        risk = {"total_vol": math.sqrt(tot * 252)}
        if cov.is_factor:
            fac = float(wv @ (cov.B @ cov.sigma_f @ cov.B.T) @ wv)
            risk.update({"factor_share": fac / tot, "idio_share": 1 - fac / tot})
            risk["exposures"] = (cov.B.T @ wv.values).tolist()
        top_w = w[w > 1e-4].sort_values(ascending=False)
        port[m] = {"label": label, "weights": [{"code": c, "name": names.get(c, c), "industry": ind_name.get(c, "NA"),
                                                  "w": float(v), "rc": float(rc.get(c, 0.0))} for c, v in top_w.items()],
                   "industry_weights": top_w.groupby(ind_name.reindex(top_w.index).fillna("NA")).sum().sort_values(ascending=False).to_dict(),
                   "risk": risk, "n_holdings": int(len(top_w)), "hhi": float((top_w ** 2).sum()),
                   "explain": strat.last_diagnostics.get("model", {})}
    # correlation of the NLS covariance for the 40 largest holdings, ordered by industry
    nls = MinVariance(lambda: build("lw_nonlinear"), rule, keep_covariance=True)
    nls.target_weights(ctx)
    S = nls.last_diagnostics["cov"].as_frame()
    sel = pd.Series(port["lw_nonlinear"]["weights"][:40]).map(lambda d: d["code"]).tolist() if port["lw_nonlinear"]["weights"] else []
    sel = sorted(sel, key=lambda c: (ind_name.get(c, ""), c))
    sd = np.sqrt(np.diag(S.loc[sel, sel]))
    corr = S.loc[sel, sel].values / np.outer(sd, sd)
    dump(out, "portfolio", {"meta": meta, "asof": t, "universe_size": len(uni), "models": port,
                            "corr": {"codes": sel, "names": [names.get(c, c) for c in sel],
                                     "industries": [ind_name.get(c, "") for c in sel], "matrix": corr.round(3).tolist()}})
    try:   # keep the precomputed DeFaCo portfolio (thesis M4) in the offline fallback
        from export_defaco import add_to_portfolio_json
        add_to_portfolio_json(out / "portfolio.json", data_dir=cfg.data_dir)
    except Exception as exc:  # noqa: BLE001  (models not exported yet, etc.)
        log.warning("DeFaCo not added to portfolio.json: %s", exc)

    # ------------------------------------------------------------------ backtests (2016 -> latest)
    rebal = list(month_ends(p.dates, "2016-01-01"))
    results = {}
    strats = [("EW", "等权 EW", EqualWeight(rule))] + \
             [(m, MODELS[m], MinVariance(lambda m=m: build(m), rule, max_weight=0.10)) for m in ("lw_nonlinear", "pca_factor", "sample")]
    for key, label, s in strats:
        res = run_backtest(p, s, rebal, costs=CostModel())
        nav = res.nav
        perf, ms, ws = performance(res.returns_net), monthly_sd_sharpe(res.returns_net), weight_stats(res.weights, res.turnover)
        monthly = (1 + res.returns_net).groupby(res.returns_net.index.to_period("M")).prod() - 1
        dd = nav / nav.cummax() - 1
        ind_w = {d.strftime("%Y-%m"): w.groupby(ind_name.reindex(w.index).fillna("其他")).sum().to_dict()
                 for d, w in list(res.weights.items())[::3]}
        results[key] = {"label": label, "nav": [[d.strftime("%Y-%m-%d"), v] for d, v in nav.iloc[::2].items()],
                        "drawdown": [[d.strftime("%Y-%m-%d"), v] for d, v in dd.iloc[::2].items()],
                        "monthly": {str(k): v for k, v in monthly.items()},
                        "metrics": {**perf, **ms, **ws, "avg_cost_bps": float(res.costs.mean() * 1e4)},
                        "industry_weights": ind_w}
        log.info("backtest %s done", key)
    dump(out, "backtest", {"meta": meta, "start": rebal[0], "end": p.dates[-1], "results": results,
                           "costs": "佣金 2bp/边 + 卖出印花税（2023-08-28 前 10bp，之后 5bp）"})
    return 0


if __name__ == "__main__":
    os.environ.setdefault("DEFACO_DATA", "data_smoke")
    raise SystemExit(main())
