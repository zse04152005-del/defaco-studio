"""Batch analysis tools: the user's watchlist / candidate pool / holdings, and live portfolios.

The browser sends its lists in the request context (codes; holdings with share counts). Share
counts never reach the LLM: only the derived weights do (privacy rule in 06 §9).
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd

from .tools import FACTOR_CN, _all_securities, _json, _resolve, _screener, _securities, technicals

KEY_FACTORS = ["size", "bm", "ep", "roe", "mom_12_2", "vol_12m", "beta_12m", "turn_1m"]
SOURCE_CN = {"watchlist": "自选股", "pool": "候选池", "holdings": "持仓", "codes": "指定股票"}
MAX_STOCKS = 20


def resolve_codes(ctx: dict | None, source: str, codes: list[str] | None) -> tuple[list[str], str]:
    ctx = ctx or {}
    if source == "watchlist":
        raw = ctx.get("watch") or []
    elif source == "pool":
        raw = ctx.get("pool") or []
    elif source == "holdings":
        raw = [h["code"] for h in ctx.get("holdings") or []]
    else:
        raw = codes or []
    out = []
    for c in raw:
        r = _resolve(str(c))
        if r and r not in out:
            out.append(r)
    return out[:MAX_STOCKS], SOURCE_CN.get(source, source)


async def daily_bars(md, codes: list[str], years: int = 1) -> dict[str, list]:
    res = await asyncio.gather(*(md.get_kline(c, years) for c in codes), return_exceptions=True)
    return {c: (r.get("bars") or []) for c, r in zip(codes, res) if isinstance(r, dict)}


def returns_frame(bars: dict[str, list], days: int = 250) -> pd.DataFrame:
    cols = {c: pd.Series({b[0]: float(b[4]) for b in v}) for c, v in bars.items() if len(v) > 20}
    if not cols:
        return pd.DataFrame()
    px = pd.DataFrame(cols).sort_index()
    return px.pct_change().iloc[1:].tail(days)


def name_of(code: str) -> str:
    sec = _all_securities()
    return str(sec.loc[code, "name"]) if code in sec.index else code


def industry_of(code: str) -> str:
    sec = _all_securities()
    return str(sec.loc[code, "industry"]) if code in sec.index else ""


async def analyze_stocks(box, source: str = "watchlist", codes: list[str] | None = None) -> dict:
    cl, label = resolve_codes(box.ctx, source, codes)
    if not cl:
        return {"error": f"{label}为空或代码无法识别", "hint": "用户可以在个股页点 ☆ 加入自选，或点「加入候选池」"}
    quotes, bars = await asyncio.gather(box.md.get_quotes(cl), daily_bars(box.md, cl, 2))
    scr = _screener()
    held = {}
    for m, p in _json("portfolio.json")["models"].items():
        for w in p["weights"]:
            held.setdefault(w["code"], {})[p["label"]] = w["w"]
    rows = []
    for c in cl:
        q = quotes.get(c, {})
        t = technicals(bars[c]) if len(bars.get(c, [])) > 30 else {}
        row: dict[str, Any] = {"code": c, "name": q.get("name") or name_of(c),
                               "industry": industry_of(c),
                               "price": q.get("price"), "pct_today": q.get("pct"), "pe": q.get("pe"), "pb": q.get("pb")}
        for k in ("ret_20d", "ret_60d", "ret_250d", "ann_vol_60d", "max_drawdown_250d", "rsi14", "above_ma60", "pos_in_250d_range"):
            if k in t:
                row[k] = t[k]
        if c in scr.index:
            row["factor_pct"] = {FACTOR_CN[f]: float(scr.loc[c, f]) for f in KEY_FACTORS if pd.notna(scr.loc[c, f])}
        if c in held:
            row["model_weights"] = held[c]
        rows.append(row)
    out: dict[str, Any] = {"source": label, "n": len(rows), "stocks": rows,
                           "industry_counts": pd.Series([r["industry"] or "未知" for r in rows]).value_counts().to_dict(),
                           "note": "ret_*/ann_vol/max_drawdown/pct_today 为小数；factor_pct 为截面分位（0–1，截至 2025-12-31）；model_weights 为各模型组合中的权重"}
    R = returns_frame(bars)
    if R.shape[1] >= 2:
        C = R.corr(min_periods=60)
        pairs = [(C.columns[i], C.columns[j], C.iat[i, j]) for i in range(len(C)) for j in range(i + 1, len(C)) if pd.notna(C.iat[i, j])]
        pairs.sort(key=lambda x: -x[2])
        fmt = lambda p: {"a": name_of(p[0]), "b": name_of(p[1]), "corr": round(float(p[2]), 3)}  # noqa: E731
        out["correlation_250d"] = {"average": round(float(np.mean([p[2] for p in pairs])), 3) if pairs else None,
                                   "most_correlated": [fmt(p) for p in pairs[:3]],
                                   "least_correlated": [fmt(p) for p in pairs[-3:][::-1]]}
        ew = R.mean(axis=1)
        out["equal_weight_group"] = {"ann_vol": float(ew.std() * np.sqrt(252)),
                                     "avg_single_ann_vol": float((R.std() * np.sqrt(252)).mean()),
                                     "diversification_note": "等权组合波动 / 单股平均波动 越低，分散效果越好"}
    return out


def _holding_weights(box) -> tuple[pd.Series, list[str]]:
    hold = [(h["code"], float(h.get("shares", 0))) for h in (box.ctx or {}).get("holdings") or [] if h.get("shares")]
    return pd.Series({c: s for c, s in hold}), [c for c, _ in hold]


async def analyze_holdings(box, model: str = "lw_nonlinear") -> dict:
    shares, codes = _holding_weights(box)
    if shares.empty:
        return {"error": "用户还没有录入持仓（在「持仓调仓」页录入）"}
    q = await box.md.get_quotes(codes)
    sec = _securities()
    px = pd.Series({c: (q.get(c, {}).get("price") or (sec.loc[c, "close"] if c in sec.index else np.nan)) for c in codes}, dtype=float)
    value = (shares * px).dropna()
    w = value / value.sum()
    out: dict[str, Any] = {
        "n": int(len(w)), "weights_by_market_value": {name_of(c): round(float(v), 4) for c, v in w.sort_values(ascending=False).items()},
        "top1": float(w.max()), "top3": float(w.nlargest(3).sum()), "hhi": float((w ** 2).sum()),
        "note": "权重按实时价 × 股数计算；不向模型透露股数与金额",
    }
    pf = box.pf
    if pf is None:
        out["risk"] = "研究面板尚未加载，暂无风险估计"
        return out
    asof = pf.panel.dates[-1]
    ok, dropped = pf.universe(asof, list(w.index))
    if len(ok) < 2:
        out["risk"] = f"可用于风险估计的股票不足 2 只（剔除 {len(dropped)} 只：上市不足一年 / 停牌 / 不在研究数据中）"
        return out
    from ..portfolio import MODELS, describe

    wv = w.reindex(ok).fillna(0.0)
    wv = wv / wv.sum()
    cov, explain = await asyncio.to_thread(pf.covariance, model, asof, ok)
    ind = pf.industries(asof, ok)
    d = describe(cov, wv, ind, pf.names, MODELS.get(model, model), explain, corr_top=0)
    out["risk"] = {"model": MODELS.get(model, model), "asof": asof.strftime("%Y-%m-%d"),
                   "ann_vol": d["risk"]["total_vol"], "factor_share": d["risk"].get("factor_share"),
                   "risk_contributions": {x["name"]: {"w": x["w"], "rc": x["rc"]} for x in d["weights"]},
                   "industry_weights": d["industry_weights"],
                   "excluded": [name_of(c) for c in dropped],
                   "note": "协方差用研究数据（截至 2025-12-31）的 250 日窗口估计；rc = 风险贡献占比"}
    try:
        mv = await asyncio.to_thread(pf.optimize, model, "minvar", 0.4, None, ok)
        out["min_variance_reference"] = {
            "ann_vol": mv["risk"]["total_vol"],
            "weights": {x["name"]: x["w"] for x in mv["weights"]},
            "note": "同一批股票、单股上限 40% 的最小方差组合，仅作风险对照，不是交易建议"}
    except Exception as exc:  # noqa: BLE001
        out["min_variance_reference"] = f"无法求解：{exc}"
    return out


async def get_live_portfolio(box, model: str | None = None, max_weight: float | None = None,
                             industry_cap: float | None = None, use_pool: bool | None = None) -> dict:
    pf = box.pf
    if pf is None:
        return {"error": "组合引擎尚未加载，可改用 get_portfolio（预计算结果）"}
    s = (box.ctx or {}).get("portfolio_settings") or {}
    model = model or s.get("model") or "pca_factor"
    max_weight = max_weight or s.get("max_weight") or 0.10
    industry_cap = industry_cap if industry_cap is not None else s.get("industry_cap")
    use_pool = s.get("use_pool", False) if use_pool is None else use_pool
    universe = (box.ctx or {}).get("pool") if use_pool else None
    from ..portfolio import OptimizeError

    try:
        r = await asyncio.to_thread(pf.optimize, model, "minvar", float(max_weight), industry_cap, universe or None)
    except OptimizeError as exc:
        return {"error": str(exc)}
    return {"label": r["label"], "asof": r["asof"], "universe": "候选池" if universe else "沪深300 市值前 100",
            "universe_size": r["universe_size"], "constraints": r["constraints"], "n_holdings": r["n_holdings"],
            "hhi": r["hhi"], "risk": {k: v for k, v in r["risk"].items() if k != "exposures"},
            "top_holdings": r["weights"][:12], "industry_weights": r["industry_weights"],
            "dropped": [name_of(c) for c in r.get("dropped", [])],
            "note": "实时求解的多头最小方差组合；w = 权重，rc = 风险贡献占比；total_vol 为年化"}
