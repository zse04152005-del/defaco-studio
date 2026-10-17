"""make_chart: ECharts options built server-side from engine data, so the numbers drawn are the
engine's, not the model's. Only `bar` / `line` take values from the model; those are checked
against the tool results like any number in an answer.

Each builder returns (option, summary). The option goes to the browser as a `chart` event; the
LLM only sees the summary (so it can refer to "图 1" and quote its key numbers).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .analysis import daily_bars, name_of, resolve_codes, returns_frame
from .tools import FACTOR_CN, _json, _resolve, _screener

RADAR = ["size", "bm", "ep", "roe", "mom_12_2", "vol_12m", "beta_12m", "turn_1m"]
KINDS = {
    "price_compare": "多只股票的累计收益走势（起点归一为 0%）",
    "kline": "单只股票的日 K 线 + MA20",
    "factor_radar": "1–4 只股票的因子分位雷达图",
    "correlation": "多只股票近 250 日收益相关性热力图",
    "industry_pie": "行业分布：source=watchlist/pool/codes 按只数，source=holdings/portfolio 按权重",
    "weights_bar": "权重与风险贡献：source=holdings 为持仓，否则为当前实时组合",
    "backtest_nav": "回测净值曲线",
    "bar": "自定义柱状图（labels + series，数值须来自工具结果）",
    "line": "自定义折线图（labels + series，数值须来自工具结果）",
}


class ChartError(ValueError):
    pass


def _title(t: str) -> dict:
    return {"text": t, "left": 0, "top": 0, "textStyle": {"fontSize": 13, "fontWeight": 600}}


async def build_chart(box, kind: str, codes: list[str] | None = None, source: str | None = None,
                      days: int = 250, model: str | None = None, models: list[str] | None = None,
                      title: str | None = None, labels: list[str] | None = None,
                      series: list[dict] | None = None, unit: str = "") -> tuple[dict, dict]:
    if kind not in KINDS:
        raise ChartError(f"未知图表类型 {kind}，可选：{', '.join(KINDS)}")
    if source and not codes:
        codes, _ = resolve_codes(box.ctx, source, None)
    codes = [c for c in (_resolve(x) for x in (codes or [])) if c]
    days = int(max(20, min(days or 250, 750)))

    if kind == "price_compare":
        codes = codes[:8]
        if not codes:
            raise ChartError("需要 codes 或 source")
        bars = await daily_bars(box.md, codes, 3 if days > 250 else 1)
        px = pd.DataFrame({c: pd.Series({b[0]: float(b[4]) for b in v}) for c, v in bars.items() if v}).sort_index().tail(days).ffill()
        cum = px / px.bfill().iloc[0] - 1
        opt = {"title": _title(title or f"累计收益对比（近 {len(cum)} 个交易日）"), "tooltip": {"trigger": "axis", "valueFormatter": "pct"},
               "legend": {"top": 22, "type": "scroll"}, "grid": {"left": 48, "right": 12, "top": 56, "bottom": 28},
               "xAxis": {"type": "category", "data": list(cum.index), "boundaryGap": False},
               "yAxis": {"type": "value", "axisLabel": {"formatter": "pct"}},
               "series": [{"type": "line", "name": name_of(c), "showSymbol": False, "smooth": True,
                           "data": [None if pd.isna(v) else round(float(v), 4) for v in cum[c]]} for c in cum.columns]}
        return opt, {"区间": f"{cum.index[0]} → {cum.index[-1]}",
                     "区间累计收益": {name_of(c): round(float(cum[c].iloc[-1]), 4) for c in cum.columns}}

    if kind == "kline":
        if not codes:
            raise ChartError("需要 codes（1 只）")
        c = codes[0]
        bars = (await daily_bars(box.md, [c], 1 if days <= 250 else 3)).get(c, [])[-days:]
        if not bars:
            raise ChartError(f"{c} 暂无 K 线")
        close = pd.Series([b[4] for b in bars])
        ma = close.rolling(20).mean()
        opt = {"title": _title(title or f"{name_of(c)} 日 K 线（近 {len(bars)} 日）"), "tooltip": {"trigger": "axis"},
               "legend": {"top": 22, "data": ["K 线", "MA20"]}, "grid": {"left": 48, "right": 12, "top": 56, "bottom": 28},
               "xAxis": {"type": "category", "data": [b[0] for b in bars]}, "yAxis": {"type": "value", "scale": True},
               "series": [{"type": "candlestick", "name": "K 线", "data": [[b[1], b[4], b[3], b[2]] for b in bars]},
                          {"type": "line", "name": "MA20", "showSymbol": False, "data": [None if pd.isna(v) else round(float(v), 3) for v in ma]}]}
        return opt, {"股票": name_of(c), "最新收盘": bars[-1][4], "区间最高": max(b[2] for b in bars), "区间最低": min(b[3] for b in bars)}

    if kind == "factor_radar":
        scr = _screener()
        codes = [c for c in codes if c in scr.index][:4]
        if not codes:
            raise ChartError("这些股票不在研究股票池中，没有因子数据")
        opt = {"title": _title(title or "因子分位雷达（0–100，截至 2025-12-31）"), "tooltip": {}, "legend": {"top": 22},
               "radar": {"indicator": [{"name": FACTOR_CN[f].split("（")[0], "min": 0, "max": 100} for f in RADAR],
                         "radius": "62%", "center": ["50%", "58%"]},
               "series": [{"type": "radar", "data": [{"name": name_of(c), "value": [round(float(scr.loc[c, f]) * 100) for f in RADAR],
                                                      "areaStyle": {"opacity": 0.15}} for c in codes]}]}
        return opt, {name_of(c): {FACTOR_CN[f]: round(float(scr.loc[c, f]), 3) for f in RADAR} for c in codes}

    if kind == "correlation":
        codes = codes[:15]
        if len(codes) < 2:
            raise ChartError("相关性需要至少 2 只股票")
        R = returns_frame(await daily_bars(box.md, codes, 1), days)
        C = R.corr(min_periods=60)
        names = [name_of(c) for c in C.columns]
        data = [[j, i, None if pd.isna(C.iat[i, j]) else round(float(C.iat[i, j]), 2)] for i in range(len(C)) for j in range(len(C))]
        opt = {"title": _title(title or f"收益相关性（近 {len(R)} 日）"), "tooltip": {"position": "top"},
               "grid": {"left": 70, "right": 12, "top": 36, "bottom": 70},
               "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 45, "fontSize": 10}},
               "yAxis": {"type": "category", "data": names, "axisLabel": {"fontSize": 10}},
               "visualMap": {"min": -0.2, "max": 1, "calculable": False, "orient": "horizontal", "left": "center", "bottom": 0,
                             "itemHeight": 120, "show": False},
               "series": [{"type": "heatmap", "data": data, "label": {"show": len(C) <= 8, "fontSize": 10}}]}
        off = C.where(~np.eye(len(C), dtype=bool)).stack()
        return opt, {"平均相关": round(float(off.mean()), 3), "最高": round(float(off.max()), 3), "最低": round(float(off.min()), 3)}

    if kind == "industry_pie":
        # weights for holdings / the live portfolio; head counts for a list of stocks
        if source in ("holdings", "portfolio") or (model and not codes):
            from .analysis import analyze_holdings, get_live_portfolio
            if source == "holdings":
                h = await analyze_holdings(box, model or "lw_nonlinear")
                weights = (h.get("risk") or {}).get("industry_weights") if isinstance(h.get("risk"), dict) else None
                if not weights:
                    raise ChartError(h.get("error") or "持仓行业权重不可用")
            else:
                p = await get_live_portfolio(box, model=model)
                if "error" in p:
                    raise ChartError(p["error"])
                weights = p["industry_weights"]
            data = [{"name": k, "value": round(v * 100, 2)} for k, v in weights.items()]
            what = "持仓" if source == "holdings" else "组合"
        else:
            if not codes:
                raise ChartError("需要 source（watchlist / pool / holdings）或 codes")
            sec = _json("securities.json")
            ind = {s["code"]: s["industry"] for s in sec["items"]}
            counts = pd.Series([ind.get(c, "未知") for c in codes]).value_counts()
            data = [{"name": k, "value": int(v)} for k, v in counts.items()]
            what = {"watchlist": "自选股", "pool": "候选池"}.get(source or "", "股票")
        opt = {"title": _title(title or f"{what}行业分布"), "tooltip": {"trigger": "item"},
               "legend": {"type": "scroll", "orient": "vertical", "right": 0, "top": 28},
               "series": [{"type": "pie", "radius": ["38%", "66%"], "center": ["40%", "56%"], "data": data,
                           "label": {"formatter": "{b} {d}%", "fontSize": 10}}]}
        return opt, {"行业分布": {d["name"]: d["value"] for d in data}, "单位": "%" if what in ("持仓", "组合") else "只"}

    if kind == "weights_bar":
        from .analysis import analyze_holdings, get_live_portfolio
        if source == "holdings":
            h = await analyze_holdings(box, model or "lw_nonlinear")
            rc = (h.get("risk") or {}).get("risk_contributions") if isinstance(h.get("risk"), dict) else None
            if not rc:
                raise ChartError(h.get("error") or "持仓风险贡献不可用")
            rows = sorted(rc.items(), key=lambda kv: -kv[1]["w"])[:15]
            names, w, r = [k for k, _ in rows], [v["w"] for _, v in rows], [v["rc"] for _, v in rows]
            what = "持仓"
        else:
            p = await get_live_portfolio(box, model=model)
            if "error" in p:
                raise ChartError(p["error"])
            rows = p["top_holdings"][:15]
            names, w, r = [x["name"] for x in rows], [x["w"] for x in rows], [x["rc"] for x in rows]
            what = p["label"]
        opt = {"title": _title(title or f"{what}：权重与风险贡献"), "tooltip": {"trigger": "axis", "valueFormatter": "pct"},
               "legend": {"top": 22}, "grid": {"left": 70, "right": 16, "top": 52, "bottom": 20},
               "xAxis": {"type": "value", "axisLabel": {"formatter": "pct"}},
               "yAxis": {"type": "category", "data": names[::-1], "axisLabel": {"fontSize": 10}},
               "series": [{"type": "bar", "name": "权重", "data": [round(x, 4) for x in w[::-1]]},
                          {"type": "bar", "name": "风险贡献", "data": [round(x, 4) for x in r[::-1]]}]}
        return opt, {"权重": dict(zip(names, w)), "风险贡献": dict(zip(names, r))}

    if kind == "backtest_nav":
        B = _json("backtest.json")["results"]
        models = [m for m in (models or list(B)) if m in B]
        dates = [d for d, _ in B[models[0]]["nav"]]
        opt = {"title": _title(title or "回测净值（2016–2025，扣除交易成本）"), "tooltip": {"trigger": "axis"},
               "legend": {"top": 22}, "grid": {"left": 48, "right": 12, "top": 56, "bottom": 28},
               "xAxis": {"type": "category", "data": dates, "boundaryGap": False}, "yAxis": {"type": "value", "scale": True},
               "series": [{"type": "line", "name": B[m]["label"], "showSymbol": False, "data": [round(v, 4) for _, v in B[m]["nav"]]} for m in models]}
        return opt, {B[m]["label"]: {"期末净值": B[m]["nav"][-1][1], "年化收益": B[m]["metrics"]["ann_return"],
                                     "最大回撤": B[m]["metrics"]["max_drawdown"]} for m in models}

    # bar / line from model-supplied numbers (checked by the guard)
    if not labels or not series:
        raise ChartError("bar/line 需要 labels 和 series（[{name, values}]）")
    ser = [{"type": kind, "name": s.get("name", ""), "data": [None if v is None else float(v) for v in s.get("values", [])][:len(labels)],
            **({"smooth": True, "showSymbol": False} if kind == "line" else {})} for s in series[:6]]
    opt = {"title": _title(title or ""), "tooltip": {"trigger": "axis"}, "legend": {"top": 22},
           "grid": {"left": 52, "right": 12, "top": 56, "bottom": 40},
           "xAxis": {"type": "category", "data": [str(x) for x in labels][:60], "axisLabel": {"fontSize": 10, "rotate": 30 if len(labels) > 8 else 0}},
           "yAxis": {"type": "value", "name": unit}, "series": ser}
    return opt, {"custom_values": {s["name"]: s["data"] for s in ser}}
