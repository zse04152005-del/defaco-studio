"""High-level helpers shared by research scripts and the demo API."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .data.panel import MarketPanel
from .data.store import DataStore
from .features.characteristics import ALL_CHARS, compute_characteristics
from .features.transforms import rank_normalize

log = logging.getLogger("quantkit")


def load_panel(store: DataStore, benchmark: str = "sh000300") -> MarketPanel:
    bars = store.read("curated", "bars.parquet")
    cal = store.read("curated", "trade_dates.parquet")["date"] if store.exists("curated", "trade_dates.parquet") else None
    members = store.read("curated", "index_members.parquet") if store.exists("curated", "index_members.parquet") else None
    industry = store.read("curated", "industry.parquet") if store.exists("curated", "industry.parquet") else None
    market = None
    if store.exists("curated", "index_daily.parquet"):
        idx = store.read("curated", "index_daily.parquet")
        idx = idx[idx["symbol"] == benchmark]
        market = idx.set_index(pd.to_datetime(idx["date"]))["ret"]
    if cal is not None:
        cal = pd.to_datetime(cal)
        cal = cal[(cal >= bars["date"].min()) & (cal <= bars["date"].max())]
    return MarketPanel.from_long(bars, calendar=cal, members=members, industry=industry, market_ret=market)


def month_ends(dates: pd.DatetimeIndex, start: str | None = None) -> pd.DatetimeIndex:
    s = pd.Series(pd.DatetimeIndex(dates))
    me = pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().values)
    return me[me >= pd.Timestamp(start)] if start else me


def _ttm(fin: pd.DataFrame, col: str) -> pd.DataFrame:
    """Trailing-twelve-month value of a year-to-date per-share item (e.g. EPS) per report.

    TTM(P) = YTD(P) + Annual(Y-1) - YTD(P - 1y) for Q1-Q3; = YTD(P) for Q4. The result is
    stamped with P's announcement date, by which all inputs had already been published.
    """
    f = fin[["code", "period", "ann_date", col]].dropna(subset=[col]).copy()
    f["period"] = pd.to_datetime(f["period"])
    key = f.set_index(["code", "period"])[col]
    prev_year = f["period"] - pd.DateOffset(years=1)
    prev_annual = pd.to_datetime((f["period"].dt.year - 1).astype(str) + "-12-31")
    a = key.reindex(pd.MultiIndex.from_arrays([f["code"], prev_annual])).to_numpy()
    b = key.reindex(pd.MultiIndex.from_arrays([f["code"], prev_year + pd.offsets.QuarterEnd(0)])).to_numpy()
    is_q4 = f["period"].dt.month == 12
    f[f"{col}_ttm"] = np.where(is_q4, f[col], f[col] + a - b)
    return f


def fundamental_valuation(panel: MarketPanel, me: pd.DatetimeIndex, yjbb: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """EP / BM / CFP from point-in-time reports, used when the bar source has no ratios."""
    from .features.characteristics import _pit_align

    if yjbb is None or yjbb.empty:
        return {}
    eps = _ttm(yjbb, "eps")
    cfps = _ttm(yjbb, "cfps")
    fin = yjbb.merge(eps[["code", "period", "eps_ttm"]], on=["code", "period"], how="left") \
              .merge(cfps[["code", "period", "cfps_ttm"]], on=["code", "period"], how="left")
    al = _pit_align(fin, ["eps_ttm", "cfps_ttm", "bvps"], me, panel.codes)
    px = panel.close.reindex(me)
    return {"ep": al["eps_ttm"] / px, "bm": al["bvps"] / px, "cfp": al["cfps_ttm"] / px}


def build_characteristics(store: DataStore, panel: MarketPanel, start: str) -> pd.DataFrame:
    me = month_ends(panel.dates, start)
    yjbb = store.read("curated", "fin_yjbb.parquet") if store.exists("curated", "fin_yjbb.parquet") else None
    zcfz = store.read("curated", "fin_zcfz.parquet") if store.exists("curated", "fin_zcfz.parquet") else None
    raw = compute_characteristics(panel, me, yjbb, zcfz)
    if not panel.valuation or all(v.isna().all().all() for v in panel.valuation.values()):
        val = fundamental_valuation(panel, me, yjbb)
        for name, wide in val.items():
            s = wide.stack(future_stack=True)
            s.index.names = ["date", "code"]
            raw = raw.drop(columns=[name]).merge(s.rename(name).reset_index(), on=["date", "code"], how="left")
        raw = raw[["date", "code", *ALL_CHARS]]
    raw = raw.replace([np.inf, -np.inf], np.nan)
    coverage = raw[ALL_CHARS].notna().mean().round(3)
    log.info("characteristic coverage:\n%s", coverage.to_string())
    return raw


def normalized_characteristics(raw: pd.DataFrame, universe_by_date: dict[pd.Timestamp, list[str]] | None = None) -> pd.DataFrame:
    """Rank-normalise within each month's investable universe (if given) or all stocks."""
    if universe_by_date:
        keep = pd.concat([pd.DataFrame({"date": d, "code": c}) for d, c in universe_by_date.items()])
        raw = raw.merge(keep, on=["date", "code"])
    return rank_normalize(raw, by="date", cols=ALL_CHARS)
