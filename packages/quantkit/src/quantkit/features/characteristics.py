"""Monthly firm characteristics (point-in-time) for characteristic-instrumented loadings.

Each characteristic at month-end ``t`` uses only data dated on or before ``t``.
Financial-statement items are aligned on the *announcement date* (``ann_date``), never on
the report period, so quarterly reports only enter after they were published.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import MarketPanel

PRICE_CHARS = ["size", "mom_12_2", "mom_6_1", "rev_1m", "vol_1m", "vol_12m", "maxret_1m",
               "turn_1m", "turn_6m", "abturn", "illiq", "beta_12m", "ivol_3m", "high52"]
VALUE_CHARS = ["ep", "bm", "sp", "cfp"]
FUND_CHARS = ["roe", "gross_margin", "revenue_yoy", "profit_yoy", "asset_growth", "leverage"]
ALL_CHARS = PRICE_CHARS + VALUE_CHARS + FUND_CHARS


def _rolling_beta_ivol(r: pd.DataFrame, m: pd.Series, window: int, min_periods: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling CAPM beta and idiosyncratic volatility, vectorised over stocks."""
    mm = pd.DataFrame(np.repeat(m.values[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns)
    mm = mm.where(r.notna())  # use the market only on days the stock traded
    roll = dict(window=window, min_periods=min_periods)
    e_r, e_m = r.rolling(**roll).mean(), mm.rolling(**roll).mean()
    cov = (r * mm).rolling(**roll).mean() - e_r * e_m
    var_m = (mm * mm).rolling(**roll).mean() - e_m ** 2
    var_r = (r * r).rolling(**roll).mean() - e_r ** 2
    beta = cov / var_m
    ivol = np.sqrt((var_r - cov ** 2 / var_m).clip(lower=0))
    return beta, ivol


def price_characteristics(panel: MarketPanel, month_ends: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    r = panel.ret
    lr = np.log1p(r.fillna(0.0)).where(r.notna())
    traded = r.notna().astype(float)
    def rsum(x, w, mp):
        return x.rolling(w, min_periods=mp).sum()

    out: dict[str, pd.DataFrame] = {}
    out["size"] = np.log(panel.float_mcap.where(panel.float_mcap > 0))
    out["mom_12_2"] = rsum(lr, 231, 150).shift(21)          # t-252 .. t-21
    out["mom_6_1"] = rsum(lr, 105, 70).shift(21)            # t-126 .. t-21
    out["rev_1m"] = rsum(lr, 21, 15)
    out["vol_1m"] = r.rolling(21, min_periods=15).std()
    out["vol_12m"] = r.rolling(252, min_periods=150).std()
    out["maxret_1m"] = r.rolling(21, min_periods=15).max()
    turn = panel.turn.where(traded > 0)
    out["turn_1m"] = turn.rolling(21, min_periods=15).mean()
    out["turn_6m"] = turn.rolling(126, min_periods=80).mean()
    out["abturn"] = out["turn_1m"] / out["turn_6m"]
    out["illiq"] = (r.abs() / panel.amount.where(panel.amount > 0)).rolling(21, min_periods=15).mean() * 1e8
    market = panel.market_ret if panel.market_ret is not None else r.mean(axis=1)
    out["beta_12m"], _ = _rolling_beta_ivol(r, market.fillna(0.0), 252, 150)
    _, out["ivol_3m"] = _rolling_beta_ivol(r, market.fillna(0.0), 63, 40)
    px = np.exp(lr.fillna(0.0).cumsum())                    # total-return price index
    enough = traded.rolling(252, min_periods=1).sum() >= 150  # pre-listing days are not prices
    out["high52"] = (px / px.rolling(252, min_periods=150).max()).where(enough)
    return {k: v.reindex(month_ends) for k, v in out.items()}


def value_characteristics(panel: MarketPanel, month_ends: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    def inv(name):
        x = panel.valuation.get(name)
        if x is None:
            return pd.DataFrame(np.nan, index=month_ends, columns=panel.codes)
        x = x.ffill(limit=20).reindex(month_ends)
        return 1.0 / x.where(x != 0)
    return {"ep": inv("peTTM"), "bm": inv("pbMRQ"), "sp": inv("psTTM"), "cfp": inv("pcfNcfTTM")}


def _pit_align(fin: pd.DataFrame, cols: list[str], month_ends: pd.DatetimeIndex, codes: pd.Index) -> dict[str, pd.DataFrame]:
    """For each month-end, the latest report *announced* on or before it."""
    out = {c: pd.DataFrame(np.nan, index=month_ends, columns=codes) for c in cols}
    if fin is None or fin.empty:
        return out
    fin = fin[fin["code"].isin(codes)].dropna(subset=["ann_date"]).copy()
    fin["ann_date"] = pd.to_datetime(fin["ann_date"])
    # if a period was announced more than once keep the latest (conservative)
    fin = fin.sort_values(["code", "ann_date", "period"])
    grid = pd.DataFrame({"date": month_ends})
    for code, g in fin.groupby("code", sort=False):
        g = g.sort_values("ann_date")
        # among reports announced by each date, prefer the most recent report period
        g = g.assign(best_period=g["period"].cummax())
        g = g[g["period"] == g["best_period"]]
        m = pd.merge_asof(grid, g, left_on="date", right_on="ann_date", direction="backward")
        for c in cols:
            out[c].loc[:, code] = m[c].values
    return out


def fundamental_characteristics(month_ends: pd.DatetimeIndex, codes: pd.Index,
                                yjbb: pd.DataFrame | None, zcfz: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    a = _pit_align(yjbb, ["roe", "gross_margin", "revenue_yoy", "profit_yoy"], month_ends, codes)
    b = _pit_align(zcfz, ["assets_yoy", "debt_ratio"], month_ends, codes)
    return {**a, "asset_growth": b["assets_yoy"], "leverage": b["debt_ratio"]}


def compute_characteristics(panel: MarketPanel, month_ends: pd.DatetimeIndex,
                            yjbb: pd.DataFrame | None = None, zcfz: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table: one row per (date, code) with raw (un-normalised) characteristics."""
    month_ends = pd.DatetimeIndex(month_ends)
    blocks = {**price_characteristics(panel, month_ends), **value_characteristics(panel, month_ends),
              **fundamental_characteristics(month_ends, panel.codes, yjbb, zcfz)}
    long = pd.concat({k: v.stack(future_stack=True) for k, v in blocks.items()}, axis=1)
    long.index.names = ["date", "code"]
    long = long.reset_index()
    long = long.replace([np.inf, -np.inf], np.nan)
    return long[["date", "code", *ALL_CHARS]]
