"""Investable-universe filters (point-in-time)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data.panel import MarketPanel


@dataclass
class UniverseRule:
    indices: tuple[str, ...] | None = ("hs300",)
    window_days: int = 250
    min_trading_ratio: float = 0.8
    min_listing_days: int = 252
    exclude_st: bool = True
    require_trading_today: bool = True
    top_n_by_mcap: int | None = None   # e.g. 100 to mirror the benchmark paper


def eligible(panel: MarketPanel, asof: pd.Timestamp, rule: UniverseRule) -> list[str]:
    """Codes investable on ``asof`` using only information available that day."""
    i = panel.loc_date(asof)
    d = panel.dates[i]
    codes = panel.members_at(d, rule.indices) if rule.indices else list(panel.codes)
    if not codes:
        return []
    codes = pd.Index(codes)

    # listed for at least min_listing_days trading days
    first = panel.first_trade_date().reindex(codes)
    first_pos = pd.Series(panel.dates.searchsorted(first.fillna(d)), index=codes)
    keep = (i - first_pos) >= rule.min_listing_days

    win = panel.window(d, rule.window_days)
    trade_ratio = (panel.tradestatus.loc[win, codes] > 0).mean()
    keep &= trade_ratio >= rule.min_trading_ratio

    if rule.require_trading_today:
        keep &= panel.tradestatus.loc[d, codes] > 0
    if rule.exclude_st:
        keep &= ~(panel.is_st.loc[d, codes].fillna(0) > 0)

    out = codes[keep.values]
    if rule.top_n_by_mcap:
        mcap = panel.float_mcap.loc[d, out].dropna()
        out = mcap.sort_values(ascending=False).index[: rule.top_n_by_mcap]
    return sorted(out)
