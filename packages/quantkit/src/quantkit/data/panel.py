"""MarketPanel: wide (date x code) matrices built from long daily bars.

All research code and the demo read market data through this object, so look-ahead rules
live in one place: anything queried "as of" a date only uses rows dated on or before it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .codes import CHINEXT_REFORM, board_of


@dataclass
class MarketPanel:
    dates: pd.DatetimeIndex
    codes: pd.Index
    ret: pd.DataFrame            # total return; NaN when not trading
    close: pd.DataFrame
    preclose: pd.DataFrame
    amount: pd.DataFrame
    turn: pd.DataFrame           # percent
    tradestatus: pd.DataFrame    # 1 trading, 0 suspended / no data
    is_st: pd.DataFrame          # 1/0, NaN unknown
    float_mcap: pd.DataFrame
    valuation: dict[str, pd.DataFrame] = field(default_factory=dict)   # peTTM, pbMRQ, psTTM, pcfNcfTTM
    members: pd.DataFrame | None = None      # snapshot, code, index
    industry: pd.DataFrame | None = None     # code, start, sw_l1
    market_ret: pd.Series | None = None      # benchmark index daily return
    open: pd.DataFrame | None = None         # unadjusted open (same basis as close / preclose)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_long(cls, bars: pd.DataFrame, calendar: pd.Series | pd.DatetimeIndex | None = None,
                  members: pd.DataFrame | None = None, industry: pd.DataFrame | None = None,
                  market_ret: pd.Series | None = None) -> "MarketPanel":
        bars = bars.copy()
        bars["date"] = pd.to_datetime(bars["date"])
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(calendar))) if calendar is not None \
            else pd.DatetimeIndex(sorted(bars["date"].unique()))
        codes = pd.Index(sorted(bars["code"].unique()), name="code")

        def wide(col: str) -> pd.DataFrame:
            if col not in bars:
                return pd.DataFrame(np.nan, index=dates, columns=codes)
            w = bars.pivot_table(index="date", columns="code", values=col, aggfunc="last")
            return w.reindex(index=dates, columns=codes).astype(float)

        tradestatus = wide("tradestatus").fillna(0.0)
        ret = wide("ret").where(tradestatus > 0)
        raw_close = wide("close")
        # carry prices through suspensions, but never past a stock's last bar (delisting)
        alive = raw_close.notna().cumsum().where(raw_close[::-1].notna().cumsum()[::-1] > 0) > 0
        close = raw_close.ffill().where(alive)
        if "float_shares" in bars and bars["float_shares"].notna().any():
            shares = wide("float_shares")
        else:  # BaoStock: turnover% = volume / float shares
            vol, turn = wide("volume"), wide("turn")
            shares = (vol / (turn / 100.0)).where(turn > 0)
        float_mcap = (close * shares.ffill()).where(alive)
        valuation = {c: wide(c) for c in ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM") if c in bars}
        panel = cls(dates=dates, codes=codes, ret=ret, close=close, preclose=wide("preclose"), open=wide("open").where(tradestatus > 0),
                    amount=wide("amount"), turn=wide("turn"), tradestatus=tradestatus,
                    is_st=wide("isST"), float_mcap=float_mcap, valuation=valuation,
                    members=members, industry=industry,
                    market_ret=market_ret.reindex(dates) if market_ret is not None else None)
        return panel

    def subset(self, codes: list[str], start: pd.Timestamp | str) -> "MarketPanel":
        """Smaller panel (for multiprocessing): the given codes and dates from ``start``."""
        codes = pd.Index(sorted(set(codes) & set(self.codes)), name="code")
        d = self.dates[self.dates >= pd.Timestamp(start)]
        cut = lambda df: df.loc[d, codes]  # noqa: E731
        return MarketPanel(dates=d, codes=codes, ret=cut(self.ret), close=cut(self.close), preclose=cut(self.preclose),
                           amount=cut(self.amount), turn=cut(self.turn), tradestatus=cut(self.tradestatus),
                           is_st=cut(self.is_st), float_mcap=cut(self.float_mcap),
                           valuation={k: cut(v) for k, v in self.valuation.items()},
                           members=self.members[self.members["code"].isin(codes)] if self.members is not None else None,
                           industry=self.industry[self.industry["code"].isin(codes)] if self.industry is not None else None,
                           market_ret=self.market_ret.loc[d] if self.market_ret is not None else None,
                           open=cut(self.open) if self.open is not None else None)

    # ------------------------------------------------------------------ queries
    def loc_date(self, asof: pd.Timestamp) -> int:
        """Index of the last trading day <= asof."""
        i = self.dates.searchsorted(pd.Timestamp(asof), side="right") - 1
        if i < 0:
            raise ValueError(f"{asof} precedes the panel")
        return int(i)

    def window(self, asof: pd.Timestamp, days: int) -> pd.DatetimeIndex:
        i = self.loc_date(asof)
        return self.dates[max(0, i - days + 1): i + 1]

    def forward(self, asof: pd.Timestamp, days: int) -> pd.DatetimeIndex:
        i = self.loc_date(asof)
        return self.dates[i + 1: i + 1 + days]

    def first_trade_date(self) -> pd.Series:
        return self.ret.notna().idxmax().where(self.ret.notna().any())

    def members_at(self, asof: pd.Timestamp, indices: tuple[str, ...] | None = None) -> list[str]:
        """Constituents from the latest month-end snapshot on or before ``asof``."""
        if self.members is None:
            return list(self.codes)
        m = self.members
        if indices:
            m = m[m["index"].isin(indices)]
        snaps = pd.to_datetime(m["snapshot"])
        valid = snaps[snaps <= pd.Timestamp(asof)]
        if valid.empty:
            return []
        snap = valid.max()
        return sorted(set(m.loc[snaps == snap, "code"]) & set(self.codes))

    def industry_at(self, asof: pd.Timestamp, codes: list[str]) -> pd.Series:
        """Level-1 industry id effective on ``asof`` ('NA' when unknown)."""
        if self.industry is None:
            return pd.Series("NA", index=codes)
        ind = self.industry[pd.to_datetime(self.industry["start"]) <= pd.Timestamp(asof)]
        latest = ind.sort_values("start").groupby("code")["sw_l1"].last()
        return latest.reindex(codes).fillna("NA")

    # ------------------------------------------------------------------ trading constraints
    def limit_pct(self, asof: pd.Timestamp) -> pd.Series:
        """Daily price-limit fraction per code on ``asof``: 10% main board, 20% STAR / ChiNext
        (after the 2020-08-24 reform), 30% BSE, 5% for ST names on the main board."""
        d = self.dates[self.loc_date(asof)]
        boards = pd.Series([board_of(c) for c in self.codes], index=self.codes)
        st = self.is_st.loc[d].fillna(0) > 0
        pct = pd.Series(0.10, index=self.codes)
        pct[st] = 0.05
        pct[boards == "STAR"] = 0.20
        if d.date() >= CHINEXT_REFORM:
            pct[boards == "ChiNext"] = 0.20
        pct[boards == "BSE"] = 0.30
        return pct

    def limit_prices(self, asof: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
        d = self.dates[self.loc_date(asof)]
        pre, pct = self.preclose.loc[d], self.limit_pct(d)
        return (pre * (1 + pct)).round(2), (pre * (1 - pct)).round(2)

    def limit_flags(self, asof: pd.Timestamp, at: str = "close") -> tuple[pd.Series, pd.Series]:
        """(at limit-up, at limit-down) on ``asof`` for every code, judged at the close or the open."""
        d = self.dates[self.loc_date(asof)]
        px = self.close.loc[d] if at == "close" or self.open is None else self.open.loc[d]
        up, dn = self.limit_prices(d)
        trading = self.tradestatus.loc[d] > 0
        return (px >= up - 0.0051) & trading, (px <= dn + 0.0051) & trading
