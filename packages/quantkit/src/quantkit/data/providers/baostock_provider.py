"""BaoStock provider: trading calendar, historical index constituents, daily bars.

BaoStock is the primary source because it offers *historical* CSI300 / CSI500
constituents (no survivorship bias) together with ST flags, suspension status,
turnover and valuation ratios in one daily query.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import logging

import pandas as pd

from ..codes import to_baostock, to_canonical
from .base import ensure_socket_timeout, hard_timeout, retry

log = logging.getLogger("quantkit.data")

BAR_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,"
              "pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM")
NUMERIC = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg",
           "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
INDEX_QUERIES = {"hs300": "query_hs300_stocks", "zz500": "query_zz500_stocks", "sz50": "query_sz50_stocks"}


QUERY_TIMEOUT = 180  # seconds per BaoStock request (a full 17-year history takes ~10s)


def _collect(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


class BaoStockProvider:
    """Use as a context manager so that login/logout always pair up."""

    def __init__(self) -> None:
        import baostock as bs  # imported lazily: optional at runtime

        self.bs = bs
        self._logged_in = False

    def __enter__(self) -> "BaoStockProvider":
        ensure_socket_timeout()
        self.login()
        return self

    def __exit__(self, *exc) -> None:
        self.logout()

    @retry()
    def login(self) -> None:
        with contextlib.redirect_stdout(None), hard_timeout(QUERY_TIMEOUT, "baostock login"):
            lg = self.bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        self._logged_in = True

    def logout(self) -> None:
        if self._logged_in:
            with contextlib.redirect_stdout(None):
                self.bs.logout()
            self._logged_in = False

    def _relogin(self) -> None:
        try:
            self.logout()
        except Exception:  # noqa: BLE001 - the old session may already be dead
            self._logged_in = False
        self.login()

    # ------------------------------------------------------------------
    @retry()
    def trade_dates(self, start: str, end: str) -> pd.DataFrame:
        with hard_timeout(QUERY_TIMEOUT, "trade_dates"):
            df = _collect(self.bs.query_trade_dates(start_date=start, end_date=end))
        df["calendar_date"] = pd.to_datetime(df["calendar_date"])
        df["is_trading_day"] = df["is_trading_day"].astype(int).astype(bool)
        return df[df["is_trading_day"]][["calendar_date"]].rename(columns={"calendar_date": "date"})

    @retry()
    def index_members(self, index: str, date: str) -> pd.DataFrame:
        """Constituents of ``index`` (hs300/zz500/sz50) effective on ``date``."""
        fn = getattr(self.bs, INDEX_QUERIES[index])
        try:
            with hard_timeout(QUERY_TIMEOUT, f"{index} members"):
                df = _collect(fn(date=date))
        except Exception:
            self._relogin()
            raise
        if df.empty:
            return pd.DataFrame(columns=["snapshot", "code", "name", "index"])
        out = pd.DataFrame({
            "snapshot": pd.to_datetime(date),
            "code": df["code"].map(to_canonical),
            "name": df["code_name"],
            "index": index,
        })
        return out

    @retry()
    def stock_basic(self) -> pd.DataFrame:
        with hard_timeout(QUERY_TIMEOUT, "stock_basic"):
            df = _collect(self.bs.query_stock_basic())
        df["code"] = df["code"].map(to_canonical)
        for c in ("ipoDate", "outDate"):
            df[c] = pd.to_datetime(df[c].replace("", None))
        return df.rename(columns={"code_name": "name", "ipoDate": "ipo_date", "outDate": "out_date"})

    @retry()
    def industry_csrc(self) -> pd.DataFrame:
        """Current CSRC industry classification (fallback when SW history is unavailable)."""
        with hard_timeout(QUERY_TIMEOUT, "stock_industry"):
            df = _collect(self.bs.query_stock_industry())
        df["code"] = df["code"].map(to_canonical)
        return df

    def daily_bars(self, code: str, start: str, end: str, chunk_years: int = 20) -> pd.DataFrame:
        """Unadjusted daily bars. ``pctChg`` is computed by BaoStock against the exchange's
        ex-rights reference price (``preclose``), so it is a total return including
        dividends and splits; we use it as the return series."""
        frames = []
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        while s <= e:
            ce = min(e, s + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1))
            frames.append(self._bars_chunk(code, s.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
            s = ce + pd.Timedelta(days=1)
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].map(to_canonical)
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c].replace("", None), errors="coerce")
        df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(0).astype("int8")
        df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype("int8")
        df["ret"] = df["pctChg"] / 100.0
        return df.drop(columns=["pctChg"])

    @retry(times=5, wait=3.0)
    def _bars_chunk(self, code: str, start: str, end: str) -> pd.DataFrame:
        try:
            with hard_timeout(QUERY_TIMEOUT, f"bars {code}"):
                rs = self.bs.query_history_k_data_plus(to_baostock(code), BAR_FIELDS, start_date=start,
                                                       end_date=end, frequency="d", adjustflag="3")
                return _collect(rs)
        except Exception:
            self._relogin()  # the BaoStock session silently expires on long jobs
            raise


def month_ends(trade_dates: pd.Series) -> pd.DatetimeIndex:
    """Last trading day of every month."""
    s = pd.Series(pd.to_datetime(trade_dates)).sort_values()
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().values)


def today() -> str:
    return dt.date.today().strftime("%Y-%m-%d")
