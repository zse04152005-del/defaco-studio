"""Resumable ingestion jobs: raw downloads -> ``data/raw``; consolidation -> ``data/curated``.

Every step is idempotent: files that already exist are skipped unless ``refresh=True``, so a
job interrupted by a flaky network can simply be re-run.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

from .providers.akshare_provider import AkShareProvider
from .providers.baostock_provider import BaoStockProvider, month_ends
from .store import DataStore

log = logging.getLogger("quantkit.data")


def quarter_ends(start: str, end: str) -> list[str]:
    qs = pd.date_range(pd.Timestamp(start) - pd.DateOffset(months=3), end, freq="QE")
    return [d.strftime("%Y%m%d") for d in qs]


# ---------------------------------------------------------------------- BaoStock
def ingest_calendar(store: DataStore, bs: BaoStockProvider, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    if store.exists("raw", "trade_dates.parquet") and not refresh:
        return store.read("raw", "trade_dates.parquet")
    df = bs.trade_dates(start, end)
    store.write(df, "raw", "trade_dates.parquet")
    log.info("calendar: %d trading days", len(df))
    return df


def ingest_index_members(store: DataStore, bs: BaoStockProvider, index: str, start: str, end: str,
                         refresh: bool = False) -> pd.DataFrame:
    """Month-end snapshots of historical constituents."""
    part = ("raw", "index_members", f"{index}.parquet")
    done = store.read(*part) if store.exists(*part) and not refresh else pd.DataFrame()
    have = set(pd.to_datetime(done["snapshot"]).dt.strftime("%Y-%m-%d")) if not done.empty else set()
    cal = store.read("raw", "trade_dates.parquet")["date"]
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    frames = [done] if not done.empty else []
    for d in month_ends(cal):
        ds = d.strftime("%Y-%m-%d")
        if ds in have:
            continue
        frames.append(bs.index_members(index, ds))
        if len(frames) % 24 == 0:  # checkpoint every two years of snapshots
            store.write(pd.concat(frames, ignore_index=True), *part)
    out = pd.concat(frames, ignore_index=True) if frames else done
    store.write(out, *part)
    log.info("%s: %d snapshots, %d distinct codes", index, out["snapshot"].nunique(), out["code"].nunique())
    return out


BarFetcher = Callable[[str, str, str], pd.DataFrame]


def make_bar_fetcher(bs: BaoStockProvider | None, ak: AkShareProvider | None) -> BarFetcher:
    """BaoStock first (has ST/suspension/valuation fields), Sina via AkShare as fallback."""

    def fetch(code: str, start: str, end: str) -> pd.DataFrame:
        if bs is not None:
            try:
                df = bs.daily_bars(code, start, end)
                df["source"] = "baostock"
                return df
            except Exception as exc:  # noqa: BLE001
                if ak is None:
                    raise
                log.warning("baostock bars %s failed (%s); using sina", code, type(exc).__name__)
        if ak is None:
            raise RuntimeError("no bar provider available")
        df = ak.daily_bars_sina(code, start, end)
        df["source"] = "sina"
        return df

    return fetch


def ingest_bars(store: DataStore, fetch: BarFetcher, codes: list[str], start: str, end: str,
                refresh: bool = False, progress_every: int = 25) -> list[str]:
    """Download daily bars per stock; returns codes that failed (re-run to retry them)."""
    failed: list[str] = []
    todo = [c for c in codes if refresh or not store.exists("raw", "bars", f"{c}.parquet")]
    log.info("bars: %d to download (%d already present)", len(todo), len(codes) - len(todo))
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        try:
            df = fetch(code, start, end)
            if df.empty:
                df = pd.DataFrame({"code": pd.Series([], dtype=str)})
            store.write(df, "raw", "bars", f"{code}.parquet")
        except Exception as exc:  # noqa: BLE001
            log.error("bars %s failed: %s", code, exc)
            failed.append(code)
        if i % progress_every == 0:
            rate = (time.time() - t0) / i
            log.info("bars: %d/%d done, ~%.0f min left", i, len(todo), rate * (len(todo) - i) / 60)
    return failed


def ingest_stock_basic(store: DataStore, bs: BaoStockProvider) -> pd.DataFrame:
    df = bs.stock_basic()
    store.write(df, "raw", "stock_basic.parquet")
    return df


# ---------------------------------------------------------------------- AkShare
def ingest_industry(store: DataStore, ak: AkShareProvider, bs: BaoStockProvider | None = None,
                    refresh: bool = False) -> pd.DataFrame:
    if store.exists("raw", "industry.parquet") and not refresh:
        return store.read("raw", "industry.parquet")
    try:
        df = ak.sw_industry_history()
        df["source"] = "sw2021"
    except Exception as exc:  # noqa: BLE001
        if bs is None:
            raise
        log.warning("SW history unavailable (%s); falling back to current CSRC industries", exc)
        csrc = bs.industry_csrc()
        df = pd.DataFrame({"code": csrc["code"], "start": pd.Timestamp("1990-01-01"),
                           "sw_code": csrc["industry"], "sw_l1": csrc["industry"], "source": "csrc_current"})
    store.write(df, "raw", "industry.parquet")
    return df


def ingest_financials(store: DataStore, ak: AkShareProvider, start: str, end: str,
                      kinds: tuple[str, ...] = ("yjbb", "zcfz"), refresh: bool = False) -> list[str]:
    failed = []
    for kind in kinds:
        for period in quarter_ends(start, end):
            part = ("raw", "fin", kind, f"{period}.parquet")
            if store.exists(*part) and not refresh:
                continue
            try:
                store.write(ak.financial_cross_section(kind, period), *part)
            except Exception as exc:  # noqa: BLE001
                log.error("%s %s failed: %s", kind, period, exc)
                failed.append(f"{kind}:{period}")
    return failed


def ingest_disclosure_dates(store: DataStore, ak: AkShareProvider, start: str, end: str,
                            refresh: bool = False) -> list[str]:
    failed = []
    for period in quarter_ends(start, end):
        part = ("raw", "fin", "disclosure", f"{period}.parquet")
        if store.exists(*part) and not refresh:
            continue
        try:
            store.write(ak.disclosure_dates(period), *part)
        except Exception as exc:  # noqa: BLE001
            log.error("disclosure %s failed: %s", period, exc)
            failed.append(period)
    return failed


def legal_deadline(period: pd.Series) -> pd.Series:
    """Statutory latest disclosure date of a periodic report (CSRC rules)."""
    p = pd.to_datetime(period)
    y, m = p.dt.year, p.dt.month
    out = pd.to_datetime(pd.DataFrame({"year": y, "month": 4, "day": 30}))          # Q1
    out = out.where(m != 6, pd.to_datetime(pd.DataFrame({"year": y, "month": 8, "day": 31})))
    out = out.where(m != 9, pd.to_datetime(pd.DataFrame({"year": y, "month": 10, "day": 31})))
    out = out.where(m != 12, pd.to_datetime(pd.DataFrame({"year": y + 1, "month": 4, "day": 30})))
    return out


def point_in_time_dates(fin: pd.DataFrame, disclosure: pd.DataFrame | None) -> pd.DataFrame:
    """Replace Eastmoney's *latest-revision* date with the first-disclosure date.

    ``available = actual disclosure date`` when known, else ``min(eastmoney date, legal
    deadline)``; every candidate is on or after the true first disclosure, so no look-ahead.
    """
    fin = fin.rename(columns={"ann_date": "ann_date_em"})
    if disclosure is not None and not disclosure.empty:
        d = disclosure.dropna(subset=["disclosed"]).drop_duplicates(["code", "period"], keep="last")
        fin = fin.merge(d, on=["code", "period"], how="left")
    else:
        fin["disclosed"] = pd.NaT
    fallback = pd.concat([fin["ann_date_em"], legal_deadline(fin["period"])], axis=1).min(axis=1)
    fin["ann_date"] = fin["disclosed"].fillna(fallback)
    return fin


def ingest_index_daily(store: DataStore, ak: AkShareProvider, symbols: tuple[str, ...] = ("sh000300", "sh000905")) -> None:
    for s in symbols:
        store.write(ak.index_daily(s), "raw", "index_daily", f"{s}.parquet")


# ---------------------------------------------------------------------- consolidation
def consolidate(store: DataStore) -> dict[str, int]:
    """Merge per-stock / per-period raw files into curated tables."""
    sizes = {}
    bars = store.read_many("raw", "bars")
    if not bars.empty:
        bars = bars.sort_values(["code", "date"]).drop_duplicates(["code", "date"])
        store.write(bars, "curated", "bars.parquet")
        sizes["bars"] = len(bars)
    disclosure = store.read_many("raw", "fin", "disclosure")
    for kind in ("yjbb", "zcfz"):
        fin = store.read_many("raw", "fin", kind)
        if not fin.empty:
            fin = fin.sort_values(["code", "period", "ann_date"]).drop_duplicates(["code", "period"], keep="last")
            fin = point_in_time_dates(fin, disclosure)
            store.write(fin, "curated", f"fin_{kind}.parquet")
            sizes[f"fin_{kind}"] = len(fin)
    members = store.read_many("raw", "index_members")
    if not members.empty:
        store.write(members, "curated", "index_members.parquet")
        sizes["index_members"] = len(members)
    for name in ("industry", "trade_dates", "stock_basic"):
        if store.exists("raw", f"{name}.parquet"):
            df = store.read("raw", f"{name}.parquet")
            store.write(df, "curated", f"{name}.parquet")
            sizes[name] = len(df)
    idx = store.read_many("raw", "index_daily")
    if not idx.empty:
        store.write(idx, "curated", "index_daily.parquet")
        sizes["index_daily"] = len(idx)
    return sizes
