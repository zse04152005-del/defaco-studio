"""Download and consolidate A-share data.

Examples
--------
Full research dataset (historical constituents, needs BaoStock)::

    python scripts/ingest.py --steps calendar members bars industry financials index consolidate

Smoke-test dataset when BaoStock is unavailable (current CSI300 constituents, Sina bars;
**survivorship-biased, never use for paper results**)::

    python scripts/ingest.py --members current --bar-source sina --steps members bars industry financials index consolidate

Every step is resumable: re-run the same command after an interruption.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from quantkit.config import load_settings
from quantkit.data import ingest
from quantkit.data.providers.akshare_provider import AkShareProvider
from quantkit.data.providers.baostock_provider import BaoStockProvider
from quantkit.data.store import DataStore

STEPS = ["calendar", "basic", "members", "bars", "industry", "financials", "index", "consolidate"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--members", choices=["baostock", "current"], default="baostock")
    ap.add_argument("--bar-source", choices=["auto", "baostock", "sina"], default="auto")
    ap.add_argument("--limit", type=int, default=None, help="only the first N codes (debugging)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--config", default=None, help="config yaml (default configs/default.yaml)")
    ap.add_argument("--shard", default=None, help="k/n: download only codes[k::n] (run n processes in parallel)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("ingest")
    cfg = load_settings(args.config)
    store = DataStore(cfg.data_dir)
    start, end, ustart = cfg["data"]["start"], cfg["data"]["end"], cfg["data"]["universe_start"]
    ak = AkShareProvider()
    need_bs = args.members == "baostock" or args.bar_source in ("auto", "baostock") or "calendar" in args.steps or "basic" in args.steps
    bs = BaoStockProvider() if need_bs else None
    if bs is not None:
        bs.__enter__()

    try:
        if "calendar" in args.steps:
            if bs is not None:
                ingest.ingest_calendar(store, bs, start, end, refresh=args.refresh)

        if "basic" in args.steps:   # every A-share security ever listed (names, IPO / delisting dates)
            df = ingest.ingest_stock_basic(store, bs)
            log.info("stock_basic: %d securities", len(df))
            try:
                bj = ak.bse_stock_list()
                store.write(bj, "raw", "stock_basic_bj.parquet")
                log.info("stock_basic_bj: %d BSE stocks", len(bj))
            except Exception as exc:  # noqa: BLE001
                log.warning("BSE list unavailable: %s", exc)

        if "members" in args.steps:
            if args.members == "baostock":
                for idx in cfg["data"]["indices"]:
                    ingest.ingest_index_members(store, bs, idx, ustart, end, refresh=args.refresh)
            else:
                log.warning("using CURRENT CSI300 constituents: survivorship-biased smoke-test data")
                store.write(ak.current_index_members("000300"), "raw", "index_members", "csi300_current.parquet")

        codes: list[str] = []
        members = store.read_many("raw", "index_members")
        if not members.empty:
            codes = sorted(members["code"].unique())
        if args.limit:
            codes = codes[: args.limit]
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            codes = codes[k::n]
            log.info("shard %d/%d: %d codes", k, n, len(codes))

        if "bars" in args.steps:
            fetch = ingest.make_bar_fetcher(bs if args.bar_source != "sina" else None,
                                            ak if args.bar_source != "baostock" else None)
            failed = ingest.ingest_bars(store, fetch, codes, start, end, refresh=args.refresh)
            if failed:
                log.warning("%d codes failed; re-run to retry: %s", len(failed), failed[:20])

        if "industry" in args.steps:
            try:
                ingest.ingest_industry(store, ak, bs, refresh=args.refresh)
            except Exception as exc:  # noqa: BLE001
                log.error("industry classification unavailable: %s", exc)

        if "financials" in args.steps:
            failed = ingest.ingest_financials(store, ak, start, end, refresh=args.refresh)
            failed += ingest.ingest_disclosure_dates(store, ak, start, end, refresh=args.refresh)
            if failed:
                log.warning("financial periods failed: %s", failed)

        if "index" in args.steps:
            ingest.ingest_index_daily(store, ak)

        if "consolidate" in args.steps:
            if not store.exists("raw", "trade_dates.parquet"):
                idx = store.read_many("raw", "index_daily")
                if not idx.empty:  # calendar fallback: CSI300 trading days
                    cal = pd.DataFrame({"date": sorted(pd.to_datetime(idx["date"]).unique())})
                    cal = cal[(cal["date"] >= start) & (cal["date"] <= end)]
                    store.write(cal, "raw", "trade_dates.parquet")
            log.info("curated tables: %s", ingest.consolidate(store))
    finally:
        if bs is not None:
            bs.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
