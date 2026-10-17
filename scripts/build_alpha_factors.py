"""Compute the formulaic factor library (quantkit.features.alphas: Alpha158 / Alpha101 / GTJA191)
on the research universe and store month-end cross-sections.

    python scripts/build_alpha_factors.py        # -> data/curated/factors_alpha.parquet

Daily inputs from 2009 are used so that 250-day windows are filled by the first month end (2010-01).
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd

from quantkit.config import load_settings
from quantkit.data.store import DataStore
from quantkit.features.alphas import CATALOG, Inputs, compute
from quantkit.data.sw import sw_name
from quantkit.pipeline import load_panel, month_ends
from quantkit.universe import UniverseRule, eligible


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("alphas")
    cfg = load_settings()
    store = DataStore(cfg.data_dir)
    t0 = time.time()
    bars = store.read("curated", "bars.parquet", columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "ret", "tradestatus"])
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.drop_duplicates(["date", "code"], keep="last")
    dates = pd.DatetimeIndex(sorted(bars["date"].unique()))
    codes = pd.Index(sorted(bars["code"].unique()))
    log.info("bars %s: %d dates x %d codes (%.0fs)", bars.shape, len(dates), len(codes), time.time() - t0)
    inp = Inputs.from_bars(bars, dates, codes)
    del bars
    me = month_ends(dates, cfg["data"]["universe_start"])
    rows = np.searchsorted(dates, me)
    log.info("inputs ready (%.0fs); computing %d factors at %d month ends", time.time() - t0, len(CATALOG), len(rows))
    done = []
    vals = compute(inp, rows=rows, on_progress=lambda n: done.append(n) or (len(done) % 10 == 0 and log.info("  %d/%d (%s) %.0fs", len(done), len(CATALOG), n, time.time() - t0)))
    idx = pd.MultiIndex.from_product([me, codes], names=["date", "code"])
    df = pd.DataFrame({k: v.reshape(-1).astype(np.float32) for k, v in vals.items()}, index=idx).reset_index()
    df = df[df[list(vals)].notna().any(axis=1)]
    store.write(df, "curated", "factors_alpha.parquet")
    cov = df[list(vals)].notna().mean()
    log.info("wrote factors_alpha.parquet: %s, median coverage %.2f, lowest %s (%.0fs)", df.shape, cov.median(),
             cov.nsmallest(3).round(2).to_dict(), time.time() - t0)
    build_base(store, log)
    return 0


def build_base(store: DataStore, log) -> None:
    """Month-end base table for factor research: universe flags, industry, size, forward returns."""
    t0 = time.time()
    panel = load_panel(store)
    me = month_ends(panel.dates, "2010-01-01")
    pos = panel.dates.searchsorted(me)
    growth = (1 + panel.ret.fillna(0.0)).cumprod()
    lvl = growth.iloc[pos]                                  # total-return index at month ends
    rows = []
    rule300 = UniverseRule(indices=("hs300",), window_days=60, min_listing_days=252)
    rule800 = UniverseRule(indices=("hs300", "zz500"), window_days=60, min_listing_days=252)
    for k, d in enumerate(me):
        e800 = set(eligible(panel, d, rule800))
        if not e800:
            continue
        e300 = set(eligible(panel, d, rule300))
        codes = sorted(e800)
        ind = panel.industry_at(d, codes).map(lambda x: "NA" if x == "NA" else sw_name(x))
        row = {"date": d, "code": codes, "in300": [c in e300 for c in codes], "industry": ind.values,
               "log_mcap": np.log(panel.float_mcap.loc[d, codes].astype(float)).values}
        for h in (1, 3, 6):
            if k + h < len(me):
                row[f"fwd{h}"] = (lvl.iloc[k + h][codes] / lvl.iloc[k][codes] - 1).values
            else:
                row[f"fwd{h}"] = np.nan
        rows.append(pd.DataFrame(row))
    base = pd.concat(rows, ignore_index=True)
    store.write(base, "curated", "factor_base.parquet")
    log.info("wrote factor_base.parquet: %s, %d months, avg %.0f stocks/month (%.0fs)", base.shape,
             base["date"].nunique(), base.groupby("date").size().mean(), time.time() - t0)


if __name__ == "__main__":
    raise SystemExit(main())
