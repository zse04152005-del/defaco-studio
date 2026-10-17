"""Data-quality report for the curated tables (06 design doc, section 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantkit.config import load_settings
from quantkit.data.store import DataStore
from quantkit.pipeline import load_panel, month_ends


def main() -> int:
    cfg = load_settings()
    store = DataStore(cfg.data_dir)
    p = load_panel(store)
    print(f"panel: {p.dates[0].date()} .. {p.dates[-1].date()}, {len(p.dates)} days x {len(p.codes)} stocks")
    srcs = store.read("curated", "bars.parquet", columns=["source"])["source"].value_counts()
    print("bar sources:", srcs.to_dict())
    r = p.ret
    print(f"return coverage (trading cells): {r.notna().mean().mean():.3f}")
    big = (r.abs() > 0.21).sum().sum()
    print(f"|daily return| > 21%: {big} cells (expected only for IPO days / STAR / ChiNext >=2020-08-24)")
    st = p.is_st.notna().mean().mean()
    print(f"ST flag known for {st:.1%} of cells (Sina source has no ST flag)")
    me = month_ends(p.dates, cfg["data"]["universe_start"])
    hits = []
    for d in me[-24:]:
        up, dn = p.limit_flags(d)
        hits.append((up.sum(), dn.sum()))
    print("limit-up / limit-down closes on the last 24 month-ends:", hits)
    if p.members is not None:
        n = p.members.groupby("snapshot")["code"].nunique()
        print(f"member snapshots: {len(n)}, size range {n.min()}..{n.max()}, indices {sorted(p.members['index'].unique())}")
    if p.industry is not None:
        print(f"industry rows: {len(p.industry)}, level-1 groups now: {p.industry_at(p.dates[-1], list(p.codes)).nunique()}")
    if p.market_ret is not None:
        print(f"benchmark coverage: {p.market_ret.notna().mean():.3f}")
    for kind in ("yjbb", "zcfz"):
        if store.exists("curated", f"fin_{kind}.parquet"):
            f = store.read("curated", f"fin_{kind}.parquet")
            lag = (pd.to_datetime(f["ann_date"]) - pd.to_datetime(f["period"])).dt.days
            print(f"fin_{kind}: {len(f)} rows, {f['period'].nunique()} periods, ann lag median {lag.median():.0f}d, "
                  f"negative lags {(lag < 0).sum()}")
    if store.exists("curated", "characteristics_raw.parquet"):
        c = store.read("curated", "characteristics_raw.parquet")
        cov = c.drop(columns=["date", "code"]).notna().mean().sort_values()
        print("characteristic coverage (lowest 8):\n", cov.head(8).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
