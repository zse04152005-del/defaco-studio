"""Compute monthly point-in-time characteristics -> data/curated/characteristics_{raw,norm}.parquet"""
from __future__ import annotations

import logging
import sys

from quantkit.config import load_settings
from quantkit.data.store import DataStore
from quantkit.pipeline import build_characteristics, load_panel, normalized_characteristics


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    cfg = load_settings()
    store = DataStore(cfg.data_dir)
    panel = load_panel(store)
    logging.info("panel: %d days x %d stocks", len(panel.dates), len(panel.codes))
    raw = build_characteristics(store, panel, cfg["data"]["universe_start"])
    store.write(raw, "curated", "characteristics_raw.parquet")
    store.write(normalized_characteristics(raw), "curated", "characteristics_norm.parquet")
    logging.info("characteristics: %d rows, %d months", len(raw), raw["date"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
