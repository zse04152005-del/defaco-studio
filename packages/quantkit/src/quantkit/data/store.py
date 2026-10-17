"""Parquet-backed dataset store with DuckDB views.

Layout under ``data_dir``::

    raw/bars/{code}.parquet          one file per stock (resume-friendly ingestion)
    raw/index_members/{index}.parquet
    raw/industry.parquet
    raw/fin/{kind}/{period}.parquet  cross-sectional financial reports
    raw/index_daily/{symbol}.parquet
    raw/trade_dates.parquet
    curated/{name}.parquet           consolidated tables used by research and the demo
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class DataStore:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- paths ----------
    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, *parts: str) -> bool:
        return self.root.joinpath(*parts).exists()

    # ---------- parquet io ----------
    def write(self, df: pd.DataFrame, *parts: str) -> Path:
        p = self.path(*parts)
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)  # atomic: a crash never leaves a half-written file
        return p

    def read(self, *parts: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.root.joinpath(*parts), columns=columns)

    def read_many(self, *parts: str) -> pd.DataFrame:
        """Read every parquet file in a directory (e.g. raw/bars) into one frame."""
        d = self.root.joinpath(*parts)
        files = sorted(d.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return self.sql(f"SELECT * FROM read_parquet('{d.as_posix()}/*.parquet', union_by_name=true)")

    # ---------- duckdb ----------
    def sql(self, query: str) -> pd.DataFrame:
        with duckdb.connect() as con:
            return con.execute(query).df()
