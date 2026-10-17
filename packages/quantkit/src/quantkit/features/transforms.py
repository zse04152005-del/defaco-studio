"""Cross-sectional transforms."""
from __future__ import annotations

import pandas as pd


def rank_normalize(x: pd.DataFrame, by: str = "date", cols: list[str] | None = None) -> pd.DataFrame:
    """Map each characteristic to [-0.5, 0.5] by cross-sectional rank; missing -> 0 (the median).

    rank/(n+1) - 0.5 as in Gu, Kelly & Xiu (2020); n counts non-missing values that date.
    """
    cols = cols or [c for c in x.columns if c not in (by, "code")]
    out = x.copy()
    g = x.groupby(by)[cols]
    ranks = g.rank(method="average")
    n = g.transform("count")
    out[cols] = (ranks / (n + 1) - 0.5).fillna(0.0)
    return out


def characteristic_matrix(long: pd.DataFrame, asof: pd.Timestamp, codes: list[str],
                          cols: list[str], add_const: bool = True) -> pd.DataFrame:
    """N x L matrix Z_t for ``codes`` at month-end ``asof`` (rows for unknown codes are 0)."""
    z = long[long["date"] == pd.Timestamp(asof)].set_index("code")[cols].reindex(codes).fillna(0.0)
    if add_const:
        z.insert(0, "const", 1.0)
    return z
