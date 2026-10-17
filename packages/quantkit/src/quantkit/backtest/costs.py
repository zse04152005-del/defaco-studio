"""A-share transaction costs."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STAMP_DUTY_CUT = pd.Timestamp("2023-08-28")   # stamp duty halved from 0.1% to 0.05%


@dataclass
class CostModel:
    commission_bps: float = 2.0     # both sides
    slippage_bps: float = 0.0       # both sides
    stamp_before_bps: float = 10.0  # sell side, before 2023-08-28
    stamp_after_bps: float = 5.0    # sell side, from 2023-08-28

    @classmethod
    def flat(cls, total_bps: float) -> "CostModel":
        """Symmetric cost of ``total_bps`` per unit traded (as in the benchmark paper)."""
        return cls(commission_bps=total_bps, slippage_bps=0.0, stamp_before_bps=0.0, stamp_after_bps=0.0)

    def cost(self, date: pd.Timestamp, buys: float, sells: float) -> float:
        """Cost as a fraction of portfolio value given traded weight on each side."""
        stamp = self.stamp_before_bps if pd.Timestamp(date) < STAMP_DUTY_CUT else self.stamp_after_bps
        both = (self.commission_bps + self.slippage_bps) * (buys + sells)
        return (both + stamp * sells) / 1e4
