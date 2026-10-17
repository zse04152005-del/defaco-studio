"""Reference strategies: equal weight, value weight and minimum variance with any RiskModel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ..features.transforms import characteristic_matrix
from ..optim.minvar import Constraints, min_variance
from ..riskmodels.base import RiskModel, RiskModelInput
from ..universe import UniverseRule, eligible
from .engine import RebalanceContext


@dataclass
class EqualWeight:
    universe: UniverseRule
    name: str = "EW"

    def target_weights(self, ctx: RebalanceContext) -> pd.Series:
        codes = eligible(ctx.panel, ctx.asof, self.universe)
        return pd.Series(1.0 / len(codes), index=codes)


@dataclass
class ValueWeight:
    universe: UniverseRule
    name: str = "VW"

    def target_weights(self, ctx: RebalanceContext) -> pd.Series:
        codes = eligible(ctx.panel, ctx.asof, self.universe)
        m = ctx.panel.float_mcap.loc[ctx.asof, codes].fillna(0.0)
        return m / m.sum()


@dataclass
class MinVariance:
    """Long-only minimum variance with a pluggable risk model (the paper's main portfolio)."""

    model_factory: Callable[[], RiskModel]
    universe: UniverseRule
    name: str = "MinVar"
    window_days: int = 250
    max_weight: float = 0.10
    turnover_penalty: float = 0.0
    characteristics: pd.DataFrame | None = None      # long, rank-normalised
    char_cols: list[str] | None = None
    keep_covariance: bool = False
    last_diagnostics: dict[str, Any] = field(default_factory=dict)

    def build_input(self, ctx: RebalanceContext, codes: list[str]) -> RiskModelInput:
        p = ctx.panel
        win = p.window(ctx.asof, self.window_days)
        chars = None
        if self.characteristics is not None:
            month_end = self.characteristics.loc[self.characteristics["date"] <= ctx.asof, "date"].max()
            chars = characteristic_matrix(self.characteristics, month_end, codes, self.char_cols or [])
        return RiskModelInput(asof=ctx.asof, returns=p.ret.loc[win, codes], characteristics=chars,
                              industries=p.industry_at(ctx.asof, codes),
                              market=p.market_ret.loc[win] if p.market_ret is not None else None)

    def target_weights(self, ctx: RebalanceContext) -> pd.Series:
        codes = eligible(ctx.panel, ctx.asof, self.universe)
        model = self.model_factory().fit(self.build_input(ctx, codes))
        cov = model.covariance()
        bounds, outside = ctx.bounds_for(codes, self.max_weight)
        cons = Constraints(max_weight=self.max_weight, bounds=bounds,
                           budget=1.0 - sum(outside.values()),
                           turnover_penalty=self.turnover_penalty, prev_weights=ctx.drift_weights)
        w = min_variance(cov, cons)
        self.last_diagnostics = {"n_assets": len(codes), "model": model.explain()}
        if self.keep_covariance:
            self.last_diagnostics["cov"] = cov
        return pd.concat([w, pd.Series(outside, dtype=float)])
