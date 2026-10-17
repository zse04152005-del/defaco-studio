"""Monthly-rebalanced backtest engine with A-share trading restrictions.

Timeline at a rebalance date ``t`` (a trading day):
  1. the day-``t`` return accrues to the *old* (drifted) weights;
  2. at the close the strategy sees only data up to ``t`` and proposes target weights;
  3. trading restrictions are enforced: suspended names cannot trade, names closing at the
     limit-up cannot be bought more, names at the limit-down cannot be sold;
  4. costs are charged on the traded weight and deducted from day ``t``'s return;
  5. weights drift with daily returns until the next rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from ..data.panel import MarketPanel
from .costs import CostModel


@dataclass
class RebalanceContext:
    asof: pd.Timestamp
    panel: MarketPanel
    drift_weights: pd.Series              # current holdings (sum to 1), before trading
    suspended: set[str]
    no_buy: set[str]                      # closed at limit-up
    no_sell: set[str]                     # closed at limit-down

    def bounds_for(self, codes: list[str], max_weight: float) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
        """Per-asset bounds for ``codes`` and fixed weights for held names outside ``codes``."""
        bounds: dict[str, tuple[float, float]] = {}
        held = self.drift_weights
        for c in codes:
            h = float(held.get(c, 0.0))
            lo, hi = 0.0, max_weight
            if c in self.suspended:
                lo = hi = h
            else:
                if c in self.no_buy:
                    hi = min(hi, h)
                if c in self.no_sell:
                    lo = h
                    hi = max(hi, h)
            if (lo, hi) != (0.0, max_weight):
                bounds[c] = (lo, hi)
        outside = {c: float(w) for c, w in held.items()
                   if c not in set(codes) and w > 0 and (c in self.suspended or c in self.no_sell)}
        return bounds, outside


class Strategy(Protocol):
    name: str

    def target_weights(self, ctx: RebalanceContext) -> pd.Series: ...


@dataclass
class BacktestResult:
    name: str
    returns_gross: pd.Series
    returns_net: pd.Series
    weights: dict[pd.Timestamp, pd.Series]
    turnover: pd.Series
    costs: pd.Series
    diagnostics: dict[pd.Timestamp, Any] = field(default_factory=dict)

    @property
    def nav(self) -> pd.Series:
        return (1 + self.returns_net).cumprod()


def enforce_restrictions(target: pd.Series, ctx: RebalanceContext) -> pd.Series:
    """Project a target onto what can actually be traded, keeping the total at 1."""
    names = target.index.union(ctx.drift_weights[ctx.drift_weights > 0].index)
    tgt = target.reindex(names).fillna(0.0)
    held = ctx.drift_weights.reindex(names).fillna(0.0)
    out = tgt.copy()
    fixed = pd.Series(False, index=names)
    for c in names:
        if c in ctx.suspended:
            out[c], fixed[c] = held[c], True
        elif c in ctx.no_buy and tgt[c] > held[c]:
            out[c], fixed[c] = held[c], True
        elif c in ctx.no_sell and tgt[c] < held[c]:
            out[c], fixed[c] = held[c], True
    free_budget = 1.0 - out[fixed].sum()
    free = out[~fixed]
    if free_budget <= 1e-12:
        out[~fixed] = 0.0                       # restricted names already absorb the whole book
    elif free.sum() > 0:
        out[~fixed] = free * (free_budget / free.sum())
    else:                                       # nothing tradable was targeted: keep proportions
        out = out / out.sum()
    return out[out > 1e-10]


def run_backtest(panel: MarketPanel, strategy: Strategy, rebalance_dates: list[pd.Timestamp],
                 end: pd.Timestamp | None = None, costs: CostModel | None = None,
                 on_rebalance: Callable[[pd.Timestamp, pd.Series], None] | None = None) -> BacktestResult:
    costs = costs or CostModel()
    rebal = [panel.dates[panel.loc_date(d)] for d in rebalance_dates]
    rebal = sorted(set(rebal))
    last = panel.dates[panel.loc_date(end)] if end is not None else panel.dates[-1]
    days = panel.dates[(panel.dates >= rebal[0]) & (panel.dates <= last)]
    rset = set(rebal)

    w = pd.Series(dtype=float)       # current (drifted) weights
    gross, net, turn, cost_s = {}, {}, {}, {}
    weights: dict[pd.Timestamp, pd.Series] = {}
    diags: dict[pd.Timestamp, Any] = {}
    ret = panel.ret

    for d in days:
        # 1) day-d return on the holdings carried into the day
        if len(w):
            r = ret.loc[d, w.index].fillna(0.0)   # suspended / delisted -> 0
            pr = float((w * r).sum())
            w = w * (1 + r) / (1 + pr)
        else:
            pr = 0.0
        c = 0.0
        # 2-4) rebalance at the close
        if d in rset:
            ts = panel.tradestatus.loc[d]
            up, dn = panel.limit_flags(d)
            ctx = RebalanceContext(asof=d, panel=panel, drift_weights=w.copy(),
                                   suspended=set(ts.index[ts <= 0]),
                                   no_buy=set(up.index[up]), no_sell=set(dn.index[dn]))
            target = strategy.target_weights(ctx)
            new_w = enforce_restrictions(target, ctx)
            names = new_w.index.union(w.index)
            delta = new_w.reindex(names).fillna(0.0) - w.reindex(names).fillna(0.0)
            buys, sells = float(delta.clip(lower=0).sum()), float((-delta).clip(lower=0).sum())
            c = costs.cost(d, buys, sells)
            turn[d], cost_s[d] = buys + sells, c
            w = new_w
            weights[d] = new_w
            if hasattr(strategy, "last_diagnostics"):
                diags[d] = strategy.last_diagnostics
            if on_rebalance:
                on_rebalance(d, new_w)
        gross[d] = pr
        net[d] = (1 + pr) * (1 - c) - 1

    return BacktestResult(name=getattr(strategy, "name", "strategy"),
                          returns_gross=pd.Series(gross), returns_net=pd.Series(net),
                          weights=weights, turnover=pd.Series(turn), costs=pd.Series(cost_s),
                          diagnostics=diags)
