"""Realistic simulator: consistency with the research engine, T+1 fills, limits, liquidity, delisting, costs."""
import copy

import numpy as np
import pandas as pd

from quantkit.backtest.costs import CostModel
from quantkit.backtest.engine import run_backtest
from quantkit.backtest.simulator import CASH, Execution, simulate
from quantkit.backtest.strategies import EqualWeight
from quantkit.universe import UniverseRule

RULE = UniverseRule(indices=("hs300",), min_listing_days=60, window_days=60)


class Fixed:
    """Always target the given weights."""
    name = "fixed"

    def __init__(self, w):
        self.w = pd.Series(w)

    def target_weights(self, ctx):
        return self.w


def rebal(panel, n=4, start=300, step=21):
    return [panel.dates[start + i * step] for i in range(n)]


def test_close_frictionless_matches_research_engine(synth):
    p = synth["panel"]
    dates = rebal(p)
    old = run_backtest(p, EqualWeight(RULE), dates, costs=CostModel.flat(0.0))
    new = simulate(p, EqualWeight(RULE), dates, Execution.frictionless("close"))
    common = old.returns_gross.index.intersection(new.returns.index)
    assert np.allclose(old.returns_gross[common].to_numpy(), new.returns[common].to_numpy(), atol=1e-12)


def test_next_open_untraded_position_earns_close_to_close(synth):
    p = synth["panel"]
    codes = synth["codes"][:4]
    dates = [p.dates[300]]                      # one signal, filled at the open of day 301
    res = simulate(p, Fixed({c: 0.25 for c in codes}), dates, Execution.frictionless("next_open"))
    assert res.returns.loc[p.dates[300]] == 0.0               # still all cash on the signal day
    fill = p.dates[301]
    assert fill in res.weights and abs(res.weights[fill].drop(CASH).sum() - 1) < 1e-12
    # after the fill an untraded position compounds the daily close-to-close returns
    w = res.weights[fill].drop(CASH)                       # weights at the open fill ...
    w = w * (p.close.loc[fill, codes] / p.open.loc[fill, codes])   # ... drifted to the close of the fill day
    w = w / w.sum()
    R = p.ret.loc[p.dates[302]:p.dates[320], codes].fillna(0.0)
    bh = float((w * (1 + R).prod()).sum())
    nav = float((1 + res.returns.loc[p.dates[302]:p.dates[320]]).prod())
    assert abs(nav - bh) < 1e-9


def test_limit_up_at_open_blocks_buy(synth):
    p = copy.deepcopy(synth["panel"])
    c = synth["codes"][2]
    fill = p.dates[401]
    up, _ = p.limit_prices(fill)
    p.open.loc[fill, c] = up[c]                  # opens at limit-up
    res = simulate(p, Fixed({c: 0.5, synth["codes"][4]: 0.5}), [p.dates[400]], Execution.frictionless("next_open"))
    assert res.blocked[fill][c] == "涨停"
    assert c not in res.weights[fill].index and res.weights[fill][CASH] > 0.49


def test_participation_cap_leaves_cash(synth):
    p = synth["panel"]
    codes = synth["codes"][:4]
    ex = Execution(timing="next_open", aum=1e13, participation_cap=0.05, impact_y=0)
    res = simulate(p, Fixed({c: 0.25 for c in codes}), [p.dates[300]], ex)
    fill = p.dates[301]
    assert res.unfilled[fill] > 0.5 and res.weights[fill][CASH] > 0.5
    assert all(v == "流动性" for v in res.blocked[fill].values())


def test_delisted_name_is_liquidated(synth):
    p = copy.deepcopy(synth["panel"])
    c = synth["codes"][6]
    gone = p.dates[420]
    for f in (p.close, p.open, p.ret, p.preclose):
        f.loc[gone:, c] = np.nan
    p.tradestatus.loc[gone:, c] = 0
    res = simulate(p, Fixed({c: 0.5, synth["codes"][4]: 0.5}), [p.dates[400], p.dates[430]], Execution.frictionless("next_open"))
    assert c not in res.weights[p.dates[431]].index


def test_costs_and_min_commission(synth):
    p = synth["panel"]
    codes = synth["codes"][:10]
    big = simulate(p, Fixed({c: 0.1 for c in codes}), [p.dates[300]], Execution(aum=1e9, impact_y=0, participation_cap=None))
    small = simulate(p, Fixed({c: 0.1 for c in codes}), [p.dates[300]], Execution(aum=1e4, impact_y=0, participation_cap=None))
    fill = p.dates[301]
    traded = big.turnover[fill]                                               # initial build: bought weight
    assert abs(big.costs.loc[fill, "commission"] - traded * 2.5e-4) < 1e-12   # 2.5 bp
    n_orders = round(traded / 0.1)
    assert abs(small.costs.loc[fill, "commission"] - n_orders * 5 / 1e4) < 1e-9   # 5 yuan minimum per order
    with_impact = simulate(p, Fixed({c: 0.1 for c in codes}), [p.dates[300]], Execution(aum=1e9, impact_y=1.0, participation_cap=None))
    assert with_impact.costs.loc[fill, "impact"] > 0
