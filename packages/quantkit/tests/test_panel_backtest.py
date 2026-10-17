import numpy as np
import pandas as pd

from quantkit.backtest.costs import CostModel
from quantkit.backtest.engine import RebalanceContext, enforce_restrictions, run_backtest
from quantkit.backtest.strategies import EqualWeight, MinVariance
from quantkit.data.codes import limit_pct, to_baostock, to_canonical
from quantkit.metrics.performance import cov_losses, monthly_sd_sharpe, performance, weight_stats
from quantkit.riskmodels import build
from quantkit.universe import UniverseRule, eligible


def test_code_conventions():
    assert to_canonical("sh.600519") == "600519.SH"
    assert to_canonical("sz000001") == "000001.SZ"
    assert to_canonical("300750") == "300750.SZ"
    assert to_canonical("920982") == "920982.BJ" and to_canonical("430047") == "430047.BJ"
    assert to_canonical("900901") == "900901.SH"
    assert to_baostock("600519.SH") == "sh.600519"
    import datetime as dt
    assert limit_pct("300750.SZ", dt.date(2019, 1, 2), False) == 0.10
    assert limit_pct("300750.SZ", dt.date(2021, 1, 4), False) == 0.20
    assert limit_pct("600000.SH", dt.date(2021, 1, 4), True) == 0.05
    assert limit_pct("688981.SH", dt.date(2021, 1, 4), True) == 0.20


def test_limit_flags_detect_forced_limit_up(synth):
    p, codes, d = synth["panel"], synth["codes"], synth["dates"][650]
    up, dn = p.limit_flags(d)
    assert up[codes[3]] and not dn[codes[3]]
    assert up.sum() <= 2   # only the forced case (clipped synthetic returns stay below the band)


def test_universe_filters(synth):
    p, codes = synth["panel"], synth["codes"]
    rule = UniverseRule(indices=("hs300",), min_listing_days=252)
    at = synth["dates"][310]
    u = eligible(p, at, rule)
    assert codes[5] not in u            # suspended that day
    assert codes[7] not in u            # not yet listed
    late = eligible(p, synth["dates"][-1], rule)
    assert codes[9] not in late         # ST
    assert codes[7] not in late         # listed < 252 days
    top = eligible(p, at, UniverseRule(indices=("hs300",), top_n_by_mcap=10))
    assert len(top) == 10


def test_enforce_restrictions_keeps_frozen_names(synth):
    p = synth["panel"]
    held = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    ctx = RebalanceContext(asof=synth["dates"][400], panel=p, drift_weights=held,
                           suspended={"A"}, no_buy={"B"}, no_sell={"C"})
    tgt = pd.Series({"B": 0.6, "C": 0.0, "D": 0.4})
    out = enforce_restrictions(tgt, ctx)
    assert abs(out["A"] - 0.5) < 1e-12        # suspended: unchanged
    assert abs(out["B"] - 0.3) < 1e-12        # limit-up: cannot increase
    assert abs(out["C"] - 0.2) < 1e-12        # limit-down: cannot decrease
    assert abs(out.sum() - 1.0) < 1e-12


def test_equal_weight_backtest_accounting(synth):
    p, dates = synth["panel"], synth["dates"]
    rebal = [d for d in pd.DatetimeIndex(dates).to_series().groupby(dates.to_period("M")).max()][13:]
    rule = UniverseRule(indices=("hs300",), min_listing_days=252)
    res0 = run_backtest(p, EqualWeight(rule), rebal, costs=CostModel.flat(0.0))
    res = run_backtest(p, EqualWeight(rule), rebal, costs=CostModel.flat(20.0))
    assert np.allclose(res0.returns_gross, res0.returns_net)
    # day after the first rebalance: EW return equals the cross-sectional mean
    d0 = rebal[0]
    w0 = res0.weights[d0]
    nxt = p.dates[p.loc_date(d0) + 1]
    assert abs(res0.returns_gross[nxt] - p.ret.loc[nxt, w0.index].fillna(0).mean()) < 1e-12
    # costs: net < gross by exactly the charged cost on rebalance days
    for d in res.costs.index:
        assert abs((1 + res.returns_gross[d]) * (1 - res.costs[d]) - 1 - res.returns_net[d]) < 1e-12
    assert res.nav.iloc[-1] < res0.nav.iloc[-1]
    assert all(abs(w.sum() - 1) < 1e-9 for w in res.weights.values())


class PeekingGuard:
    """Wraps a strategy and records the latest date of data it was given."""
    name = "guard"

    def __init__(self, inner):
        self.inner, self.violations = inner, []

    def target_weights(self, ctx):
        orig = self.inner.build_input

        def spy(c, codes):
            inp = orig(c, codes)
            if inp.returns.index.max() > ctx.asof:
                self.violations.append(ctx.asof)
            return inp
        self.inner.build_input = spy
        try:
            return self.inner.target_weights(ctx)
        finally:
            self.inner.build_input = orig


def test_min_variance_backtest_no_lookahead_and_beats_ew_risk(synth):
    p, dates = synth["panel"], synth["dates"]
    rebal = list(pd.DatetimeIndex(dates).to_series().groupby(dates.to_period("M")).max())[13:]
    rule = UniverseRule(indices=("hs300",), min_listing_days=252)
    mv = MinVariance(lambda: build("lw_linear"), rule, max_weight=0.15)
    guard = PeekingGuard(mv)
    res = run_backtest(p, guard, rebal, costs=CostModel.flat(0.0))
    assert guard.violations == []
    ew = run_backtest(p, EqualWeight(rule), rebal, costs=CostModel.flat(0.0))
    assert performance(res.returns_net)["ann_vol"] < performance(ew.returns_net)["ann_vol"]
    stats = weight_stats(res.weights, res.turnover)
    assert stats["max_w"] <= 0.15 + 1e-6
    m = monthly_sd_sharpe(res.returns_net)
    assert m["sd_monthly"] > 0


def test_cov_losses_zero_at_truth():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(10, 10))
    S = A @ A.T + np.eye(10)
    loss = cov_losses(S, S)
    assert loss["mse"] < 1e-20 and loss["mae"] < 1e-12 and abs(loss["asym"]) < 1e-8
    # QLIKE is minimised at the truth
    assert loss["qlike"] < cov_losses(1.3 * S, S)["qlike"]


def test_delisted_stock_has_no_price_after_last_bar(synth):
    from quantkit.data.panel import MarketPanel
    bars = synth["bars"]
    code = synth["codes"][11]
    cut = synth["dates"][400]
    b2 = bars[~((bars["code"] == code) & (bars["date"] > cut))]
    p = MarketPanel.from_long(b2, calendar=synth["dates"])
    assert p.close.loc[cut, code] > 0
    assert np.isnan(p.close.loc[synth["dates"][450], code])
    assert np.isnan(p.float_mcap.loc[synth["dates"][-1], code])
    # suspension inside the history is still forward-filled
    assert not np.isnan(synth["panel"].close.loc[synth["dates"][310], synth["codes"][5]])
