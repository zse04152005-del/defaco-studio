"""Whole-lot portfolio search (quantkit.optim.lots)."""
import math

import numpy as np
import pandas as pd
import pytest

from quantkit.optim import lots

CODES = ["600000.SH", "600001.SH", "600002.SH", "000001.SZ", "000002.SZ", "688001.SH"]
PRICES = pd.Series([8.0, 12.0, 25.0, 6.5, 40.0, 30.0], index=CODES)
IND = pd.Series(["银行", "银行", "电力", "电力", "食品", "电子"], index=CODES)


def cov(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(len(CODES), 2)) * 0.15
    return B @ B.T + np.diag(rng.uniform(0.02, 0.06, len(CODES)))


def test_fees_and_lots():
    assert lots.fee(10_000, "buy") == pytest.approx(5 + 0.1)
    assert lots.fee(1_000_000, "sell") == pytest.approx(250 + 10 + 500)
    assert lots.lot_of("688981.SH") == (200, 1) and lots.lot_of("600519.SH") == (100, 100)


def test_lot_portfolio_respects_budget_range_caps_and_lots():
    S = cov()
    w = pd.Series(1 / len(CODES), index=CODES)
    for budget, wmax in [(10_000, 0.30), (100_000, 0.25)]:              # both have feasible whole-lot holdings
        r = lots.lot_portfolio(CODES, S, PRICES, budget, w, wmax, IND, ind_cap=0.5)
        sh = r.shares
        assert all(n == 0 or (n % 100 == 0 if not c.startswith("688") else n >= 200) for c, n in sh.items())
        assert 0.90 <= r.spend_ratio <= 1.10 + 1e-9
        v = sh * PRICES
        assert (v / v.sum()).max() <= wmax + 1e-9


def test_lot_portfolio_is_no_riskier_than_plain_rounding():
    S = cov(1)
    w = pd.Series([0.3, 0.1, 0.2, 0.25, 0.05, 0.10], index=CODES)
    budget = 20_000
    r = lots.lot_portfolio(CODES, S, PRICES, budget, w, 0.40, IND, ind_cap=0.6)
    naive = (np.floor(w * budget / PRICES / 100) * 100).clip(lower=0)
    naive["688001.SH"] = 0 if naive["688001.SH"] < 200 else naive["688001.SH"]
    vn = (naive * PRICES).to_numpy(); wn = vn / vn.sum()
    assert r.risk <= math.sqrt(wn @ S @ wn) + 1e-9


def test_frozen_stocks_are_not_traded():
    S = cov(2)
    w = pd.Series(1 / len(CODES), index=CODES)
    r = lots.lot_portfolio(CODES, S, PRICES, 50_000, w, 0.3, IND, ind_cap=0.6, no_buy={"600000.SH"})
    assert r.shares["600000.SH"] == 0
    cur = pd.Series([1000, 0, 0, 0, 0, 0], index=CODES)            # 8,000 in a stock that may not be sold
    r = lots.lot_portfolio(CODES, S, PRICES, 50_000, w, 0.3, IND, ind_cap=0.6, lo=0.0, hi=1.0, current=cur,
                           no_sell={"600000.SH"})
    assert r.shares["600000.SH"] >= 1000


def test_rebalance_does_not_trade_when_already_at_the_best_holdings():
    S = cov(3)
    w = pd.Series(1 / len(CODES), index=CODES)
    first = lots.lot_portfolio(CODES, S, PRICES, 100_000, w, 0.3, IND, ind_cap=0.6)
    value = float((first.shares * PRICES).sum()) + 500
    again = lots.lot_portfolio(CODES, S, PRICES, value, w, 0.3, IND, ind_cap=0.6, hi=1.0, current=first.shares)
    moved = (again.shares - first.shares).abs().sum()
    assert moved <= 200                                             # at most a lot or two, never a reshuffle


def test_equal_weight_tracking_gets_close_to_equal():
    w = pd.Series(1 / len(CODES), index=CODES)
    r = lots.lot_portfolio(CODES, np.eye(len(CODES)), PRICES, 200_000, w, 0.3, IND, ind_cap=0.6, track=True)
    v = r.shares * PRICES
    assert (v / v.sum() - 1 / len(CODES)).abs().max() < 0.02
