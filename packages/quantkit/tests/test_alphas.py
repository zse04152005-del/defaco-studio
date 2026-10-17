"""Vectorised operators against straightforward pandas / numpy references."""
import numpy as np
import pandas as pd

from quantkit.features import alphas as A


def data(T=80, N=5, seed=0):
    rng = np.random.default_rng(seed)
    x = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    y = rng.lognormal(10, 0.5, (T, N))
    x[5, 1] = np.nan
    return x, y


def test_rolling_basics():
    x, y = data()
    df = pd.DataFrame(x)
    assert np.allclose(A.ts_mean(x, 10)[20:], df.rolling(10, min_periods=8).mean().to_numpy()[20:], equal_nan=True)
    assert np.allclose(A.ts_std(x, 10)[20:], df.rolling(10, min_periods=8).std().to_numpy()[20:], equal_nan=True)
    assert np.allclose(A.delay(x, 3)[3:], x[:-3], equal_nan=True) and np.isnan(A.delay(x, 3)[:3]).all()


def test_corr_matches_pandas():
    x, y = data()
    ref = pd.DataFrame(x).rolling(10).corr(pd.DataFrame(y)).to_numpy()
    got = A.ts_corr(x, y, 10)
    m = ~np.isnan(ref) & ~np.isnan(got)
    assert m.sum() > 200 and np.allclose(got[m], ref[m], atol=1e-8)


def test_rank_argmax_slope():
    x = np.array([[1.0], [3.0], [2.0], [5.0], [4.0]])
    assert A.ts_rank(x, 3)[-1, 0] == 0.5                 # 4 is the middle of (2, 5, 4)
    assert A.ts_argmax(x, 3)[-1, 0] == 2                 # max 5 sits at position 2 of 3 (oldest = 1)
    t = np.arange(30, dtype=float)[:, None]
    line = 2 + 0.5 * t
    assert np.allclose(A.ts_slope(line, 10)[10:], 0.5) and np.allclose(A.ts_rsquare(line, 10)[10:], 1.0)
    assert np.allclose(A.ts_resid(line, 10)[10:], 0.0, atol=1e-9)
    noisy = line + np.random.default_rng(1).normal(0, 1, line.shape)
    b = np.polyfit(np.arange(10), noisy[-10:, 0], 1)[0]
    assert abs(A.ts_slope(noisy, 10)[-1, 0] - b) < 1e-9


def test_cross_sectional_rank_and_sma():
    x = np.array([[3.0, 1.0, 2.0, np.nan]])
    assert np.allclose(A.rank(x)[0, :3], [1.0, 1 / 3, 2 / 3]) and np.isnan(A.rank(x)[0, 3])
    s = A.sma(np.ones((20, 1)) * 5, 10, 2)
    assert np.allclose(s, 5)


def test_catalog_runs_on_synthetic_inputs():
    T, N = 300, 8
    rng = np.random.default_rng(2)
    c = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    o = c * (1 + rng.normal(0, 0.005, (T, N)))
    h, l = np.fmax(o, c) * 1.01, np.fmin(o, c) * 0.99
    v = rng.lognormal(12, 0.4, (T, N))
    inp = A.Inputs(open=o, high=h, low=l, close=c, vwap=(h + l + c) / 3, volume=v, amount=v * c, ret=np.vstack([np.full((1, N), np.nan), c[1:] / c[:-1] - 1]),
                   raw_open=o, raw_high=h, raw_low=l, raw_close=c)
    out = A.compute(inp)
    assert len(out) >= 100
    bad = [k for k, val in out.items() if np.isnan(val[-1]).all() or np.nanstd(val[-1]) == 0 and k not in ("WQ007",)]
    assert not bad, bad
