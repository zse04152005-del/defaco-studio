import numpy as np
import pandas as pd

from quantkit.metrics.report import deflate, drawdown_stats, expected_max_sr, full_metrics, psr, sharpe_stats


def series(mu=4e-4, sd=0.01, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.bdate_range("2016-01-01", periods=n))


def test_sharpe_se_normal_case():
    s = series(n=5000)
    st = sharpe_stats(s)
    # normal returns: SE ≈ sqrt((1 + SR²/2) / T)
    assert abs(st["se_daily"] - np.sqrt((1 + st["sr_daily"] ** 2 / 2) / 4999)) < 2e-3


def test_psr_and_deflation():
    assert psr(0.05, 0.02, 0.0) > 0.99 and abs(psr(0.0, 0.02, 0.0) - 0.5) < 1e-12
    assert expected_max_sr([0.01, 0.02, 0.03, 0.0, -0.01]) > 0
    m = deflate({f"s{i}": full_metrics(series(seed=i)) for i in range(8)})
    assert all(v["dsr"] <= v["psr_0"] + 1e-12 for v in m.values()) and m["s0"]["dsr_trials"] == 8


def test_benchmark_relative():
    b = series(mu=2e-4, seed=3)
    r = 0.5 * b + series(mu=3e-4, sd=0.004, seed=4)
    m = full_metrics(r, benchmark=b)
    assert abs(m["beta"] - 0.5) < 0.05 and m["tracking_error"] > 0 and "information_ratio" in m


def test_drawdown_dates():
    nav = pd.Series([1, 1.2, 0.9, 1.0, 1.25, 1.1], index=pd.bdate_range("2020-01-01", periods=6))
    d = drawdown_stats(nav)
    assert abs(d["max_drawdown"] + 0.25) < 1e-12 and d["dd_peak"] == "2020-01-02" and d["dd_recovered"] == "2020-01-07"
