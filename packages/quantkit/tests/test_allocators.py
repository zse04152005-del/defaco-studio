import numpy as np
import pandas as pd

from quantkit.optim.allocators import bayes_stein_mean, cap_weights, hrp, mean_variance, risk_contributions, risk_parity
from quantkit.optim.minvar import Constraints, min_variance
from quantkit.riskmodels.base import CovarianceRepr


def random_cov(n=30, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, (n, 3))
    S = (B @ B.T + np.diag(rng.uniform(0.5, 3, n))) * 1e-4
    return CovarianceRepr([f"{600000 + i}.SH" for i in range(n)], dense_cov=S), S


def test_erc_equal_risk_contributions():
    cov, S = random_cov()
    w = risk_parity(cov)
    rc = risk_contributions(S, w.to_numpy())
    assert abs(w.sum() - 1) < 1e-8 and (w > 0).all()
    assert rc.max() - rc.min() < 1e-4


def test_hrp_is_inverse_variance_on_diagonal_cov():
    var = np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4
    cov = CovarianceRepr(list("ABCD"), dense_cov=np.diag(var))
    w = hrp(cov)
    iv = (1 / var) / (1 / var).sum()
    assert np.allclose(w.to_numpy(), iv, atol=1e-10)


def test_caps():
    w = cap_weights(pd.Series([0.6, 0.2, 0.1, 0.1]), 0.3)
    assert abs(w.sum() - 1) < 1e-12 and w.max() <= 0.3 + 1e-12
    cov, _ = random_cov(20)
    assert hrp(cov, cap=0.08).max() <= 0.08 + 1e-9 and risk_parity(cov, cap=0.08).max() <= 0.08 + 1e-9


def test_mean_variance_limits():
    cov, S = random_cov(25)
    mu = np.linspace(-1e-4, 1e-3, 25)
    mv_heavy = mean_variance(cov, mu, risk_aversion=1e6, constraints=Constraints(max_weight=0.2))
    gmv = min_variance(cov, Constraints(max_weight=0.2))
    assert np.abs(mv_heavy.to_numpy() - gmv.reindex(mv_heavy.index).fillna(0).to_numpy()).max() < 5e-3
    mv_light = mean_variance(cov, mu, risk_aversion=0.5, constraints=Constraints(max_weight=0.2))
    assert mv_light.to_numpy() @ mu > mv_heavy.to_numpy() @ mu and mv_light.max() <= 0.2 + 1e-6


def test_bayes_stein():
    cov, S = random_cov(10)
    x = np.random.default_rng(1).multivariate_normal(np.full(10, 5e-4), S, size=250)
    mu, d = bayes_stein_mean(x, S)
    assert 0 <= d <= 1 and mu.std() <= x.mean(axis=0).std() + 1e-12
