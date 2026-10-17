import numpy as np
import pandas as pd
import pytest

from quantkit.optim.minvar import Constraints, gmv_closed_form, min_variance, precision
from quantkit.riskmodels import CovarianceRepr, IndustryBlockResidual, RiskModelInput, available, build
from quantkit.riskmodels.baselines import analytical_nonlinear_shrinkage


@pytest.fixture(scope="module")
def window(synth):
    R = synth["R"].iloc[:300, :30].dropna(axis=1)
    return RiskModelInput(asof=R.index[-1], returns=R)


@pytest.mark.parametrize("name", ["sample", "lw_linear", "oas", "lw_nonlinear", "ewma", "pca_factor"])
def test_models_return_valid_covariance(window, name):
    cov = build(name).fit(window).covariance()
    S = cov.dense()
    assert S.shape == (len(window.codes),) * 2
    assert np.allclose(S, S.T, atol=1e-12)
    assert np.linalg.eigvalsh(S).min() > 0


def test_registry_lists_all():
    assert {"sample", "lw_linear", "oas", "lw_nonlinear", "ewma", "pca_factor"} <= set(available())


def test_nls_shrinks_eigenvalue_dispersion():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 80))          # true covariance = I
    x -= x.mean(axis=0)
    sample_ev = np.linalg.eigvalsh(x.T @ x / 119)
    nls_ev = np.linalg.eigvalsh(analytical_nonlinear_shrinkage(x))
    assert nls_ev.std() < 0.5 * sample_ev.std()
    assert abs(nls_ev.mean() - 1.0) < 0.15


def test_industry_block_inverse_matches_dense():
    rng = np.random.default_rng(2)
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    blk = IndustryBlockResidual(sigma2=rng.uniform(1e-4, 4e-4, 10), groups=groups, rho={0: 0.3, 1: -0.2, 2: 0.15})
    assert np.allclose(blk.inv(), np.linalg.inv(blk.dense()))


@pytest.mark.parametrize("resid", ["diag", "block"])
def test_woodbury_precision_and_gmv(resid):
    rng = np.random.default_rng(3)
    N, K = 25, 3
    B = rng.normal(size=(N, K))
    Sf = np.diag([4e-4, 1e-4, 5e-5])
    s2 = rng.uniform(1e-4, 3e-4, N)
    su = s2 if resid == "diag" else IndustryBlockResidual(s2, np.arange(N) % 5, {g: 0.2 for g in range(5)})
    cov = CovarianceRepr([f"c{i}" for i in range(N)], B=B, sigma_f=Sf, sigma_u=su)
    assert np.allclose(precision(cov), np.linalg.inv(cov.dense()))
    w = gmv_closed_form(cov)
    dense_w = np.linalg.solve(cov.dense(), np.ones(N))
    assert np.allclose(w.values, dense_w / dense_w.sum())


def test_long_only_qp_constraints(window):
    cov = build("lw_linear").fit(window).covariance()
    codes = cov.codes
    cons = Constraints(max_weight=0.08, bounds={codes[0]: (0.05, 0.05)}, budget=0.9)
    w = min_variance(cov, cons)
    assert abs(w.sum() - 0.9) < 1e-6
    assert w.min() >= -1e-8 and w.max() <= 0.08 + 1e-6
    assert abs(w[codes[0]] - 0.05) < 1e-5


def test_group_cap_binds(window):
    cov = build("lw_linear").fit(window).covariance()
    groups = pd.Series([i % 3 for i in range(len(cov.codes))], index=cov.codes)
    free = min_variance(cov, Constraints(max_weight=0.2))
    top = free.groupby(groups).sum().max()
    w = min_variance(cov, Constraints(max_weight=0.2, groups=groups, group_cap=0.36))
    assert top > 0.36 + 1e-3                       # the cap actually binds on this data
    assert w.groupby(groups).sum().max() <= 0.36 + 1e-6 and abs(w.sum() - 1) < 1e-8


def test_factor_form_qp_matches_dense(window):
    fac = build("pca_factor", n_factors=4).fit(window).covariance()
    dense = CovarianceRepr(fac.codes, dense_cov=fac.dense())
    w1, w2 = min_variance(fac, Constraints(max_weight=0.1)), min_variance(dense, Constraints(max_weight=0.1))
    S = fac.dense()
    assert abs(w1 @ S @ w1 - w2 @ S @ w2) / (w2 @ S @ w2) < 1e-4


def test_turnover_penalty_reduces_trading(window):
    cov = build("sample").fit(window).covariance()
    prev = pd.Series(1.0 / len(cov.codes), index=cov.codes)
    free = min_variance(cov, Constraints(max_weight=0.2))
    pen = min_variance(cov, Constraints(max_weight=0.2, turnover_penalty=2e-4, prev_weights=prev))
    assert (pen - prev).abs().sum() < (free - prev).abs().sum()
