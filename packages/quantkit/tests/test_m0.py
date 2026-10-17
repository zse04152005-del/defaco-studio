import numpy as np
import pytest

from quantkit.dynamics import dcc11, garch11
from quantkit.latent import FactorExtractor, rank_transform
from quantkit.metrics.tests import dm_test, sharpe_test, variance_test
from quantkit.riskmodels import RiskModelInput
from quantkit.riskmodels.latent_factor import FactorFit, factor_covariance


def test_rank_transform_range():
    x = np.array([[1.0, 3.0, np.nan, 2.0]])
    z = rank_transform(x)
    assert z[0, 2] == 0.0 and z[0].max() < 0.5 and z[0].min() > -0.5
    assert z[0, 1] > z[0, 3] > z[0, 0]


@pytest.mark.parametrize("method", ["pca", "pls", "spca", "spls", "aen", "dae"])
def test_extractors_shapes(synth, method):
    if method in ("aen", "dae"):
        pytest.importorskip("torch")       # optional extra: pip install -e "packages/quantkit[dl]"
    z = rank_transform(synth["R"].iloc[:250, :30].to_numpy())
    F = FactorExtractor(method, 3, {"alpha": 0.5} if method == "spca" else {}).fit_transform(z)
    assert F.shape[0] == 250 and 1 <= F.shape[1] <= 3
    assert np.isfinite(F).all()


def test_garch_and_dcc_on_simulated_data():
    rng = np.random.default_rng(7)
    T = 1500
    # GARCH(1,1) series
    e = rng.standard_normal(T)
    h = np.empty(T); x = np.empty(T); h[0] = 1e-4
    for t in range(T):
        if t: h[t] = 1e-6 + 0.08 * x[t - 1] ** 2 + 0.9 * h[t - 1]
        x[t] = np.sqrt(h[t]) * e[t]
    g = garch11(x)
    assert g.ok and g.var_forecast > 0
    z = np.column_stack([rng.standard_normal(T), rng.standard_normal(T)])
    z[:, 1] = 0.6 * z[:, 0] + 0.8 * z[:, 1]
    d = dcc11(z)
    R = d.corr_forecast(21)
    assert abs(R[0, 1] - 0.6) < 0.1 and d.a + d.b < 1


@pytest.mark.parametrize("spec", ["sfc", "dfc", "dbc", "dec"])
@pytest.mark.parametrize("resid", ["efm", "afm"])
def test_factor_covariance_specs_psd(synth, spec, resid):
    R = synth["R"].iloc[:250, :20]
    fit = FactorFit(RiskModelInput(asof=R.index[-1], returns=R), FactorExtractor("pca", 3))
    S = factor_covariance(fit, spec, resid).dense()
    assert S.shape == (20, 20) and np.linalg.eigvalsh(0.5 * (S + S.T)).min() > 0


def test_tests_signs():
    rng = np.random.default_rng(3)
    a = rng.normal(0.001, 0.01, 2000)
    b = rng.normal(0.001, 0.02, 2000)
    t, p = variance_test(a, b)
    assert t > 0 and p < 0.01               # a has lower variance -> positive
    t2, _ = sharpe_test(a + 0.002, a)
    assert t2 > 0
    t3, p3 = dm_test(rng.normal(1.0, 0.1, 200), rng.normal(2.0, 0.1, 200))
    assert t3 > 0


def test_xs_rank_factors_miss_market_raw_do_not(synth):
    """Cross-sectional ranks remove the common (market) direction; raw-return PCA keeps it."""
    R = synth["R"].iloc[:300, :30]
    inp = RiskModelInput(asof=R.index[-1], returns=R)
    ew = R.mean(axis=1).to_numpy()
    corr = lambda fit: max(abs(np.corrcoef(fit.F[:, k], ew)[0, 1]) for k in range(fit.K))  # noqa: E731
    raw = FactorFit(inp, FactorExtractor("pca", 3, {}), transform="raw")
    xs = FactorFit(inp, FactorExtractor("pca", 3, {}))
    assert corr(raw) > 0.9
    assert np.nanmean(np.abs(np.corrcoef(raw.U, rowvar=False)[np.triu_indices(30, 1)])) < \
           np.nanmean(np.abs(np.corrcoef(xs.U, rowvar=False)[np.triu_indices(30, 1)]))
