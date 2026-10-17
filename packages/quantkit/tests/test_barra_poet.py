"""External benchmarks: POET and the Barra-style industry + style factor model."""
import numpy as np
import pandas as pd
import pytest

from quantkit.ipca import CharacteristicPanel
from quantkit.riskmodels import RiskModelInput, build
from quantkit.riskmodels.barra import BarraFit, industry_dummies
from quantkit.riskmodels.baselines import poet_covariance
from quantkit.riskmodels.latent_factor import factor_covariance


def test_poet_recovers_factor_count_and_is_pd():
    rng = np.random.default_rng(0)
    T, N, K = 500, 80, 3
    F = rng.standard_normal((T, K)) * 0.02
    B = rng.standard_normal((N, K))
    x = F @ B.T + 0.01 * rng.standard_normal((T, N))
    S, k, c = poet_covariance(x - x.mean(0))
    assert k == K and np.isfinite(c)
    assert np.linalg.eigvalsh(S).min() > 0
    m = build("poet").fit(RiskModelInput(asof=pd.Timestamp("2020-12-31"), returns=pd.DataFrame(x)))
    assert m.explain()["K"] == K


def test_small_industries_are_pooled():
    D, names = industry_dummies(["A", "A", "A", "B", "NA", "C", "C", "C"], min_industry=3)
    assert names == ["A", "C", "OTHER"] and D.sum(axis=1).tolist() == [1.0] * 8


def test_barra_fit_recovers_style_returns():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2020-01-01", periods=260)
    codes = [f"{600000 + i:06d}.SH" for i in range(90)]
    month_ends = pd.Series(days).groupby(days.to_period("M")).max()
    base = rng.standard_normal((90, 2))
    raw = pd.concat([pd.DataFrame({"date": d, "code": codes, "size": base[:, 0] + 0.1 * rng.standard_normal(90),
                                   "beta_12m": base[:, 1] + 0.1 * rng.standard_normal(90)}) for d in month_ends])
    chars = CharacteristicPanel(raw, ["size", "beta_12m"])
    ind = np.repeat(["X", "Y", "Z"], 30)
    f_style = 0.01 * rng.standard_normal((260, 2))
    f_ind = 0.01 * rng.standard_normal((260, 3))
    D = np.repeat(np.eye(3), 30, axis=0)
    R = f_ind @ D.T + f_style @ base.T + 0.002 * rng.standard_normal((260, 90))
    inp = RiskModelInput(asof=days[-1], returns=pd.DataFrame(R, index=days, columns=codes))
    fit = BarraFit(inp, chars, ind, np.ones(90), styles=("size", "beta_12m"), ridge=1e-6)
    assert fit.factor_names == ["X", "Y", "Z", "Size", "Beta"]
    # style factor returns are recovered up to the exposure standardisation (scale ≈ sd of base)
    corr = [np.corrcoef(fit.F[30:, 3 + j], f_style[30:, j])[0, 1] for j in range(2)]
    assert min(corr) > 0.95
    for resid in ("efm", "afm"):
        S = factor_covariance(fit, "sfc", resid).dense()
        assert np.linalg.eigvalsh(S).min() > 0
