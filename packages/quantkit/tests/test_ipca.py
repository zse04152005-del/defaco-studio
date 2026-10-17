"""IPCA (M1): recovery on data simulated from the model, point-in-time characteristics."""
import numpy as np
import pandas as pd
import pytest

from quantkit.ipca import CharacteristicPanel, ipca_fit
from quantkit.riskmodels import RiskModelInput
from quantkit.riskmodels.ipca_factor import IPCAFit, ipca_covariance


def simulate(T=320, N=150, n_chars=5, K=2, noise=0.01, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=T)
    month_ends = pd.Series(days).groupby(days.to_period("M")).max()
    codes = [f"{600000 + i:06d}.SH" for i in range(N)]
    # slowly evolving characteristics, snapshot at every month end
    base = rng.standard_normal((N, n_chars))
    rows = []
    for j, d in enumerate(month_ends):
        x = base + 0.2 * rng.standard_normal((N, n_chars))
        df = pd.DataFrame(x, columns=[f"c{k}" for k in range(n_chars)])
        df["date"], df["code"] = d, codes
        rows.append(df)
    raw = pd.concat(rows)
    chars = CharacteristicPanel(raw, [f"c{k}" for k in range(n_chars)])
    G = np.linalg.qr(rng.standard_normal((n_chars + 1, K)))[0]
    F = 0.01 * rng.standard_normal((T, K)) + 0.0005
    Zm, idx = chars.window(days, codes)
    R = np.einsum("tna,ak,tk->tn", Zm[idx], G, F) + noise * rng.standard_normal((T, N))
    R[rng.random((T, N)) < 0.03] = np.nan                     # suspensions
    return pd.DataFrame(R, index=days, columns=codes), chars, G, Zm, idx


def test_snapshot_is_point_in_time():
    R, chars, *_ = simulate(T=60)
    d = chars.dates[1]
    assert chars.snapshot_for(d, strict=True) == chars.dates[0]      # returns on d use last month's Z
    assert chars.snapshot_for(d, strict=False) == d                   # forecast at d uses Z_d
    assert chars.snapshot_for(chars.dates[0], strict=True) is None
    Z = chars.matrix(d, list(R.columns[:10]) + ["999999.SZ"])
    assert Z.shape == (11, 6) and (Z[:, 0] == 1).all() and (Z[-1, 1:] == 0).all()
    assert Z[:, 1:].max() < 0.5 and Z[:, 1:].min() > -0.5


def test_ipca_recovers_loading_space():
    R, chars, G, Zm, idx = simulate(noise=0.002)
    ok = idx > 0                                                    # drop days before the first snapshot
    res = ipca_fit(R.to_numpy()[ok], Zm, idx[ok], K=2)
    assert np.allclose(res.gamma.T @ res.gamma, np.eye(2), atol=1e-8)
    assert np.allclose(np.corrcoef(res.factors, rowvar=False)[0, 1], 0, atol=1e-6)
    # the true Γ is defined on raw characteristics but Z is rank-normalised, so compare fit quality
    assert res.r2 > 0.8 and res.converged


@pytest.mark.parametrize("spec", ["sfc", "dfc", "dec"])
@pytest.mark.parametrize("resid", ["efm", "afm"])
def test_ipca_covariance_psd(spec, resid):
    R, chars, *_ = simulate(T=280, N=40)
    R = R.iloc[25:]                                                 # window starts after the first snapshot
    fit = IPCAFit(RiskModelInput(asof=R.index[-1], returns=R), chars, K=3)
    cov = ipca_covariance(fit, spec, resid, 21).dense()
    assert cov.shape == (40, 40)
    assert np.linalg.eigvalsh(0.5 * (cov + cov.T)).min() > 0
    assert fit.B.shape == (40, 3)
    with pytest.raises(ValueError):
        ipca_covariance(fit, "dbc", resid)


def test_smoothed_characteristics_average_snapshots():
    R, chars, *_ = simulate(T=120)
    raw = pd.concat([pd.DataFrame({"date": d, "code": s.index, **{c: s[c].values for c in chars.cols}})
                     for d, s in chars._snap.items()])
    sm = CharacteristicPanel(raw, chars.cols, smooth=3)
    codes = list(R.columns[:20])
    d = chars.dates[3]
    expect = np.mean([chars.matrix(chars.dates[i], codes) for i in (1, 2, 3)], axis=0)
    assert np.allclose(sm.matrix(d, codes), expect)
    assert np.allclose(sm.matrix(chars.dates[0], codes), chars.matrix(chars.dates[0], codes))


def test_ipca_plus_limits_and_free_part():
    """Returns with a stock-specific loading component the characteristics cannot explain."""
    R, chars, G, Zm, idx = simulate(noise=0.002, seed=3)
    rng = np.random.default_rng(5)
    T, N = R.shape
    extra = rng.normal(0, 0.6, (N, 2))                       # free loadings
    F = 0.01 * rng.standard_normal((T, 2))
    R2 = R + (F @ extra.T)
    ok = idx > 0
    from quantkit.ipca import ipca_plus_fit
    big = ipca_plus_fit(R2.to_numpy()[ok], Zm, idx[ok], K=2, c=1e6)          # ≈ IPCA
    ip = ipca_fit(R2.to_numpy()[ok], Zm, idx[ok], K=2)
    small = ipca_plus_fit(R2.to_numpy()[ok], Zm, idx[ok], K=2, c=0.01)
    assert abs(big.r2 - ip.r2) < 0.01 and big.free_share < 0.01
    assert small.r2 > ip.r2 + 0.02 and small.free_share > 0.05
    assert np.allclose(small.gamma.T @ small.gamma, np.eye(2), atol=1e-8)


def test_ipca_plus_covariance_psd():
    from quantkit.riskmodels.ipca_factor import IPCAPlusFit
    R, chars, *_ = simulate(T=280, N=40)
    R = R.iloc[25:]
    fit = IPCAPlusFit(RiskModelInput(asof=R.index[-1], returns=R), chars, K=3, c=1.0)
    for spec in ("sfc", "dfc", "dec"):
        cov = ipca_covariance(fit, spec, "efm", 21).dense()
        assert np.linalg.eigvalsh(0.5 * (cov + cov.T)).min() > 0


def test_ipca_plus_factor_returns_bounded_on_weak_directions():
    """Regression (2022-06 blow-up): a characteristic that is ~0 inside the window must not get an
    unbounded factor return. The relative ridge κ·tr(B_d'B_d)/K keeps F on the scale of IPCA's."""
    from quantkit.ipca import ipca_plus_fit
    R, chars, G, Zm, idx = simulate(T=200, N=80, K=2)
    Zm = Zm.copy()
    Zm[..., -1] = 1e-6 * np.random.default_rng(3).standard_normal(Zm.shape[:2])   # near-constant column
    r = R.to_numpy()
    ok = ~np.isnan(r).all(axis=1)
    base = ipca_fit(r[ok], Zm, idx[ok], K=3)
    for c in (0.1, 1.0):
        res = ipca_plus_fit(r[ok], Zm, idx[ok], K=3, c=c)
        assert np.isfinite(res.factors).all()
        assert res.factors.std(axis=0).max() < 5 * base.factors.std(axis=0).max()
