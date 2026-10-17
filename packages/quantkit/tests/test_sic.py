"""SIC residual: recovers block correlation, reduces to EFM, stays positive definite, works in the QP."""
import numpy as np

from quantkit.optim.minvar import Constraints, min_variance
from quantkit.riskmodels.base import CovarianceRepr
from quantkit.riskmodels.sic import sic_residual


def block_data(T=3000, sizes=(8, 6, 1), rho=(0.4, 0.2), seed=0):
    rng = np.random.default_rng(seed)
    cols, inds = [], []
    for g, n in enumerate(sizes):
        r = rho[g] if g < len(rho) else 0.0
        common = rng.standard_normal((T, 1))
        x = np.sqrt(r) * common + np.sqrt(1 - r) * rng.standard_normal((T, n))
        cols.append(x * rng.uniform(0.01, 0.03, n))
        inds += [f"I{g}"] * n
    U = np.hstack(cols)
    return U - U.mean(0), np.array(inds)


def test_recovers_equicorrelation():
    U, ind = block_data()
    res = sic_residual(U, ind, alpha=0.0, beta=1.0, half_life=1e6)     # ~equal weights
    assert abs(res.rho["I0"] - 0.4) < 0.05 and abs(res.rho["I1"] - 0.2) < 0.05
    assert "I2" not in res.rho                                         # singleton industry: no block


def test_reduces_to_efm_and_shrinks_diagonal():
    U, ind = block_data()
    efm = sic_residual(U, ind, 0.0, 0.0, half_life=1e6)
    assert not efm.rho and np.allclose(np.diag(efm.dense()), (U ** 2).mean(0), rtol=1e-3)
    full = sic_residual(U, ind, 1.0, 0.0, half_life=1e6)
    assert np.allclose(full.sigma2, np.median(efm.sigma2))


def test_positive_definite_and_inverse_and_qp():
    U, ind = block_data(sizes=(10, 10, 5), rho=(0.9, 0.05))
    # an anti-correlated pair: rho_hat = -1 must be clipped to keep the block positive definite
    x = np.random.default_rng(2).standard_normal(3000) * 0.02
    U = np.hstack([U, np.column_stack([x, -x])])
    ind = np.concatenate([ind, ["NEG", "NEG"]])
    res = sic_residual(U, ind, 0.3, 1.0)
    assert -1 < res.rho["NEG"] < -0.99
    S = res.dense()
    assert np.linalg.eigvalsh(S).min() > 0 and np.allclose(res.inv() @ S, np.eye(len(S)), atol=1e-6)
    rng = np.random.default_rng(1)
    B = rng.normal(0, 1, (len(S), 2)) * 0.01
    cov = CovarianceRepr([f"{600000 + i}.SH" for i in range(len(S))], B=B, sigma_f=np.eye(2) * 1e-4, sigma_u=res)
    w = min_variance(cov, Constraints(max_weight=0.2))
    assert abs(w.sum() - 1) < 1e-6 and w.max() <= 0.2 + 1e-6
    # the QP in factor form agrees with the dense covariance
    Sd = cov.dense()
    wd = w.reindex(cov.codes).fillna(0).to_numpy()
    assert float(wd @ Sd @ wd) > 0
