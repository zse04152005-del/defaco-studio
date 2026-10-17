"""Benchmark covariance estimators used in the paper and offered in the demo.

* Sample covariance
* Linear shrinkage, Ledoit & Wolf (2004)            -> "LINS" in the benchmark paper
* OAS shrinkage, Chen et al. (2010)
* Analytical nonlinear shrinkage, Ledoit & Wolf (2020) -> "NLS"
* EWMA covariance (RiskMetrics style)
* Statistical PCA factor model, exact factor form    -> "SFC-PCA" of Conlon et al. (2025)
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import OAS, LedoitWolf

from .base import CovarianceRepr, RiskModel, RiskModelInput, demeaned, register


class _DenseModel(RiskModel):
    def __init__(self) -> None:
        self._cov: CovarianceRepr | None = None

    def covariance(self) -> CovarianceRepr:
        if self._cov is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        return self._cov


@register("sample")
class SampleCovariance(_DenseModel):
    name = "sample"

    def fit(self, data: RiskModelInput) -> "SampleCovariance":
        x = demeaned(data.returns)
        self._cov = CovarianceRepr(data.codes, dense_cov=x.T @ x / max(len(x) - 1, 1))
        return self


@register("lw_linear")
class LedoitWolfLinear(_DenseModel):
    name = "lw_linear"

    def fit(self, data: RiskModelInput) -> "LedoitWolfLinear":
        self.estimator_ = LedoitWolf(assume_centered=True).fit(demeaned(data.returns))
        self._cov = CovarianceRepr(data.codes, dense_cov=self.estimator_.covariance_)
        return self

    def explain(self) -> dict:
        return {"shrinkage": float(self.estimator_.shrinkage_)}


@register("oas")
class OASShrinkage(_DenseModel):
    name = "oas"

    def fit(self, data: RiskModelInput) -> "OASShrinkage":
        self.estimator_ = OAS(assume_centered=True).fit(demeaned(data.returns))
        self._cov = CovarianceRepr(data.codes, dense_cov=self.estimator_.covariance_)
        return self


def analytical_nonlinear_shrinkage(x: np.ndarray, demeaned_input: bool = True) -> np.ndarray:
    """Ledoit & Wolf (2020), 'Analytical nonlinear shrinkage of large-dimensional covariance
    matrices', Annals of Statistics 48(5). Port of the authors' published Matlab algorithm.

    ``x`` is T x N; if it was demeaned the effective sample size is T-1.
    """
    n, p = x.shape
    if demeaned_input:
        n = n - 1
    sample = x.T @ x / n
    lam, u = np.linalg.eigh(sample)                 # ascending
    lam = lam.clip(min=0.0)
    lam_nz = lam[max(0, p - n):]                    # non-zero eigenvalues
    m = len(lam_nz)
    L = np.tile(lam_nz, (m, 1)).T                   # L[i, j] = lam_i
    h = n ** (-1.0 / 3.0)
    H = h * L.T                                     # H[i, j] = h * lam_j
    xx = (L - L.T) / H
    ftilde = (3.0 / 4.0 / np.sqrt(5.0)) * np.mean(np.maximum(1.0 - xx ** 2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = (-3.0 / 10.0 / np.pi) * xx + (3.0 / 4.0 / np.sqrt(5.0) / np.pi) * (1.0 - xx ** 2 / 5.0) \
            * np.log(np.abs((np.sqrt(5.0) - xx) / (np.sqrt(5.0) + xx)))
    edge = np.isclose(np.abs(xx), np.sqrt(5.0))
    hf[edge] = (-3.0 / 10.0 / np.pi) * xx[edge]
    Hftilde = np.mean(hf / H, axis=1)
    if p <= n:
        c = p / n
        dtilde = lam_nz / ((np.pi * c * lam_nz * ftilde) ** 2 + (1.0 - c - np.pi * c * lam_nz * Hftilde) ** 2)
    else:
        Hftilde0 = (1.0 / np.pi) * (3.0 / 10.0 / h ** 2 + 3.0 / 4.0 / np.sqrt(5.0) / h * (1.0 - 1.0 / 5.0 / h ** 2)
                                     * np.log((1.0 + np.sqrt(5.0) * h) / (1.0 - np.sqrt(5.0) * h))) * np.mean(1.0 / lam_nz)
        dtilde0 = 1.0 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lam_nz / (np.pi ** 2 * lam_nz ** 2 * (ftilde ** 2 + Hftilde ** 2))
        dtilde = np.concatenate([np.full(p - n, dtilde0), dtilde1])
    return (u * dtilde) @ u.T


@register("lw_nonlinear")
class LedoitWolfNonlinear(_DenseModel):
    name = "lw_nonlinear"

    def fit(self, data: RiskModelInput) -> "LedoitWolfNonlinear":
        self._cov = CovarianceRepr(data.codes, dense_cov=analytical_nonlinear_shrinkage(demeaned(data.returns)))
        return self


@register("ewma")
class EWMACovariance(_DenseModel):
    name = "ewma"

    def __init__(self, half_life: float = 63.0) -> None:
        super().__init__()
        self.half_life = half_life

    def fit(self, data: RiskModelInput) -> "EWMACovariance":
        x = demeaned(data.returns)
        T = len(x)
        w = 0.5 ** (np.arange(T)[::-1] / self.half_life)
        w /= w.sum()
        self._cov = CovarianceRepr(data.codes, dense_cov=(x * w[:, None]).T @ x)
        return self


def poet_covariance(x: np.ndarray, k_max: int = 8, C: float = 0.5) -> tuple[np.ndarray, int, float]:
    """POET (Fan, Liao & Mincheva, 2013): K principal components of the sample covariance plus an
    adaptively soft-thresholded residual covariance, τ_ij = C·√(log N / T)·√θ_ij.
    K by the Bai-Ng (2002) IC_p2 criterion; C is raised until the estimate is positive definite."""
    T, N = x.shape
    S = x.T @ x / T
    lam, V = np.linalg.eigh(S)
    lam, V = lam[::-1], V[:, ::-1]
    tot = np.clip(lam, 0, None)
    pen = (N + T) / (N * T) * np.log(min(N, T))
    ic = [np.log(max(tot[k:].sum() / N, 1e-300)) + k * pen for k in range(0, min(k_max, N - 1) + 1)]
    K = max(1, int(np.argmin(ic)))
    L = (V[:, :K] * lam[:K]) @ V[:, :K].T
    u = x - x @ V[:, :K] @ V[:, :K].T
    R = u.T @ u / T
    theta = (u ** 2).T @ (u ** 2) / T - R ** 2
    base = np.sqrt(np.log(N) / T) * np.sqrt(np.clip(theta, 0, None))
    for c in np.arange(C, 3.01, 0.1):
        Rt = np.sign(R) * np.clip(np.abs(R) - c * base, 0, None)
        np.fill_diagonal(Rt, np.diag(R))
        Sig = L + Rt
        if np.linalg.eigvalsh(Sig).min() > 1e-12 * np.trace(Sig) / N:
            return Sig, K, float(c)
    return L + np.diag(np.diag(R)), K, float("inf")


@register("poet")
class POETCovariance(_DenseModel):
    name = "poet"

    def __init__(self, k_max: int = 8, C: float = 0.5) -> None:
        super().__init__()
        self.k_max, self.C = k_max, C

    def fit(self, data: RiskModelInput) -> "POETCovariance":
        S, self.K_, self.C_ = poet_covariance(demeaned(data.returns), self.k_max, self.C)
        self._cov = CovarianceRepr(data.codes, dense_cov=S)
        return self

    def explain(self) -> dict:
        return {"K": self.K_, "C": self.C_}


@register("pca_factor")
class PCAFactorModel(RiskModel):
    """Static exact factor model with K principal-component factors (SFC-PCA).

    Factors are the top-K principal components of the window's returns; loadings are the
    OLS betas on those factors; the residual covariance is diagonal.
    """
    name = "pca_factor"

    def __init__(self, n_factors: int = 5) -> None:
        self.n_factors = n_factors
        self._cov: CovarianceRepr | None = None

    def fit(self, data: RiskModelInput) -> "PCAFactorModel":
        x = demeaned(data.returns)
        T, N = x.shape
        K = min(self.n_factors, N - 1, T - 1)
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        F = x @ vt[:K].T                                   # T x K factor returns
        B = np.linalg.lstsq(F, x, rcond=None)[0].T          # N x K OLS loadings
        resid = x - F @ B.T
        sigma_f = F.T @ F / (T - 1)
        sigma_u = np.maximum(resid.var(axis=0, ddof=1), 1e-12)
        self._cov = CovarianceRepr(data.codes, B=B, sigma_f=sigma_f, sigma_u=sigma_u)
        self.explained_ = float((s[:K] ** 2).sum() / (s ** 2).sum())
        return self

    def covariance(self) -> CovarianceRepr:
        if self._cov is None:
            raise RuntimeError("call fit() first")
        return self._cov

    def explain(self) -> dict:
        return {"n_factors": self.n_factors, "explained_variance": self.explained_}
