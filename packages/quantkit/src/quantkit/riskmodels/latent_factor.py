"""Latent-factor covariance models of the benchmark paper (M0 replication).

Σ = B Σ_f B' + Σ_u with four specifications
    sfc — static factor covariance
    dfc — dynamic factor covariance: Σ_f,t from GARCH(1,1)+DCC(1,1) on the factors
    dbc — dynamic betas: β_i,t = Σ_f,t^{-1} σ_{f r_i, t} from a DCC on (factors, r_i)
    dec — dynamic error covariance: GARCH(1,1) residual variances
and two residual structures
    efm — exact factor model: diagonal Σ_u
    afm — approximate factor model: Ledoit-Wolf linearly shrunk residual covariance;
          for dec, GARCH variances x NLS-shrunk residual correlation (a static-correlation
          stand-in for the DCC-NL used in the paper; documented in the M0 notes).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from ..dynamics import dcc11, dynamic_covariance, garch11
from ..latent import FactorExtractor, rank_transform
from .base import CovarianceRepr, RiskModel, RiskModelInput, demeaned, register
from .baselines import analytical_nonlinear_shrinkage
from .sic import sic_residual

SPECS = ("sfc", "dfc", "dbc", "dec")


class FactorFit:
    """Factors + OLS loadings for one window; shared by all specs of a method.

    ``transform="xs_rank"`` follows the benchmark paper literally: factors are extracted from
    returns ranked cross-sectionally each day. Every day's ranks have the same mean, so these
    factors cannot span the market (equal-weight) direction. ``"raw"`` extracts them from the
    demeaned raw returns instead (the standard statistical factor model).
    """

    def __init__(self, data: RiskModelInput, extractor: FactorExtractor, transform: str = "xs_rank"):
        self.codes = data.codes
        self.x = demeaned(data.returns)                       # T x N raw (demeaned) returns
        if transform not in ("xs_rank", "raw"):
            raise ValueError(transform)
        z = rank_transform(data.returns.to_numpy(dtype=float)) if transform == "xs_rank" else self.x
        F = extractor.fit_transform(z, target=self.x.mean(axis=1))
        self.F = F - F.mean(axis=0)
        self.B = np.linalg.lstsq(self.F, self.x, rcond=None)[0].T   # N x K
        self.U = self.x - self.F @ self.B.T
        self.Sf = np.atleast_2d(np.cov(self.F, rowvar=False))
        self.K = self.F.shape[1]
        self._garch_x = None
        self._dbc_B = None

    def garch_assets(self, horizon: int):
        if self._garch_x is None:
            self._garch_x = [garch11(self.x[:, i], horizon) for i in range(self.x.shape[1])]
        return self._garch_x


def factor_covariance(fit: FactorFit, spec: str, resid: str, horizon: int = 21) -> CovarianceRepr:
    B, Sf, U = fit.B, fit.Sf, fit.U
    T = U.shape[0]
    if spec == "dfc":
        Sf = dynamic_covariance(fit.F, horizon) if fit.K > 1 else np.array([[garch11(fit.F[:, 0], horizon).var_forecast]])
    if spec == "dbc" and fit._dbc_B is not None:
        B = fit._dbc_B
    elif spec == "dbc":
        gF = [garch11(fit.F[:, k], horizon) for k in range(fit.K)]
        zF = np.column_stack([g.z for g in gF])
        sF = np.sqrt([g.var_forecast for g in gF])
        Bt = np.empty_like(B)
        for i, gi in enumerate(fit.garch_assets(horizon)):
            R = dcc11(np.column_stack([zF, gi.z])).corr_forecast(horizon)
            s = np.append(sF, np.sqrt(gi.var_forecast))
            H = R * np.outer(s, s)
            Bt[i] = np.linalg.solve(H[:-1, :-1] + 1e-12 * np.eye(fit.K), H[:-1, -1])
        B = fit._dbc_B = Bt
    # residual covariance
    if spec == "dec":
        g = [garch11(U[:, i], horizon) for i in range(U.shape[1])]
        d = np.array([r.var_forecast for r in g])
        if resid == "sic":
            a, b = getattr(fit, "sic_params", (0.0, 0.0))
            return CovarianceRepr(fit.codes, B=B, sigma_f=Sf, sigma_u=sic_residual(U, fit.industries, a, b, variances=d))
        if resid == "efm":
            return CovarianceRepr(fit.codes, B=B, sigma_f=Sf, sigma_u=np.maximum(d, 1e-12))
        zu = np.column_stack([r.z for r in g])
        C = analytical_nonlinear_shrinkage(zu - zu.mean(axis=0))
        sd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
        C = np.nan_to_num(C / np.outer(sd, sd))
        np.fill_diagonal(C, 1.0)
        Su = C * np.outer(np.sqrt(d), np.sqrt(d))
        return CovarianceRepr(fit.codes, dense_cov=B @ Sf @ B.T + Su)
    if resid == "efm":
        return CovarianceRepr(fit.codes, B=B, sigma_f=Sf, sigma_u=np.maximum(U.var(axis=0, ddof=1), 1e-12))
    if resid == "sic":     # M2: industry-block structured residual (needs fit.industries, fit.sic_params)
        a, b = getattr(fit, "sic_params", (0.0, 0.0))
        return CovarianceRepr(fit.codes, B=B, sigma_f=Sf, sigma_u=sic_residual(U, fit.industries, a, b))
    Su = LedoitWolf(assume_centered=True).fit(U).covariance_ * T / (T - 1)
    return CovarianceRepr(fit.codes, dense_cov=B @ Sf @ B.T + Su)


@register("latent_factor")
class LatentFactorModel(RiskModel):
    def __init__(self, method: str = "pca", spec: str = "sfc", resid: str = "efm", n_factors: int = 5,
                 params: dict | None = None, horizon: int = 21, seed: int = 0):
        self.method, self.spec, self.resid = method, spec, resid
        self.extractor = FactorExtractor(method, n_factors, params or {}, seed)
        self.horizon = horizon
        self.name = f"{method}-{spec}-{resid}"
        self._cov: CovarianceRepr | None = None

    def fit(self, data: RiskModelInput) -> "LatentFactorModel":
        self.fit_ = FactorFit(data, self.extractor)
        self._cov = factor_covariance(self.fit_, self.spec, self.resid, self.horizon)
        return self

    def covariance(self) -> CovarianceRepr:
        if self._cov is None:
            raise RuntimeError("call fit() first")
        return self._cov

    def explain(self) -> dict:
        return {"method": self.method, "spec": self.spec, "resid": self.resid, "K": self.fit_.K}
