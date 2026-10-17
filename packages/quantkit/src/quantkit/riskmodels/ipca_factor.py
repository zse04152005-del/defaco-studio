"""M1: IPCA characteristic-driven loadings in place of return-only latent factors.

Same covariance specifications as the M0 latent-factor models (see latent_factor.py),
with the loadings at the forecast date given by the latest characteristics:

    B_T = Z_T Γ,   Σ = B_T Σ_f B_T' + Σ_u

where Γ and the daily factor returns come from mixed-frequency IPCA over the estimation
window. ``dbc`` is not defined here: IPCA loadings are already conditional (they move with
the characteristics), which is exactly what DBC approximates with DCC-implied betas.
"""
from __future__ import annotations

import numpy as np

from ..dynamics import garch11
from ..ipca import CharacteristicPanel, ipca_fit, ipca_plus_fit
from .base import CovarianceRepr, RiskModel, RiskModelInput, demeaned, register
from .latent_factor import factor_covariance

SPECS = ("sfc", "dfc", "dec")


class IPCAFit:
    """Interface-compatible with latent_factor.FactorFit (B, F, U, Sf, K, codes)."""

    def __init__(self, data: RiskModelInput, chars: CharacteristicPanel, K: int, **kw):
        self.codes = data.codes
        R = data.returns.to_numpy(dtype=float)
        Zm, idx = chars.window(data.returns.index, list(self.codes))
        res = ipca_fit(R, Zm, idx, K, **kw)
        self.result = res
        self.gamma = res.gamma
        self.K = res.gamma.shape[1]
        self.x = demeaned(data.returns)
        fitted = np.einsum("tna,ak,tk->tn", Zm[idx], res.gamma, res.factors)
        U = np.where(np.isnan(R), 0.0, R - fitted)
        self.U = U - U.mean(axis=0)
        self.F = res.factors - res.factors.mean(axis=0)
        self.Sf = np.atleast_2d(np.cov(self.F, rowvar=False))
        self.Z_T = chars.matrix(chars.snapshot_for(data.asof, strict=False), list(self.codes))
        self.B = self.Z_T @ res.gamma                     # N x K loadings at the forecast date
        self._garch_x = None
        self._dbc_B = None

    def garch_assets(self, horizon: int):
        if self._garch_x is None:
            self._garch_x = [garch11(self.x[:, i], horizon) for i in range(self.x.shape[1])]
        return self._garch_x


class IPCAPlusFit(IPCAFit):
    """M1+: loadings B_T = Z_T Γ + B̃ (characteristic part + ridge-shrunk free part)."""

    def __init__(self, data: RiskModelInput, chars: CharacteristicPanel, K: int, c: float = 1.0, **kw):
        self.codes = data.codes
        R = data.returns.to_numpy(dtype=float)
        Zm, idx = chars.window(data.returns.index, list(self.codes))
        res = ipca_plus_fit(R, Zm, idx, K, c=c, **kw)
        self.result = res
        self.gamma = res.gamma
        self.free = res.free
        self.K = res.gamma.shape[1]
        self.x = demeaned(data.returns)
        fitted = np.einsum("tnk,tk->tn", np.einsum("tna,ak->tnk", Zm[idx], res.gamma) + res.free[None], res.factors)
        U = np.where(np.isnan(R), 0.0, R - fitted)
        self.U = U - U.mean(axis=0)
        self.F = res.factors - res.factors.mean(axis=0)
        self.Sf = np.atleast_2d(np.cov(self.F, rowvar=False))
        self.Z_T = chars.matrix(chars.snapshot_for(data.asof, strict=False), list(self.codes))
        self.B = self.Z_T @ res.gamma + res.free
        self._garch_x = None
        self._dbc_B = None


def ipca_covariance(fit: IPCAFit, spec: str, resid: str, horizon: int = 21) -> CovarianceRepr:
    if spec not in SPECS:
        raise ValueError(f"IPCA supports {SPECS}, got {spec!r}")
    return factor_covariance(fit, spec, resid, horizon)


@register("ipca_factor")
class IPCAFactorModel(RiskModel):
    def __init__(self, chars: CharacteristicPanel, spec: str = "sfc", resid: str = "efm", n_factors: int = 5,
                 horizon: int = 21):
        self.chars, self.spec, self.resid, self.K, self.horizon = chars, spec, resid, n_factors, horizon
        self.name = f"ipca-{spec}-{resid}"
        self._cov: CovarianceRepr | None = None

    def fit(self, data: RiskModelInput) -> "IPCAFactorModel":
        self.fit_ = IPCAFit(data, self.chars, self.K)
        self._cov = ipca_covariance(self.fit_, self.spec, self.resid, self.horizon)
        return self

    def covariance(self) -> CovarianceRepr:
        if self._cov is None:
            raise RuntimeError("call fit() first")
        return self._cov

    def explain(self) -> dict:
        """Characteristic importance: row norms of Γ (how strongly each characteristic drives loadings)."""
        g = np.linalg.norm(self.fit_.gamma, axis=1)
        return {"K": self.fit_.K, "r2": self.fit_.result.r2,
                "importance": dict(zip(self.chars.names, (g / g.sum()).round(4).tolist()))}
