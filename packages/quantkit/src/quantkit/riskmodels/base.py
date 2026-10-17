"""RiskModel interface: the plug between the engine, the paper's DeFaCo model and the demo.

Every risk model consumes a :class:`RiskModelInput` (data known at ``asof``) and returns a
:class:`CovarianceRepr`, either dense or in factor form ``B Σ_f B' + Σ_u``. Optimizers and
risk attribution accept both forms, so a new model (e.g. DeFaCo) only has to implement
``fit`` / ``covariance`` to appear everywhere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- residual blocks
@dataclass
class IndustryBlockResidual:
    """Σ_u = D^{1/2} blkdiag((1-ρ_g) I + ρ_g 11') D^{1/2} (equicorrelation within industry)."""

    sigma2: np.ndarray            # (N,) residual variances
    groups: np.ndarray            # (N,) industry ids
    rho: dict[Any, float]         # industry id -> within-industry correlation

    def dense(self) -> np.ndarray:
        s = np.sqrt(self.sigma2)
        R = np.eye(len(s))
        for g, r in self.rho.items():
            idx = np.flatnonzero(self.groups == g)
            if len(idx) > 1 and r != 0.0:
                R[np.ix_(idx, idx)] = r
                R[idx, idx] = 1.0
        return s[:, None] * R * s[None, :]

    def inv(self) -> np.ndarray:
        """Closed-form inverse, block by block (O(N) per block)."""
        s_inv = 1.0 / np.sqrt(self.sigma2)
        Rinv = np.diag(np.ones(len(s_inv)))
        for g, r in self.rho.items():
            idx = np.flatnonzero(self.groups == g)
            n = len(idx)
            if n > 1 and r != 0.0:
                blk = (np.eye(n) - r / (1 - r + n * r) * np.ones((n, n))) / (1 - r)
                Rinv[np.ix_(idx, idx)] = blk
        return s_inv[:, None] * Rinv * s_inv[None, :]


Residual = np.ndarray | IndustryBlockResidual   # 1-D array = diagonal residual


@dataclass
class CovarianceRepr:
    codes: list[str]
    dense_cov: np.ndarray | None = None
    B: np.ndarray | None = None          # N x K loadings
    sigma_f: np.ndarray | None = None    # K x K
    sigma_u: Residual | None = None
    parts: dict[str, np.ndarray] | None = None   # optional additive decomposition of the dense Σ (e.g. factor / industry / specific)

    @property
    def is_factor(self) -> bool:
        return self.B is not None

    def residual_dense(self) -> np.ndarray:
        if isinstance(self.sigma_u, IndustryBlockResidual):
            return self.sigma_u.dense()
        return np.diag(np.asarray(self.sigma_u, dtype=float))

    def residual_inv(self) -> np.ndarray:
        if isinstance(self.sigma_u, IndustryBlockResidual):
            return self.sigma_u.inv()
        return np.diag(1.0 / np.asarray(self.sigma_u, dtype=float))

    def dense(self) -> np.ndarray:
        if self.dense_cov is not None:
            return self.dense_cov
        return self.B @ self.sigma_f @ self.B.T + self.residual_dense()

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.dense(), index=self.codes, columns=self.codes)


@dataclass
class RiskModelInput:
    asof: pd.Timestamp
    returns: pd.DataFrame                      # T x N daily returns in the estimation window
    characteristics: pd.DataFrame | None = None  # N x L at asof (rank-normalised)
    industries: pd.Series | None = None        # N industry ids at asof
    market: pd.Series | None = None            # T benchmark returns
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def codes(self) -> list[str]:
        return list(self.returns.columns)


class RiskModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, data: RiskModelInput) -> "RiskModel": ...

    @abstractmethod
    def covariance(self) -> CovarianceRepr: ...

    def exposures(self) -> pd.DataFrame | None:
        cov = self.covariance()
        if cov.B is None:
            return None
        return pd.DataFrame(cov.B, index=cov.codes, columns=[f"f{k + 1}" for k in range(cov.B.shape[1])])

    def explain(self) -> dict[str, Any]:
        return {}


def demeaned(returns: pd.DataFrame) -> np.ndarray:
    """Column-demeaned returns with missing values set to 0 (i.e. to the column mean)."""
    x = returns - returns.mean()
    return x.fillna(0.0).to_numpy(dtype=float)


# --------------------------------------------------------------------------- registry
_REGISTRY: dict[str, Callable[..., RiskModel]] = {}


def register(name: str):
    def deco(factory: Callable[..., RiskModel]):
        _REGISTRY[name] = factory
        return factory
    return deco


def build(name: str, **kwargs) -> RiskModel:
    if name not in _REGISTRY:
        raise KeyError(f"unknown risk model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
