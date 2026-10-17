"""SIC — structured industry-block residual covariance (DeFaCo module 3, methodology eqs. 6-8).

    σ̂²_i  = EWMA of u²_i (half-life h_u)             or a GARCH forecast (DEC specification)
    σ̃²_i  = (1 − α) σ̂²_i + α · median_j σ̂²_j           shrink the diagonal to the cross-section
    e_i,d = u_i,d / σ̂_i
    ρ̂_g   = (Var(Σ_{i∈g} e_i,d) − n_g) / (n_g (n_g − 1))  moment estimate of the within-industry
                                                         equicorrelation (same EWMA weights)
    ρ̃_g   = clip(β ρ̂_g, −1/(n_g−1) + ε, 1 − ε)
    Σ_u   = D̃^{1/2} blkdiag((1 − ρ̃_g) I + ρ̃_g 11') D̃^{1/2}

α = β = 0 gives the diagonal (EFM) residual, so SIC strictly generalises the benchmark paper.
Parameters grow like N + G (+2), not N². In the two-stage ablation (M2) α and β are chosen on
the validation sample; DeFaCo (M3/M4) learns them end to end.
"""
from __future__ import annotations

import numpy as np

from .base import IndustryBlockResidual

EPS = 1e-4


def ewma_weights(T: int, half_life: float) -> np.ndarray:
    w = 0.5 ** (np.arange(T)[::-1] / half_life)
    return w / w.sum()


def sic_residual(U: np.ndarray, industries: np.ndarray, alpha: float = 0.0, beta: float = 0.0,
                 half_life: float = 63.0, variances: np.ndarray | None = None) -> IndustryBlockResidual:
    """U: T x N residuals (demeaned, missing = 0); industries: length-N labels.
    ``variances`` overrides the EWMA diagonal (e.g. GARCH forecasts for DEC)."""
    T, N = U.shape
    w = ewma_weights(T, half_life)
    s2_hat = (w[:, None] * U ** 2).sum(axis=0)
    s2_hat = np.maximum(s2_hat, 1e-12)
    base = np.asarray(variances, float) if variances is not None else s2_hat
    base = np.maximum(base, 1e-12)
    s2 = (1 - alpha) * base + alpha * np.median(base)
    e = U / np.sqrt(s2_hat)
    groups = np.asarray(industries).astype(str)
    rho = {}
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        n = len(idx)
        if n < 2 or g in ("NA", "nan", "None") or beta == 0:
            continue
        agg = e[:, idx].sum(axis=1)
        var_agg = float((w * (agg - (w * agg).sum()) ** 2).sum())
        # E[Var(Σe)] = Σ Var(e_i) + Σ_{i≠j} ρ = (Σ Var(e_i)) + n(n−1)ρ with Var(e_i) ≈ 1 by construction
        var_ind = float((w[:, None] * (e[:, idx] - (w[:, None] * e[:, idx]).sum(0)) ** 2).sum())
        rho_hat = (var_agg - var_ind) / (n * (n - 1))
        if np.isfinite(rho_hat):
            rho[g] = float(np.clip(beta * rho_hat, -1.0 / (n - 1) + EPS, 1 - EPS))
    return IndustryBlockResidual(sigma2=s2, groups=groups, rho=rho)


def block_rho_summary(res: IndustryBlockResidual) -> dict:
    vals = np.array(list(res.rho.values())) if res.rho else np.array([0.0])
    return {"n_blocks": len(res.rho), "rho_mean": float(vals.mean()), "rho_max": float(vals.max())}
