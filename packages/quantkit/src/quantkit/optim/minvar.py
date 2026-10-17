"""Minimum-variance portfolio construction.

* :func:`gmv_closed_form` — unconstrained global minimum variance; uses the Woodbury identity
  when the covariance is in factor form (only a K x K inverse is needed).
* :func:`min_variance` — long-only QP with a weight cap, an optional per-group (industry) cap,
  frozen positions (A-share trading restrictions) and an optional L1 turnover penalty
  (Olivares-Nadal & DeMiguel, 2018).
  Factor-form covariances are passed to the solver in factor form, never densified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from ..riskmodels.base import CovarianceRepr, IndustryBlockResidual


def precision(cov: CovarianceRepr) -> np.ndarray:
    if not cov.is_factor:
        return np.linalg.inv(cov.dense())
    Su_inv = cov.residual_inv()
    B, Sf = cov.B, cov.sigma_f
    M = np.linalg.inv(Sf) + B.T @ Su_inv @ B           # K x K
    SB = Su_inv @ B
    return Su_inv - SB @ np.linalg.solve(M, SB.T)


def gmv_closed_form(cov: CovarianceRepr) -> pd.Series:
    P = precision(cov)
    w = P.sum(axis=1)
    return pd.Series(w / w.sum(), index=cov.codes)


@dataclass
class Constraints:
    max_weight: float = 0.10
    long_only: bool = True
    frozen: dict[str, float] = field(default_factory=dict)   # code -> weight that must be kept
    turnover_penalty: float = 0.0                            # kappa in min w'Σw + kappa ||w - w0||_1
    prev_weights: pd.Series | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)  # per-asset (lo, hi)
    budget: float = 1.0                                      # sum of weights (1 - weight held outside)
    groups: pd.Series | None = None                          # code -> group id (e.g. SW level-1 industry)
    group_cap: float | None = None                           # max total weight per group


def _residual_sqrt(cov: CovarianceRepr) -> np.ndarray | None:
    """Matrix G with G'G = Σ_u, or None when Σ_u is diagonal (handled as a weighted sum)."""
    if isinstance(cov.sigma_u, IndustryBlockResidual):
        return np.linalg.cholesky(cov.sigma_u.dense() + 1e-12 * np.eye(len(cov.codes))).T
    return None


def min_variance(cov: CovarianceRepr, cons: Constraints | None = None) -> pd.Series:
    cons = cons or Constraints()
    codes = cov.codes
    n = len(codes)
    w = cp.Variable(n)

    if cov.is_factor:
        Lf = np.linalg.cholesky(cov.sigma_f + 1e-12 * np.eye(cov.sigma_f.shape[0]))
        risk = cp.sum_squares(Lf.T @ (cov.B.T @ w))
        G = _residual_sqrt(cov)
        if G is None:
            risk = risk + cp.sum_squares(cp.multiply(np.sqrt(np.asarray(cov.sigma_u)), w))
        else:
            risk = risk + cp.sum_squares(G @ w)
    else:
        S = cov.dense()
        S = 0.5 * (S + S.T)
        try:
            L = np.linalg.cholesky(S + 1e-12 * np.eye(n))
        except np.linalg.LinAlgError:
            evals, evecs = np.linalg.eigh(S)
            L = evecs * np.sqrt(evals.clip(min=1e-12))
        risk = cp.sum_squares(L.T @ w)

    constraints = [cp.sum(w) == cons.budget]
    if cons.long_only:
        constraints.append(w >= 0)
    if cons.max_weight is not None:
        constraints.append(w <= cons.max_weight)
    idx = {c: i for i, c in enumerate(codes)}
    for c, v in cons.frozen.items():
        if c in idx:
            constraints.append(w[idx[c]] == v)
    for c, (lo, hi) in cons.bounds.items():
        if c in idx:
            constraints += [w[idx[c]] >= lo, w[idx[c]] <= hi]
    if cons.group_cap is not None and cons.groups is not None:
        g = cons.groups.reindex(codes).to_numpy()
        for gid in pd.unique(g):
            members = np.flatnonzero(g == gid)
            if len(members) * (cons.max_weight if cons.max_weight is not None else 1.0) > cons.group_cap:
                constraints.append(cp.sum(w[members]) <= cons.group_cap)

    objective = risk
    if cons.turnover_penalty > 0 and cons.prev_weights is not None:
        w0 = cons.prev_weights.reindex(codes).fillna(0.0).to_numpy()
        objective = objective + cons.turnover_penalty * cp.norm1(w - w0)

    # daily variances are ~1e-4; rescale so solver tolerances are meaningful (argmin unchanged)
    if cov.is_factor:
        diag = np.einsum("ik,kl,il->i", cov.B, cov.sigma_f, cov.B) + np.diag(cov.residual_dense())
    else:
        diag = np.diag(cov.dense())
    scale = 1.0 / max(float(np.mean(diag)), 1e-12)
    prob = cp.Problem(cp.Minimize(scale * objective), constraints)
    for solver in ("CLARABEL", "OSQP", "SCS"):
        if solver in cp.installed_solvers():
            try:
                prob.solve(solver=solver)
            except cp.error.SolverError:
                continue
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
    if w.value is None:
        raise RuntimeError(f"min_variance failed: {prob.status}")
    x = np.asarray(w.value)
    if cons.long_only:
        x = x.clip(min=0.0)
    return pd.Series(x * (cons.budget / x.sum()), index=codes)
