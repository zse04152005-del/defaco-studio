"""Allocators beyond minimum variance: risk parity (ERC), HRP and mean-variance.

All take a ``CovarianceRepr`` (so every risk model of the project can drive them) and return
long-only weights summing to one, indexed by the covariance's codes.

* Risk parity / equal risk contribution — Spinu (2013): the convex programme
      min_y  ½ y'Σy − Σ_i b_i log y_i ,   y > 0,   w = y / Σy
  has risk contributions w_i(Σw)_i proportional to the budgets b_i at its optimum.
* Hierarchical risk parity — López de Prado (2016): correlation-distance clustering,
  quasi-diagonal ordering, recursive bisection with inverse-variance cluster weights.
* Mean-variance — max w'μ − (γ/2) w'Σw under the same constraints as minimum variance, with
  expected returns shrunk by Jorion's (1986) Bayes-Stein estimator towards the return of the
  minimum-variance portfolio (raw sample means are far too noisy to optimise on).

ERC and HRP have no native weight caps; ``cap_weights`` enforces a cap afterwards by clipping
and redistributing the excess pro rata (exact for the cap, approximate for the risk budgets).
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from ..riskmodels.base import CovarianceRepr
from .minvar import Constraints, min_variance


def cap_weights(w: pd.Series, cap: float | None, iters: int = 100) -> pd.Series:
    """Clip weights at ``cap`` and hand the excess to uncapped names pro rata (sums to 1)."""
    w = w.clip(lower=0) / w.clip(lower=0).sum()
    if cap is None or cap >= 1 or cap * len(w) < 1 - 1e-9:
        return w
    for _ in range(iters):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def risk_contributions(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = S @ w
    return w * m / float(w @ m)


def risk_parity(cov: CovarianceRepr, budgets: np.ndarray | None = None, cap: float | None = None) -> pd.Series:
    S = cov.dense()
    n = S.shape[0]
    b = np.ones(n) / n if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    scale = float(np.mean(np.diag(S)))
    y = cp.Variable(n, pos=True)
    L = np.linalg.cholesky(S / scale + 1e-10 * np.eye(n))
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(L.T @ y) - b @ cp.log(y)))
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        prob.solve()
    if y.value is None:
        raise RuntimeError(f"risk parity failed: {prob.status}")
    w = pd.Series(np.maximum(y.value, 0) / np.maximum(y.value, 0).sum(), index=cov.codes)
    return cap_weights(w, cap)


def _cluster_var(S: np.ndarray, idx: list[int]) -> float:
    sub = S[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    iv /= iv.sum()
    return float(iv @ sub @ iv)


def hrp(cov: CovarianceRepr, cap: float | None = None, method: str = "single") -> pd.Series:
    S = cov.dense()
    sd = np.sqrt(np.clip(np.diag(S), 1e-16, None))
    C = np.clip(S / np.outer(sd, sd), -1, 1)
    D = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    np.fill_diagonal(D, 0.0)
    order = list(leaves_list(linkage(squareform(D, checks=False), method=method)))
    w = np.ones(len(order))
    clusters = [order]
    while clusters:                      # recursive bisection
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            a, b = c[: len(c) // 2], c[len(c) // 2:]
            va, vb = _cluster_var(S, a), _cluster_var(S, b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return cap_weights(pd.Series(w / w.sum(), index=cov.codes), cap)


def bayes_stein_mean(returns: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, float]:
    """Jorion (1986) shrinkage of sample means (T x N returns) towards the GMV portfolio's mean.
    Returns (shrunk means, shrinkage intensity in [0, 1])."""
    x = np.nan_to_num(returns)
    T, N = x.shape
    mu = x.mean(axis=0)
    Si = np.linalg.pinv(S)
    one = np.ones(N)
    mu0 = float(one @ Si @ mu / (one @ Si @ one))
    d = mu - mu0 * one
    q = float(d @ Si @ d)
    delta = float(min(1.0, (N + 2) / ((N + 2) + T * q))) if q > 0 else 1.0
    return (1 - delta) * mu + delta * mu0 * one, delta


def mean_variance(cov: CovarianceRepr, mu: np.ndarray, risk_aversion: float = 5.0,
                  constraints: Constraints | None = None) -> pd.Series:
    """max w'μ − (γ/2) w'Σw, long-only with the minimum-variance constraint set."""
    cons = constraints or Constraints()
    S = cov.dense()
    n = S.shape[0]
    scale = float(np.mean(np.diag(S)))
    w = cp.Variable(n)
    L = np.linalg.cholesky(S / scale + 1e-10 * np.eye(n))
    obj = cp.Maximize((mu / scale) @ w - 0.5 * risk_aversion * cp.sum_squares(L.T @ w))
    cl = [cp.sum(w) == 1, w >= 0]
    if cons.max_weight is not None:
        cl.append(w <= cons.max_weight)
    if cons.groups is not None and cons.group_cap is not None:
        g = pd.Series(cons.groups).reindex(list(cov.codes))
        for name in g.dropna().unique():
            mask = (g == name).to_numpy()
            cl.append(cp.sum(w[mask]) <= cons.group_cap)
    prob = cp.Problem(obj, cl)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        prob.solve()
    if w.value is None:
        raise RuntimeError(f"mean-variance failed: {prob.status}")
    v = np.clip(w.value, 0, None)
    return pd.Series(v / v.sum(), index=cov.codes)
