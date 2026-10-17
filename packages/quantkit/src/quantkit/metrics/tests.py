"""HAC-robust tests used for the paper tables.

* :func:`variance_test` — equality of variances (Ledoit & Wolf, 2011, *Wilmott*): log-variance
  difference, delta method on the first two moments, HAC covariance.
* :func:`sharpe_test`   — equality of Sharpe ratios via the same delta-method/HAC framework
  (Ledoit & Wolf, 2008; asymptotically equivalent to the non-iid Opdyke (2007) test used in the
  benchmark paper).
* :func:`dm_test`       — Diebold-Mariano test of equal predictive loss, Newey-West HAC.

Sign convention (as in the benchmark paper's Table 5): a *positive* statistic means strategy
``a`` outperforms benchmark ``b`` (lower variance / higher Sharpe / lower loss).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def hac(g: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Newey-West (Bartlett) long-run covariance of the rows of ``g`` (T x p), demeaned."""
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    T = g.shape[0]
    g = g - g.mean(axis=0)
    L = lags if lags is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    S = g.T @ g / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = g[l:].T @ g[:-l] / T
        S += w * (G + G.T)
    return S


def _moments(a: np.ndarray, b: np.ndarray):
    return np.column_stack([a, b, a * a, b * b])


def variance_test(a: np.ndarray, b: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """t-stat and two-sided p-value for H0: var(a) = var(b); positive t => var(a) < var(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    m = _moments(a, b).mean(axis=0)
    va, vb = m[2] - m[0] ** 2, m[3] - m[1] ** 2
    delta = np.log(vb) - np.log(va)
    grad = np.array([2 * m[0] / va, -2 * m[1] / vb, -1 / va, 1 / vb])
    se = np.sqrt(grad @ hac(_moments(a, b), lags) @ grad / T)
    t = delta / se
    return float(t), float(2 * (1 - norm.cdf(abs(t))))


def sharpe_test(a: np.ndarray, b: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """t-stat and p-value for H0: SR(a) = SR(b) (excess returns); positive t => SR(a) > SR(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    m = _moments(a, b).mean(axis=0)
    va, vb = m[2] - m[0] ** 2, m[3] - m[1] ** 2
    delta = m[0] / np.sqrt(va) - m[1] / np.sqrt(vb)
    grad = np.array([m[2] / va ** 1.5, -m[3] / vb ** 1.5, -0.5 * m[0] / va ** 1.5, 0.5 * m[1] / vb ** 1.5])
    se = np.sqrt(grad @ hac(_moments(a, b), lags) @ grad / T)
    t = delta / se
    return float(t), float(2 * (1 - norm.cdf(abs(t))))


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 12) -> tuple[float, float]:
    """Diebold-Mariano; positive t => model a has *lower* average loss than b."""
    d = np.asarray(loss_b, float) - np.asarray(loss_a, float)
    d = d[~np.isnan(d)]
    T = len(d)
    se = np.sqrt(hac(d, lags)[0, 0] / T)
    t = d.mean() / se if se > 0 else 0.0
    return float(t), float(2 * (1 - norm.cdf(abs(t))))


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
