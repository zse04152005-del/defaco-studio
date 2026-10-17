"""Univariate GARCH(1,1) and Engle (2002) DCC(1,1) for the dynamic covariance specifications.

Forecasts are averaged over the holding period (``horizon`` trading days), because the
portfolio is held for a month while the models are estimated on daily data.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class GarchResult:
    z: np.ndarray          # standardised residuals (T,)
    var_forecast: float    # average conditional variance over the horizon (return units^2)
    ok: bool


def garch11(x: np.ndarray, horizon: int = 21) -> GarchResult:
    """Zero-mean GARCH(1,1) on a demeaned series; falls back to the sample variance."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    s2 = float(x.var()) if x.var() > 0 else 1e-8
    try:
        from arch import arch_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale = 100.0
            res = arch_model(x * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False) \
                .fit(disp="off", show_warning=False, options={"maxiter": 200})
            if res.convergence_flag != 0:
                raise RuntimeError("no convergence")
            cv = res.conditional_volatility / scale
            f = res.forecast(horizon=horizon, reindex=False).variance.values[-1] / scale ** 2
            z = x / np.maximum(cv, 1e-12)
            return GarchResult(z=z, var_forecast=float(np.mean(f)), ok=True)
    except Exception:  # noqa: BLE001 - any estimation failure -> static fallback
        return GarchResult(z=x / np.sqrt(s2), var_forecast=s2, ok=False)


def _dcc_path(z: np.ndarray, a: float, b: float, Qbar: np.ndarray):
    """Correlation-part log-likelihood and Q_{T+1}; recursion in a loop, algebra batched."""
    T, m = z.shape
    Qs = np.empty((T, m, m))
    Q = Qbar.copy()
    c = (1 - a - b) * Qbar
    for t in range(T):
        Qs[t] = Q
        Q = c + a * np.outer(z[t], z[t]) + b * Q
    d = np.sqrt(np.clip(np.einsum("tii->ti", Qs), 1e-12, None))
    R = Qs / (d[:, :, None] * d[:, None, :])
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return -1e10, Q
    quad = np.einsum("ti,ti->t", z, np.linalg.solve(R, z[:, :, None])[:, :, 0])
    ll = -0.5 * float(np.sum(logdet + quad - np.einsum("ti,ti->t", z, z)))
    return ll, Q


@dataclass
class DCCResult:
    a: float
    b: float
    Qbar: np.ndarray
    Q_next: np.ndarray

    def corr_forecast(self, horizon: int = 21) -> np.ndarray:
        """Average correlation over the horizon: Q_{T+k} = Qbar + (a+b)^{k-1} (Q_{T+1} - Qbar)."""
        p = self.a + self.b
        w = np.mean([p ** (k - 1) for k in range(1, horizon + 1)])
        Q = self.Qbar + w * (self.Q_next - self.Qbar)
        d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
        return Q / np.outer(d, d)


def dcc11(z: np.ndarray) -> DCCResult:
    """Estimate DCC(1,1) by (correlation-part) quasi maximum likelihood on standardised residuals."""
    z = np.asarray(z, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Qbar = np.corrcoef(z, rowvar=False)
    Qbar = np.nan_to_num(np.atleast_2d(Qbar), nan=0.0)   # constant columns -> uncorrelated
    np.fill_diagonal(Qbar, 1.0)
    if z.shape[1] == 1:
        return DCCResult(0.0, 0.0, np.ones((1, 1)), np.ones((1, 1)))

    def nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.995:
            return 1e10
        return -_dcc_path(z, a, b, Qbar)[0]

    best = min(((nll(p0), p0) for p0 in ((0.02, 0.95), (0.05, 0.90), (0.01, 0.80))), key=lambda x: x[0])
    res = minimize(nll, x0=np.array(best[1]), method="L-BFGS-B", bounds=[(1e-6, 0.3), (0.0, 0.994)],
                   options={"maxiter": 60})
    a, b = (res.x if res.fun <= best[0] else best[1])
    _, Qn = _dcc_path(z, a, b, Qbar)
    return DCCResult(float(a), float(b), Qbar, Qn)


def dynamic_covariance(x: np.ndarray, horizon: int = 21) -> np.ndarray:
    """GARCH(1,1) variances + DCC(1,1) correlations, averaged over the horizon (m x m)."""
    g = [garch11(x[:, j], horizon) for j in range(x.shape[1])]
    z = np.column_stack([r.z for r in g])
    R = dcc11(z).corr_forecast(horizon)
    s = np.sqrt([r.var_forecast for r in g])
    return R * np.outer(s, s)
