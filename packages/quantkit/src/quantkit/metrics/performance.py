"""Portfolio performance, weight characteristics and covariance-forecast losses."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance(daily: pd.Series, rf_daily: float | pd.Series = 0.0) -> dict[str, float]:
    r = daily.dropna()
    ex = r - (rf_daily.reindex(r.index).fillna(0.0) if isinstance(rf_daily, pd.Series) else rf_daily)
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    ann_ret = float(nav.iloc[-1] ** (TRADING_DAYS / len(r)) - 1) if len(r) else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    down = ex[ex < 0]
    var95 = float(np.quantile(r, 0.05))
    return {
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std() > 0 else np.nan,
        "sortino": float(ex.mean() / np.sqrt((down ** 2).mean()) * np.sqrt(TRADING_DAYS)) if len(down) else np.nan,
        "max_drawdown": float(dd.min()),
        "cvar95_daily": float(r[r <= var95].mean()),
        "n_days": int(len(r)),
    }


def monthly_sd_sharpe(daily: pd.Series, rf_monthly: pd.Series | float = 0.0) -> dict[str, float]:
    """Monthly SD and Sharpe ratio, the headline metrics of the benchmark paper."""
    m = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    ex = m - rf_monthly
    return {"sd_monthly": float(m.std(ddof=1)), "sr_monthly": float(ex.mean() / ex.std(ddof=1))}


def weight_stats(weights: dict[pd.Timestamp, pd.Series], turnover: pd.Series) -> dict[str, float]:
    rows = []
    for w in weights.values():
        w = w[w > 1e-8]
        rows.append({"max_w": w.max(), "min_nz_w": w.min(), "hhi": float((w ** 2).sum()),
                     "n_nonzero": len(w), "eff_n": 1.0 / float((w ** 2).sum())})
    df = pd.DataFrame(rows)
    out = df.mean().to_dict()
    out["turnover"] = float(turnover.iloc[1:].mean()) if len(turnover) > 1 else np.nan  # skip initial build
    return out


# ---------------------------------------------------------------------- covariance losses
def cov_losses(forecast: np.ndarray, realized: np.ndarray, b: float = 3.0) -> dict[str, float]:
    """MSE, MAE, QLIKE and the asymmetric loss used by Conlon et al. (2025)."""
    N = forecast.shape[0]
    diff = forecast - realized
    sign, logdet = np.linalg.slogdet(forecast)
    qlk = float(logdet + np.trace(np.linalg.solve(forecast, realized))) if sign > 0 else np.nan
    # ASYM (Laurent, Rombouts & Violante 2013, b = 3): penalises over-prediction
    def mpow(A, p):
        ev, V = np.linalg.eigh(0.5 * (A + A.T))
        return (V * ev.clip(min=0) ** p) @ V.T
    asym = float(1 / (b * (b - 1)) * np.trace(mpow(realized, b) - mpow(forecast, b))
                 - 1 / (b - 1) * np.trace(mpow(forecast, b - 1) @ (realized - forecast)))
    return {"mse": float((diff ** 2).sum() / N), "mae": float(np.abs(diff).sum() / N ** 2),
            "qlike": qlk, "asym": asym}
