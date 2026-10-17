"""Backtest statistics for the demo's reports.

Besides the usual return / risk numbers this reports how much a Sharpe ratio can be trusted:

* Sharpe standard error with non-normal returns (Mertens, 2002; Opdyke, 2007):
      SE(SR) = sqrt((1 - γ3·SR + (γ4 - 1)/4 · SR²) / (T - 1)),  γ4 = raw kurtosis
* Probabilistic Sharpe Ratio PSR(SR*) = Φ((SR - SR*) / SE)  (Bailey & López de Prado, 2012)
* Deflated Sharpe Ratio (Bailey & López de Prado, 2014): PSR against the Sharpe ratio one would
  expect from the best of ``n_trials`` unskilled strategies,
      SR0 = sqrt(V[SR]) · ((1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N e)))
  so comparing many strategies (or parameter settings) on the same data is penalised.
All per-period Sharpe quantities are computed on daily returns and annualised for display.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

TD = 252
EULER = 0.5772156649


def sharpe_stats(ex: pd.Series) -> dict[str, float]:
    x = ex.dropna().to_numpy()
    T = len(x)
    sd = x.std(ddof=1)
    if T < 20 or sd == 0:
        return {"sr_daily": np.nan, "se_daily": np.nan, "skew": np.nan, "kurt": np.nan, "T": T}
    sr = x.mean() / sd
    g3, g4 = float(skew(x)), float(kurtosis(x, fisher=False))
    se = float(np.sqrt(max(1e-12, (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1))))
    return {"sr_daily": float(sr), "se_daily": se, "skew": g3, "kurt": g4, "T": T}


def psr(sr: float, se: float, benchmark_sr: float = 0.0) -> float:
    return float(norm.cdf((sr - benchmark_sr) / se)) if se and np.isfinite(se) else np.nan


def expected_max_sr(trial_srs: list[float]) -> float:
    """SR0: expected maximum (daily) Sharpe of len(trial_srs) unskilled trials with their dispersion."""
    n = len(trial_srs)
    if n < 2:
        return 0.0
    v = float(np.var(trial_srs, ddof=1))
    return float(np.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e))))


def drawdown_stats(nav: pd.Series) -> dict:
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    rec = nav.loc[trough:][nav.loc[trough:] >= nav.loc[peak]]
    recovered = rec.index[0] if len(rec) else None
    # longest underwater spell (trading days)
    under = (dd < 0).astype(int)
    spells = under.groupby((under != under.shift()).cumsum()).sum()
    return {"max_drawdown": float(dd.min()), "dd_peak": str(peak.date()), "dd_trough": str(trough.date()),
            "dd_recovered": str(recovered.date()) if recovered is not None else None,
            "longest_underwater_days": int(spells.max()) if len(spells) else 0}


def full_metrics(daily: pd.Series, benchmark: pd.Series | None = None, rf_daily: pd.Series | float = 0.0,
                 turnover: pd.Series | None = None, costs: pd.DataFrame | None = None) -> dict:
    r = daily.dropna()
    rf = rf_daily.reindex(r.index).fillna(0.0) if isinstance(rf_daily, pd.Series) else pd.Series(rf_daily, index=r.index)
    ex = r - rf
    nav = (1 + r).cumprod()
    years = len(r) / TD
    ann = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TD))
    st = sharpe_stats(ex)
    sr_ann = st["sr_daily"] * np.sqrt(TD)
    down = ex[ex < 0]
    dd = drawdown_stats(nav)
    var95 = float(np.quantile(r, 0.05))
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    out = {
        "ann_return": ann, "ann_vol": vol, "sharpe": float(sr_ann),
        "sharpe_ci95": [float((st["sr_daily"] - 1.96 * st["se_daily"]) * np.sqrt(TD)),
                        float((st["sr_daily"] + 1.96 * st["se_daily"]) * np.sqrt(TD))],
        "psr_0": psr(st["sr_daily"], st["se_daily"], 0.0),
        "sortino": float(ex.mean() / np.sqrt((down ** 2).mean()) * np.sqrt(TD)) if len(down) else np.nan,
        "calmar": float(ann / abs(dd["max_drawdown"])) if dd["max_drawdown"] < 0 else np.nan,
        **dd,
        "var95_daily": var95, "cvar95_daily": float(r[r <= var95].mean()),
        "skew": st["skew"], "kurtosis": st["kurt"],
        "monthly_win_rate": float((monthly > 0).mean()), "best_month": float(monthly.max()), "worst_month": float(monthly.min()),
        "n_days": int(len(r)), "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "_sr_daily": st["sr_daily"], "_se_daily": st["se_daily"],
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0.0)
        act = r - b
        cov = np.cov(r, b, ddof=1)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
        bnav = (1 + b).cumprod()
        b_ann = float(bnav.iloc[-1] ** (1 / years) - 1)
        te = float(act.std(ddof=1) * np.sqrt(TD))
        out.update({
            "benchmark_ann_return": b_ann, "excess_ann_return": ann - b_ann,
            "tracking_error": te, "information_ratio": float(act.mean() * TD / te) if te > 0 else np.nan,
            "beta": beta, "alpha_ann": float((ex.mean() - beta * (b - rf).mean()) * TD),
            "corr_benchmark": float(np.corrcoef(r, b)[0, 1]),
            "up_capture": float(r[b > 0].mean() / b[b > 0].mean()) if (b > 0).any() else np.nan,
            "down_capture": float(r[b < 0].mean() / b[b < 0].mean()) if (b < 0).any() else np.nan,
        })
    if turnover is not None and len(turnover):
        per_year = len(turnover) / years if years else np.nan
        out["turnover_per_rebalance"] = float(turnover.iloc[1:].mean()) if len(turnover) > 1 else float(turnover.mean())
        out["annual_turnover"] = float(turnover.iloc[1:].mean() * per_year) if len(turnover) > 1 else np.nan
    if costs is not None and len(costs):
        tot = costs.sum()
        out["cost_drag_ann"] = float(costs.sum(axis=1).sum() / years)
        out["cost_breakdown_ann"] = {k: float(v / years) for k, v in tot.items()}
    return out


def deflate(metrics: dict[str, dict]) -> dict[str, dict]:
    """Add the Deflated Sharpe Ratio to each strategy's metrics, treating all of them as trials."""
    srs = [m["_sr_daily"] for m in metrics.values() if np.isfinite(m.get("_sr_daily", np.nan))]
    sr0 = expected_max_sr(srs)
    for m in metrics.values():
        m["dsr"] = psr(m["_sr_daily"], m["_se_daily"], sr0) if len(srs) > 1 else m.get("psr_0")
        m["dsr_trials"] = len(srs)
        m["sr0_ann"] = float(sr0 * np.sqrt(TD))
    return metrics
