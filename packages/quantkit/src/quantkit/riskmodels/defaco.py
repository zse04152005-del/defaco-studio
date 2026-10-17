"""DeFaCo (the thesis model, M4) as a demo risk model: numpy inference of the trained parameters.

The parameters Θ = {Γ, α, β} were learned end to end in the thesis (``03_代码/src/defaco/dfl``,
torch) and are exported per test year by ``scripts/export_defaco.py`` to
``data/models/defaco_m4.json``. Given Θ, the covariance is a fixed computation, re-implemented
here step by step after ``DeFaCo.forward`` (the thesis' unit test checks both agree to 1e-10):

    loadings          B_d = Z_{τ(d)} Γ   (characteristics ranked within the cross-section)
    factor returns    f_d = (B_d' O_d B_d + λ_d I)^{-1} B_d' O_d r_d,   λ_d = ϱ tr(B_d' O_d B_d) / K
    residuals         u_d = O_d (r_d - B_d f_d)
    factor cov.       Σ_f = sample covariance of f over the 250-day window (static, SFC)
    residual cov.     SIC: EWMA variances (half-life 63) shrunk to the median by α, within-industry
                      equicorrelation scaled by β
    forecast          Σ = B_t Σ_f B_t' + Σ_u,   averaged over the five seeds

The model for a rebalancing date in year Y is the one trained for test year Y (2016-2025); later
dates use the last year and are flagged ``out_of_sample_period``. Requires the characteristic
panel in ``RiskModelInput.extras["chars"]`` (a ``quantkit.ipca.CharacteristicPanel``).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CovarianceRepr, RiskModel, RiskModelInput, register

EPS_RHO = 1e-4
FAST = ("rev_1m", "vol_1m", "maxret_1m", "turn_1m", "abturn")


def default_model_file() -> Path:
    from quantkit.config import find_repo_root
    return find_repo_root() / "data" / "models" / "defaco_m4.json"


@lru_cache(maxsize=4)
def load_models(path: str | None = None) -> dict:
    p = Path(path) if path else default_model_file()
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run scripts/export_defaco.py (needs the thesis' trained models)")
    years = json.loads(p.read_text(encoding="utf-8"))["years"]
    for y in years.values():
        for s in y["seeds"]:
            s["gamma"] = np.asarray(s["gamma"], dtype=float)
    return {int(k): v for k, v in years.items()}


def industry_ids(labels) -> np.ndarray:
    lab = pd.Series(np.asarray(labels).astype(str))
    bad = lab.isin(["NA", "nan", "None", ""])
    return np.where(bad, -1, pd.factorize(lab.where(~bad))[0])


def onehot(groups: np.ndarray) -> np.ndarray:
    """N x G membership of industries with at least two stocks (unknown / singleton industries dropped)."""
    g = np.asarray(groups)
    ids = [k for k in np.unique(g[g >= 0]) if (g == k).sum() >= 2]
    return np.stack([(g == k).astype(float) for k in ids], axis=1) if ids else np.zeros((len(g), 0))


def ewma_weights(T: int, half_life: float) -> np.ndarray:
    w = 0.5 ** (np.arange(T - 1, -1, -1, dtype=float) / half_life)
    return w / w.sum()


def defaco_covariance(theta: dict, Zm: np.ndarray, idx: np.ndarray, ZT: np.ndarray, r: np.ndarray, o: np.ndarray,
                      G: np.ndarray) -> dict[str, np.ndarray]:
    """One seed's forecast. Zm: M x N x L snapshots, idx: snapshot of each day (T), ZT: N x L at the
    forecast date, r: T x N returns (0 where unobserved), o: T x N observed mask, G: N x groups one-hot.
    Returns the factor part, the residual covariance and their sum (all N x N)."""
    gamma = theta["gamma"]
    K = gamma.shape[1]
    Bd = (Zm @ gamma)[idx]                                             # T x N x K
    A = np.einsum("tnk,tn,tnl->tkl", Bd, o, Bd)
    lam = theta["ridge"] * np.trace(A, axis1=1, axis2=2) / K + 1e-10
    A = A + lam[:, None, None] * np.eye(K)
    F = np.linalg.solve(A, np.einsum("tnk,tn->tk", Bd, o * r)[..., None])[..., 0]
    U_raw = (r - np.einsum("tnk,tk->tn", Bd, F)) * o
    T, N = U_raw.shape
    Fc = F - F.mean(0)
    U = U_raw - U_raw.mean(0)
    Sf = Fc.T @ Fc / (T - 1)
    B = ZT @ gamma
    # structured industry-block residual (SIC)
    w = ewma_weights(T, theta["half_life"])
    s2_hat = np.maximum((w[:, None] * U ** 2).sum(0), 1e-12)
    a, b = theta["alpha"], theta["beta"]
    s2 = (1 - a) * s2_hat + a * np.quantile(s2_hat, 0.5)
    Rm = np.eye(N)
    if G.shape[1]:
        e = U / np.sqrt(s2_hat)
        agg = e @ G
        var_agg = (w[:, None] * (agg - (w[:, None] * agg).sum(0)) ** 2).sum(0)
        var_i = (w[:, None] * (e - (w[:, None] * e).sum(0)) ** 2).sum(0)
        n = G.sum(0)
        rho_hat = (var_agg - var_i @ G) / (n * (n - 1))
        rho = np.minimum(np.maximum(b * rho_hat, -1.0 / (n - 1) + EPS_RHO), 1 - EPS_RHO)
        Rm = ((G * rho) @ G.T) * (1 - np.eye(N)) + np.eye(N)
    sd = np.sqrt(s2)
    Su = sd[:, None] * Rm * sd[None, :]
    fac = B @ Sf @ B.T
    return {"factor": fac, "resid": Su, "Sigma": fac + Su, "B": B}


@register("defaco")
class DeFaCoModel(RiskModel):
    """Trained DeFaCo (M4): ensemble of five seeds for the test year of ``asof``."""

    name = "defaco"

    def __init__(self, model_file: str | None = None):
        self.model_file = model_file
        self._cov: CovarianceRepr | None = None
        self._explain: dict = {}

    def fit(self, data: RiskModelInput) -> "DeFaCoModel":
        chars = data.extras.get("chars")
        if chars is None:
            raise ValueError("DeFaCo needs the characteristic panel in RiskModelInput.extras['chars']")
        models = load_models(self.model_file)
        year = int(pd.Timestamp(data.asof).year)
        use = min(max(year, min(models)), max(models))
        m = models[use]
        codes = data.codes
        days = data.returns.index
        Zm, idx = chars.window(days, codes)
        ZT = chars.matrix(chars.snapshot_for(data.asof, strict=False), codes)
        R = data.returns.to_numpy(dtype=float)
        o = (~np.isnan(R)).astype(float)
        r = np.nan_to_num(R)
        groups = industry_ids(data.industries.reindex(codes).to_numpy() if data.industries is not None else ["NA"] * len(codes))
        G = onehot(groups)
        outs = [defaco_covariance(th, Zm, idx, ZT, r, o, G) for th in m["seeds"]]
        fac = np.mean([x["factor"] for x in outs], axis=0)
        res = np.mean([x["resid"] for x in outs], axis=0)
        S = fac + res
        S = (S + S.T) / 2
        diag = np.diag(np.diag(res))
        self._cov = CovarianceRepr(list(codes), dense_cov=S,
                                   parts={"factor": fac, "industry": res - diag, "specific": diag})
        names = m["char_names"]
        self._explain = {
            "model_year": use, "out_of_sample_period": year > max(models), "K": m["K"], "kappa_bp": m["kappa_bp"],
            "alpha": float(np.mean([s["alpha"] for s in m["seeds"]])), "beta": float(np.mean([s["beta"] for s in m["seeds"]])),
            "seeds": len(m["seeds"]),
            "gamma_share": {"chars": names[1:], "m4": m["share_m4"][1:], "m3": (m["share_m3"] or [None] * len(names))[1:],
                            "fast": [c for c in names if c in FAST]},
        }
        return self

    def covariance(self) -> CovarianceRepr:
        assert self._cov is not None, "call fit() first"
        return self._cov

    def explain(self) -> dict:
        return dict(self._explain)
