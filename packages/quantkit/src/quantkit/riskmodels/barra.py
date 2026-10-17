"""Barra-style fundamental factor model (an industry + style factor model in the spirit of
MSCI Barra CNE5; the external benchmark of the thesis, methodology §9.1).

    exposures X_t = [industry dummies (SW level 1), standardised style characteristics]
    r_d = X_{m(d)} f_d + u_d          daily cross-sectional WLS (weights √ free-float cap)
    Σ   = X_T Σ_f X_T' + Σ_u

Style exposures come from the point-in-time monthly snapshots: winsorised at ±3 s.d. around the
cap-weighted mean (equal-weighted s.d.), missing → 0. Industries with fewer than ``min_industry``
members are pooled (a singleton industry would absorb its stock's whole return). A small ridge
(relative to the trace) keeps the regression stable. The fit exposes the same attributes as
latent_factor.FactorFit, so every covariance specification (SFC/DFC/DEC × EFM/AFM/SIC) applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..dynamics import garch11
from ..ipca import CharacteristicPanel
from .base import RiskModelInput, demeaned

# CNE5-like styles mapped to the point-in-time characteristics
STYLES = {"size": "Size", "beta_12m": "Beta", "mom_12_2": "Momentum", "ivol_3m": "Residual volatility",
          "bm": "Book-to-price", "ep": "Earnings yield", "revenue_yoy": "Growth", "leverage": "Leverage",
          "turn_6m": "Liquidity"}


def style_exposures(raw: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    Z = np.zeros(raw.shape)
    w = np.asarray(weights, float)
    for j, c in enumerate(raw.columns):
        x = raw[c].to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 3:
            continue
        mu = np.average(x[ok], weights=w[ok]) if w[ok].sum() > 0 else x[ok].mean()
        sd = x[ok].std()
        if not sd > 0:
            continue
        z = np.clip((x - mu) / sd, -3, 3)
        Z[:, j] = np.where(ok, z, 0.0)
    return Z


def industry_dummies(labels, min_industry: int = 3) -> tuple[np.ndarray, list[str]]:
    lab = pd.Series(np.asarray(labels).astype(str))
    lab = lab.where(~lab.isin(["NA", "nan", "None", ""]), "OTHER")
    counts = lab.value_counts()
    lab = lab.where(lab.map(counts) >= min_industry, "OTHER")
    names = sorted(lab.unique())
    return np.stack([(lab == g).to_numpy(float) for g in names], axis=1), names


class BarraFit:
    """Interface-compatible with FactorFit / IPCAFit (B, F, U, Sf, K, codes, x, garch_assets)."""

    def __init__(self, data: RiskModelInput, chars: CharacteristicPanel, industries, cap_weights,
                 styles: tuple[str, ...] = tuple(STYLES), min_industry: int = 3, ridge: float = 1e-3):
        self.codes = data.codes
        codes = list(self.codes)
        R = data.returns.to_numpy(dtype=float)
        obs = ~np.isnan(R)
        r = np.where(obs, R, 0.0)
        w = np.sqrt(np.clip(np.nan_to_num(np.asarray(cap_weights, float)), 0, None))
        w = w / w.mean() if w.sum() > 0 else np.ones(len(codes))
        D, self.industry_names = industry_dummies(industries, min_industry)
        styles = [s for s in styles if s in chars.cols]
        self.factor_names = self.industry_names + [STYLES[s] for s in styles]

        def exposures(snap):
            return np.column_stack([D, style_exposures(chars.raw(snap, codes)[styles], w)])

        snaps = [chars.snapshot_for(d, strict=True) for d in data.returns.index]
        uniq = list(dict.fromkeys(snaps))
        Xm = np.stack([exposures(s) for s in uniq])
        idx = np.array([uniq.index(s) for s in snaps])
        Xd = Xm[idx]                                             # T x N x K
        K = Xd.shape[2]
        W = obs * w[None, :]
        A = np.einsum("tnk,tn,tnl->tkl", Xd, W, Xd)
        A = A + (ridge * np.trace(A, axis1=1, axis2=2) / K + 1e-12)[:, None, None] * np.eye(K)
        F = np.linalg.solve(A, np.einsum("tnk,tn->tk", Xd, W * r)[:, :, None])[:, :, 0]
        U = np.where(obs, r - np.einsum("tnk,tk->tn", Xd, F), 0.0)
        self.U = U - U.mean(axis=0)
        self.F = F - F.mean(axis=0)
        self.Sf = np.atleast_2d(np.cov(self.F, rowvar=False))
        self.K = K
        self.B = exposures(chars.snapshot_for(data.asof, strict=False))
        self.x = demeaned(data.returns)
        self._garch_x = None
        self._dbc_B = None

    def garch_assets(self, horizon: int):
        if self._garch_x is None:
            self._garch_x = [garch11(self.x[:, i], horizon) for i in range(self.x.shape[1])]
        return self._garch_x
