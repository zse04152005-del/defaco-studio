"""Instrumented PCA (Kelly, Pruitt & Su, 2019) with mixed-frequency inputs.

    r_{i,d} = z_{i,m(d)}' Γ f_d + ε_{i,d}

Daily returns r_d load on K latent factors through characteristics z observed at the
previous month end m(d) (point in time), so loadings B_t = Z_t Γ move with the stocks'
characteristics instead of being re-estimated by rolling OLS. Γ (L x K) and the factor
returns f_d are estimated by alternating least squares; the managed portfolios
x_d = Z_d' r_d and W_d = Z_d' Z_d (over stocks observed that day) are computed once, so
each ALS sweep costs O(T K^2 L^2) regardless of N.

Identification (KPS): Γ'Γ = I_K and the factors are orthogonal, ordered by variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- characteristics
class CharacteristicPanel:
    """Month-end characteristic snapshots; serves rank-normalised Z matrices for any codes.

    Characteristics are re-ranked within the requested cross-section, rank/(n+1) - 0.5
    (Gu, Kelly & Xiu, 2020; KPS, 2019), missing -> 0, and a constant column is prepended.
    ``smooth`` > 1 averages the ranked matrices of the last ``smooth`` snapshots (slower,
    less noisy loadings; still point in time).
    """

    def __init__(self, raw: pd.DataFrame, cols: list[str], smooth: int = 1):
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        self.cols = list(cols)
        self.smooth = max(1, int(smooth))
        self.dates = pd.DatetimeIndex(sorted(raw["date"].unique()))
        self._snap = {d: g.set_index("code")[self.cols] for d, g in raw.groupby("date")}

    @property
    def names(self) -> list[str]:
        return ["const", *self.cols]

    def snapshot_for(self, day: pd.Timestamp, strict: bool) -> pd.Timestamp | None:
        """Latest snapshot dated before ``day`` (strict) or on/before it."""
        i = self.dates.searchsorted(pd.Timestamp(day), side="left" if strict else "right") - 1
        return self.dates[i] if i >= 0 else None

    def _ranked(self, snap: pd.Timestamp, codes: list[str]) -> np.ndarray:
        x = self._snap[snap].reindex(codes)
        r = x.rank(method="average")
        n = x.notna().sum()
        return (r / (n + 1) - 0.5).fillna(0.0).to_numpy(dtype=float)

    def raw(self, snap: pd.Timestamp | None, codes: list[str]) -> pd.DataFrame:
        """Raw (un-ranked) characteristic values of ``codes`` in a snapshot (NaN if missing)."""
        if snap is None:
            return pd.DataFrame(np.nan, index=list(codes), columns=self.cols)
        return self._snap[snap].reindex(list(codes))

    def matrix(self, snap: pd.Timestamp | None, codes: list[str]) -> np.ndarray:
        N, L = len(codes), len(self.cols)
        if snap is None:
            z = np.zeros((N, L))
        else:
            i = self.dates.get_loc(snap)
            snaps = self.dates[max(0, i - self.smooth + 1): i + 1]
            z = np.mean([self._ranked(d, codes) for d in snaps], axis=0)
        return np.column_stack([np.ones(N), z])

    def window(self, days: pd.DatetimeIndex, codes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(M x N x L stack of distinct snapshots, length-T index of each day's snapshot)."""
        snaps = [self.snapshot_for(d, strict=True) for d in days]
        uniq = list(dict.fromkeys(snaps))
        Zm = np.stack([self.matrix(s, codes) for s in uniq])
        pos = {s: i for i, s in enumerate(uniq)}
        return Zm, np.array([pos[s] for s in snaps])


# ----------------------------------------------------------------------------- estimator
@dataclass
class IPCAResult:
    gamma: np.ndarray          # L x K
    factors: np.ndarray        # T x K
    n_iter: int
    converged: bool
    r2: float                  # total panel R^2 of the fitted returns


def _normalize(G: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (Γ, F) so that Γ'Γ = I and F has orthogonal columns with decreasing variance;
    the fitted values Z Γ f are unchanged. Signs: each factor has a positive mean."""
    Q, R = np.linalg.qr(G)
    F = F @ R.T
    ev, V = np.linalg.eigh(np.atleast_2d(np.cov(F, rowvar=False)))
    V = V[:, np.argsort(ev)[::-1]]
    G, F = Q @ V, F @ V
    s = np.where(F.mean(axis=0) < 0, -1.0, 1.0)
    return G * s, F * s


def ipca_fit(R: np.ndarray, Zm: np.ndarray, day_snap: np.ndarray, K: int, max_iter: int = 200,
             tol: float = 1e-6, ridge: float = 1e-8) -> IPCAResult:
    """ALS for restricted IPCA (no intercept Γ_α).

    R: T x N returns (NaN = not observed); Zm: M x N x L snapshots; day_snap: snapshot of each day.
    """
    T, N = R.shape
    L = Zm.shape[2]
    K = min(K, L)
    obs = ~np.isnan(R)
    r = np.where(obs, R, 0.0)
    Zd = Zm[day_snap]                                            # T x N x L
    Zo = Zd * obs[:, :, None]
    W = np.einsum("tna,tnb->tab", Zo, Zd)                        # Z_d' Z_d over observed stocks
    X = np.einsum("tna,tn->ta", Zo, r)                           # managed portfolios Z_d' r_d
    eyeK = ridge * np.eye(K)

    # initialisation: leading singular vectors of the (scaled) managed portfolios
    n_obs = np.maximum(obs.sum(axis=1), 1)[:, None]
    U, _, _ = np.linalg.svd((X / n_obs).T, full_matrices=False)
    G = U[:, :K]
    F = np.zeros((T, K))
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        # factor step: f_d = (Γ' W_d Γ)^{-1} Γ' x_d
        A = np.einsum("ak,tab,bl->tkl", G, W, G) + eyeK
        F = np.linalg.solve(A, (X @ G)[:, :, None])[:, :, 0]
        # Γ step: Σ_d (f f' ⊗ W_d) vec(Γ) = Σ_d f_d ⊗ x_d   (vec stacks Γ's columns)
        M = np.einsum("tk,tl,tab->kalb", F, F, W).reshape(K * L, K * L)
        b = np.einsum("tk,ta->ka", F, X).reshape(K * L)
        G_new = np.linalg.solve(M + ridge * np.trace(M) / (K * L) * np.eye(K * L), b).reshape(K, L).T
        G_new, F = _normalize(G_new, F)
        delta = np.abs(G_new - G).max()
        G = G_new
        if delta < tol:
            converged = True
            break
    # final factors consistent with the normalised Γ
    A = np.einsum("ak,tab,bl->tkl", G, W, G) + eyeK
    F = np.linalg.solve(A, (X @ G)[:, :, None])[:, :, 0]
    fitted = np.einsum("tna,ak,tk->tn", Zd, G, F)
    ss_res = float(((r - fitted) ** 2 * obs).sum())
    ss_tot = float((r ** 2 * obs).sum())
    return IPCAResult(gamma=G, factors=F, n_iter=it, converged=converged, r2=1 - ss_res / ss_tot if ss_tot else 0.0)


# ----------------------------------------------------------------------------- M1+: characteristics + free loadings
@dataclass
class IPCAPlusResult:
    gamma: np.ndarray          # L x K   characteristic part of the loadings
    free: np.ndarray           # N x K   stock-specific part (ridge-shrunk towards 0)
    factors: np.ndarray        # T x K
    lam: float                 # ridge penalty actually used
    n_iter: int
    converged: bool
    r2: float
    free_share: float          # share of the loading variance explained by the free part


def ipca_plus_fit(R: np.ndarray, Zm: np.ndarray, day_snap: np.ndarray, K: int, c: float = 1.0,
                  max_iter: int = 150, tol: float = 1e-6, ridge: float = 1e-8, kappa: float = 0.01) -> IPCAPlusResult:
    """r_{i,d} = (z_{i,m(d)}'Γ + b̃_i') f_d + e   with   min Σ e² + λ Σ_i ||b̃_i||²,
    λ = c · tr(F'F) / K (scale-free). λ → ∞ gives IPCA; λ → 0 free (PCA-like) loadings.
    Alternating least squares over f, Γ and b̃ (Fan, Liao & Wang 2016 projected-PCA style split
    of the loadings into a characteristic-driven and an idiosyncratic component).

    Identification: the free loadings are kept orthogonal to the characteristic space
    (B̃ ⟂ span(Z̄), Z̄ = window-average characteristics) — the projected-PCA condition E[γ | X] = 0.
    Without it B̃ can cancel part of ZΓ inside the window, the factor returns grow without bound,
    and the forecast covariance (built with the new Z_T) explodes.

    Factor returns are estimated with a relative ridge κ·tr(B_d'B_d)/K: a characteristic direction
    that is (nearly) constant inside the window — e.g. missing and filled with 0 — would otherwise
    get an unbounded factor return that blows up as soon as the direction is populated at T."""
    T, N = R.shape
    L = Zm.shape[2]
    K = min(K, L)
    obs = ~np.isnan(R)
    o = obs.astype(float)
    r = np.where(obs, R, 0.0)
    Zd = Zm[day_snap]
    Zo = Zd * o[:, :, None]
    W = np.einsum("tna,tnb->tab", Zo, Zd)
    base = ipca_fit(R, Zm, day_snap, K, max_iter=100, tol=1e-5)
    G, F = base.gamma, base.factors
    B = np.zeros((N, K))
    lam = 0.0
    Zbar = np.nanmean(np.where(o[:, :, None] > 0, Zd, np.nan), axis=0)
    Zbar = np.nan_to_num(Zbar)
    P_perp = np.eye(N) - Zbar @ np.linalg.pinv(Zbar)          # projector onto span(Z̄)^⊥
    it, converged, prev = 0, False, None
    for it in range(1, max_iter + 1):
        # f-step: cross-sectional regression on B_d = Z_d Γ + B̃
        Bd = np.einsum("tna,ak->tnk", Zd, G) + B[None]
        A = np.einsum("tnk,tn,tnl->tkl", Bd, o, Bd)
        A = A + (kappa * np.trace(A, axis1=1, axis2=2) / K + ridge)[:, None, None] * np.eye(K)
        F = np.linalg.solve(A, np.einsum("tnk,tn->tk", Bd, r)[:, :, None])[:, :, 0]
        # Γ-step on returns net of the free part
        y = (r - np.einsum("nk,tk->tn", B, F)) * o
        X = np.einsum("tna,tn->ta", Zd, y)
        M = np.einsum("tk,tl,tab->kalb", F, F, W).reshape(K * L, K * L)
        rhs = np.einsum("tk,ta->ka", F, X).reshape(K * L)
        G = np.linalg.solve(M + ridge * np.trace(M) / (K * L) * np.eye(K * L), rhs).reshape(K, L).T
        # B̃-step: per-stock ridge regression of the remaining residual on the factors
        lam = c * float(np.trace(F.T @ F)) / K
        e = (r - np.einsum("tna,ak,tk->tn", Zd, G, F)) * o
        FF = np.einsum("tk,tl,tn->nkl", F, F, o) + lam * np.eye(K)
        B = np.linalg.solve(FF, np.einsum("tk,tn->nk", F, e)[:, :, None])[:, :, 0]
        B = P_perp @ B
        # identification: Γ'Γ = I, orthogonal factors ordered by variance (B̃ rotated alongside)
        Q, Rr = np.linalg.qr(G)
        F = F @ Rr.T
        B = B @ np.linalg.inv(Rr)
        ev, V = np.linalg.eigh(np.atleast_2d(np.cov(F, rowvar=False)))
        V = V[:, np.argsort(ev)[::-1]]
        G, F, B = Q @ V, F @ V, B @ V
        s = np.where(F.mean(axis=0) < 0, -1.0, 1.0)
        G, F, B = G * s, F * s, B * s
        fitted = np.einsum("tnk,tk->tn", np.einsum("tna,ak->tnk", Zd, G) + B[None], F)
        if prev is not None and np.abs(fitted - prev).max() < tol * max(1e-12, np.abs(r).max()):
            converged = True
            break
        prev = fitted
    # final factors consistent with the final (normalised) Γ and B̃ — the loop's F belongs to the
    # previous Γ, which matters when ALS has not fully converged
    Bd = np.einsum("tna,ak->tnk", Zd, G) + B[None]
    A = np.einsum("tnk,tn,tnl->tkl", Bd, o, Bd)
    A = A + (kappa * np.trace(A, axis1=1, axis2=2) / K + ridge)[:, None, None] * np.eye(K)
    F = np.linalg.solve(A, np.einsum("tnk,tn->tk", Bd, r)[:, :, None])[:, :, 0]
    fitted = np.einsum("tnk,tk->tn", Bd, F)
    ss_res = float(((r - fitted) ** 2 * o).sum())
    ss_tot = float((r ** 2 * o).sum())
    zpart = np.einsum("tna,ak->nk", Zd[-1:], G)          # characteristic loadings on the last day
    share = float((B ** 2).sum() / ((B ** 2).sum() + (zpart ** 2).sum() + 1e-30))
    return IPCAPlusResult(gamma=G, free=B, factors=F, lam=lam, n_iter=it, converged=converged,
                          r2=1 - ss_res / ss_tot if ss_tot else 0.0, free_share=share)
