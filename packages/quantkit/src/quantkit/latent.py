"""Latent-factor extraction used by the benchmark paper (Conlon, Cotter & Kynigakis, 2025).

Factors are extracted from cross-sectionally rank-transformed returns (as in the paper and
Gu, Kelly & Xiu, 2020); loadings are then OLS betas of the *raw* returns on those factors.

Methods
-------
* ``pca``  — principal components (unsupervised, linear)
* ``pls``  — partial least squares, target = equal-weighted market return of raw returns
* ``spca`` — sparse PCA (elastic-net penalised loadings, scikit-learn ``SparsePCA``)
* ``spls`` — sparse PLS: PLS weight vectors soft-thresholded (Chun & Keleş, 2010)
* ``aen``  — single-hidden-layer autoencoder with an L1 activity penalty (sparse AE)
* ``dae``  — single-hidden-layer denoising autoencoder (Gaussian input corruption)

Implementation choices not pinned down in the paper (PLS target, SPLS algorithm, AE details)
are documented here and in the M0 notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

METHODS = ("pca", "pls", "spca", "spls", "aen", "dae")


def rank_transform(x: np.ndarray) -> np.ndarray:
    """Per day (row): rank / (n + 1) - 0.5 across assets; NaN -> 0."""
    out = np.zeros_like(x, dtype=float)
    for t in range(x.shape[0]):
        row = x[t]
        ok = ~np.isnan(row)
        n = ok.sum()
        if n == 0:
            continue
        r = np.argsort(np.argsort(row[ok]))
        out[t, ok] = (r + 1) / (n + 1) - 0.5
    return out


@dataclass
class FactorExtractor:
    method: str
    n_factors: int = 5
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def fit_transform(self, z: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
        """``z``: T x N rank-transformed returns; ``target``: T supervised target for PLS/SPLS
        (the equal-weighted market return of the *raw* returns — rank-transformed rows have a
        constant mean, so they cannot provide the target). Returns T x K factor realisations."""
        m = self.method
        K = min(self.n_factors, z.shape[1] - 1, z.shape[0] - 2)
        zc = z - z.mean(axis=0)
        if m == "pca":
            _, _, vt = np.linalg.svd(zc, full_matrices=False)
            return zc @ vt[:K].T
        y = target if target is not None else zc.mean(axis=1)
        if m == "pls":
            return _pls(zc, y, K, eta=0.0)
        if m == "spls":
            return _pls(zc, y, K, eta=self.params.get("eta", 0.5))
        if m == "spca":
            from sklearn.decomposition import SparsePCA

            sp = SparsePCA(n_components=K, alpha=self.params.get("alpha", 1.0), ridge_alpha=0.01,
                           max_iter=200, tol=1e-6, random_state=self.seed, method="cd")
            f = sp.fit_transform(zc)
            keep = np.abs(f).sum(axis=0) > 0          # drop components shrunk to zero
            return f[:, keep] if keep.any() else zc @ np.linalg.svd(zc, full_matrices=False)[2][:1].T
        if m in ("aen", "dae"):
            return _autoencoder(zc, K, sparse=m == "aen", l1=self.params.get("l1", 1e-3),
                                noise=self.params.get("noise", 0.1), seed=self.seed)
        raise ValueError(f"unknown method {m}")


def _soft(x: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def _pls(x: np.ndarray, y: np.ndarray, K: int, eta: float) -> np.ndarray:
    """PLS1 (NIPALS) with optional soft-thresholding of each weight vector (sparse PLS).

    eta in [0, 1): threshold as a fraction of max|X'y| (Chun & Keleş 2010 style)."""
    X = x.copy()
    yv = y - y.mean()
    F = []
    for _ in range(K):
        c = X.T @ yv
        if eta > 0:
            c = _soft(c, eta * np.abs(c).max())
        nrm = np.linalg.norm(c)
        if nrm < 1e-12:
            break
        w = c / nrm
        t = X @ w
        tt = t @ t
        if tt < 1e-12:
            break
        F.append(t)
        p = X.T @ t / tt
        X = X - np.outer(t, p)          # deflate X
        yv = yv - t * (t @ yv) / tt     # deflate y
    return np.column_stack(F) if F else x @ np.ones((x.shape[1], 1)) / x.shape[1]


def _autoencoder(x: np.ndarray, K: int, sparse: bool, l1: float, noise: float, seed: int,
                 epochs: int = 400, lr: float = 1e-2) -> np.ndarray:
    """Shallow autoencoder N -> K (tanh) -> N (linear); returns the K hidden activations."""
    import torch

    torch.manual_seed(seed)
    X = torch.tensor(x, dtype=torch.float32)
    N = X.shape[1]
    enc = torch.nn.Linear(N, K)
    dec = torch.nn.Linear(K, N)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        inp = X + noise * X.std() * torch.randn_like(X) if not sparse else X
        h = torch.tanh(enc(inp))
        loss = ((dec(h) - X) ** 2).mean()
        if sparse:
            loss = loss + l1 * h.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.tanh(enc(X)).numpy().astype(float)
