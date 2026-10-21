"""Whole-lot portfolios with a flexible budget (A-share board lots, minimum commissions, limit/suspension freezes).

The capital is a guide, not a hard limit: the amount spent (holdings plus fees) may lie in [lo, hi] x budget
(opening: 90%-110% of the capital; rebalance: at most the account value). Within that range we look for the
whole-lot holdings that score best, where the score is

    risk(w) + lam * |spend / budget - 1| + turnover_cost * fees / budget + cap_penalty * (excess over the caps)

- risk(w): the predicted annual volatility sqrt(w' S w) of the holdings' weights (``track=False``, the
  minimum-variance strategies), or the tracking error sqrt((w - w*)' S (w - w*)) to target weights w*
  (``track=True``; the equal-weight strategy uses S = I);
- lam = 0.02: a 5% gap between spending and the budget costs 0.1 point of volatility, so the budget moves only
  when that buys a visibly better portfolio;
- fees count only at a rebalance (the opening must buy anyway): a lot is traded only when the risk it removes is
  worth more than its fees;
- caps (single stock, industry) are soft with a steep penalty, so holdings that drifted above a cap are brought
  back lot by lot instead of blocking the search.

Search: best-improvement local search over three moves - add one lot, remove one lot, move one lot from stock i
to stock j - evaluated in closed form for all pairs at once. Opening: two starts (the lot rounding of the
continuous optimum, and a greedy build adding the lot that keeps risk lowest), the better result is kept.
Rebalance: one start, the current holdings. Stocks at the upper limit or suspended are not bought; stocks at the
lower limit or suspended are not sold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

COMMISSION, MIN_COMMISSION, TRANSFER, STAMP = 2.5e-4, 5.0, 1e-5, 5e-4
CAP_PENALTY = 10.0         # volatility per unit of weight above a cap (1 point of excess weight = 10 points of risk)


def lot_of(code: str) -> tuple[int, int]:
    """(minimum shares, step) of one order: STAR Market 200 then 1, other boards (incl. Beijing) 100."""
    return (200, 1) if code.startswith("688") else (100, 100)


def fee(amount: float, side: str) -> float:
    if amount <= 0:
        return 0.0
    f = max(MIN_COMMISSION, COMMISSION * amount) + TRANSFER * amount
    return f + (STAMP * amount if side == "sell" else 0.0)


def fee_vec(amount: np.ndarray, sell: bool = False) -> np.ndarray:
    a = np.abs(np.asarray(amount, float))
    f = np.maximum(MIN_COMMISSION, COMMISSION * a) + TRANSFER * a + (STAMP * a if sell else 0.0)
    return np.where(a > 0, f, 0.0)


@dataclass
class LotResult:
    shares: pd.Series        # code -> shares
    risk: float              # objective risk of the lot portfolio (volatility or tracking error, annual)
    risk_target: float       # of the continuous target (0 for tracking)
    te: float                # tracking error lot vs target under S (annual)
    invested: float          # market value of the holdings
    fees: float              # fees of the orders that reach these holdings
    spend_ratio: float       # (invested + fees) / budget
    moves: int


def lot_portfolio(codes: list[str], S: np.ndarray, prices: pd.Series, budget: float, w_target: pd.Series,
                  wmax: float, industry: pd.Series, ind_cap: float = 0.35, lo: float = 0.90, hi: float = 1.10,
                  lam: float = 0.02, track: bool = False, current: pd.Series | None = None,
                  no_buy: set | None = None, no_sell: set | None = None, turnover_cost: float = 1.0,
                  max_moves: int = 3000) -> LotResult:
    n = len(codes)
    p = prices.reindex(codes).to_numpy(float)
    mn = np.array([lot_of(c)[0] for c in codes]); st = np.array([lot_of(c)[1] for c in codes])
    ind = pd.Series(industry).reindex(codes).fillna("NA").to_numpy()
    _, gidx = np.unique(ind, return_inverse=True)
    G = np.zeros((gidx.max() + 1, n)); G[gidx, np.arange(n)] = 1.0
    wt = w_target.reindex(codes).fillna(0.0).to_numpy(float)
    c0 = wt if track else np.zeros(n)
    Sc = S @ c0; cSc = float(c0 @ Sc)
    rebal = current is not None
    cur = current.reindex(codes).fillna(0).to_numpy(int) if rebal else np.zeros(n, int)
    nb = np.array([c in (no_buy or set()) for c in codes]); ns = np.array([c in (no_sell or set()) for c in codes])

    def trade_fees(s: np.ndarray) -> float:
        d = (s - cur) * p
        return float(fee_vec(np.where(d > 0, d, 0)).sum() + fee_vec(np.where(d < 0, d, 0), sell=True).sum())

    def score(s: np.ndarray) -> float:
        v = s * p; I = v.sum()
        if I <= 0:
            return math.inf
        f = trade_fees(s)
        spend = I + f
        if spend > hi * budget + 1e-6 or I < lo * budget - 1e-6:
            return math.inf
        w = v / I
        risk = math.sqrt(max((w - c0) @ S @ (w - c0), 0.0))
        excess = np.maximum(w - wmax, 0).sum() + np.maximum(G @ w - ind_cap, 0).sum()
        return risk + lam * abs(spend / budget - 1) + (turnover_cost * f / budget if rebal else 0.0) + CAP_PENALTY * excess

    def steps(s: np.ndarray):
        up = np.where(s == 0, mn, st)                    # shares added by one lot
        dn = np.where(s - st < mn, s, st)                # shares removed by one lot (all, below the minimum)
        return up, dn

    def best_move(s: np.ndarray):
        """Screen every add / remove / transfer by its risk in closed form, then score the best few exactly."""
        v = s * p; I = v.sum(); Sv = S @ v; q = v @ Sv; b = Sc @ v
        up, dn = steps(s)
        a_up = np.where(nb, 0.0, up * p); a_dn = np.where((s > 0) & ~ns, dn * p, 0.0)
        ai = np.append(a_dn, 0.0)[:, None]; aj = np.append(a_up, 0.0)[None, :]
        Svi = np.append(Sv, 0.0)[:, None]; Svj = np.append(Sv, 0.0)[None, :]
        Sci = np.append(Sc, 0.0)[:, None]; Scj = np.append(Sc, 0.0)[None, :]
        dS = np.append(np.diag(S), 0.0)
        Sx = np.zeros((n + 1, n + 1)); Sx[:n, :n] = S
        q2 = q - 2 * ai * Svi + 2 * aj * Svj + ai ** 2 * dS[:, None] + aj ** 2 * dS[None, :] - 2 * ai * aj * Sx
        b2 = b - ai * Sci + aj * Scj
        I2 = I - ai + aj
        valid = np.ones((n + 1, n + 1), bool)
        valid[:n, :] &= (a_dn > 0)[:, None]              # remove only what is held and may be sold
        valid[:, :n] &= (a_up > 0)[None, :]              # add only what may be bought
        np.fill_diagonal(valid, False); valid[n, n] = False
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = q2 / I2 ** 2 - 2 * b2 / I2 + cSc
        risk = np.where(valid & (I2 > 0), np.sqrt(np.maximum(r2, 0)), np.inf)
        base = score(s)
        best, arg = base, None
        for k in np.argsort(risk, axis=None)[:300]:
            if not np.isfinite(risk.flat[k]):
                break
            i, j = divmod(int(k), n + 1)
            t = s.copy()
            if i < n:
                t[i] -= dn[i]
            if j < n:
                t[j] += up[j]
            o = score(t)
            if o < best - 1e-9:
                best, arg = o, t
        return arg

    def search(s: np.ndarray) -> tuple[np.ndarray, int]:
        k = 0
        while k < max_moves:
            t = best_move(s)
            if t is None:
                break
            s, k = t, k + 1
        return s, k

    def greedy() -> np.ndarray:
        """From empty: add the lot that keeps risk lowest, caps measured against the budget, until the budget is met."""
        t = np.zeros(n, int)
        while t @ p < budget:
            up, _ = steps(t)
            tv = (t * p)[None, :] + np.diag(up * p)
            I2 = tv.sum(1)
            ok = ~nb & (tv.max(1) <= wmax * budget * 0.998) & ((G @ tv.T).max(0) <= ind_cap * budget * 0.998)
            ok &= I2 + np.array([fee(x, "buy") for x in up * p]) + trade_fees(t) <= hi * budget
            if not ok.any():
                break
            W = tv / I2[:, None] - c0[None, :]
            r = np.einsum("ij,jk,ik->i", W, S, W)
            k = int(np.argmin(np.where(ok, r, np.inf)))
            t[k] += up[k]
        return t

    if rebal:
        s, moves = search(cur.copy())
    else:
        x = np.floor(wt * budget / np.where(p > 0, p, np.inf)).astype(int)
        start = np.where(nb | (x < mn), 0, mn + (x - mn) // st * st)
        runs = [search(start), search(greedy())]
        s, moves = min(runs, key=lambda r: score(r[0]))

    v = s * p; I = v.sum(); w = v / I if I > 0 else v
    f = trade_fees(s)
    return LotResult(shares=pd.Series(s, index=codes),
                     risk=float(math.sqrt(max((w - c0) @ S @ (w - c0), 0))),
                     risk_target=float(math.sqrt(max((wt - c0) @ S @ (wt - c0), 0))),
                     te=float(math.sqrt(max((w - wt) @ S @ (w - wt), 0))),
                     invested=float(I), fees=f, spend_ratio=float((I + f) / budget), moves=moves)
