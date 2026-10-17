"""Realistic A-share portfolio simulator (used by the demo's live backtests).

Design follows the execution model of the mature open-source engines (Qlib's ``Exchange``,
zipline / backtrader's next-bar fills) with A-share rules made explicit:

* **No look-ahead.** The strategy is called after the close of the signal day ``t`` and only
  sees data up to ``t``. With ``timing="next_open"`` the orders fill at the open of ``t+1``
  (``"close"`` keeps the paper's same-close convention for comparability).
* **Tradability at the fill.** Suspended names cannot trade; a name opening (or, for
  ``"close"`` fills, closing) at the limit-up cannot be bought, at the limit-down cannot be sold.
  Limits: 10% main board, 5% ST, 20% STAR / ChiNext after 2020-08-24, 30% BSE.
* **Delisting.** A name that never trades again is sold at its last price into cash.
* **Liquidity.** Each order fills at most ``participation_cap`` of the day's traded amount;
  the rest is not executed (buys stay in cash, sells stay held).
* **Costs.** Commission (with a per-order minimum in yuan, given ``aum``), stamp duty on sells
  (0.1% → 0.05% from 2023-08-28), transfer fee, and square-root market impact
  ``Y · σ_daily · sqrt(order / day amount)`` (Almgren et al., 2005; Y ≈ 0.5–1).
* **Cash** earns the daily risk-free rate when one is supplied.

Daily return on a fill day e (next_open):
    (1 + r_e) = Σ w_i (open_i / preclose_i)  ->  trade at the open  ->  Σ w'_i (close_i / open_i)
which multiplies back to the close-to-close return of an untraded position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..data.panel import MarketPanel
from .costs import STAMP_DUTY_CUT
from .engine import RebalanceContext, Strategy

CASH = "CASH"


@dataclass
class Execution:
    timing: str = "next_open"            # "next_open" | "close"
    aum: float = 1e8                     # portfolio value at the start, yuan
    commission_bps: float = 2.5          # both sides
    min_commission: float = 5.0          # yuan per order
    stamp_before_bps: float = 10.0       # sells, before 2023-08-28
    stamp_after_bps: float = 5.0         # sells, from 2023-08-28
    transfer_bps: float = 0.1            # 过户费 0.001%, both sides
    impact_y: float = 0.5                # square-root impact coefficient (0 = off)
    participation_cap: float | None = 0.10   # max share of a day's amount per order (None = unlimited)

    @classmethod
    def frictionless(cls, timing: str = "close") -> "Execution":
        return cls(timing=timing, commission_bps=0, min_commission=0, stamp_before_bps=0, stamp_after_bps=0,
                   transfer_bps=0, impact_y=0, participation_cap=None)


@dataclass
class SimResult:
    name: str
    returns: pd.Series                   # daily net returns
    returns_gross: pd.Series             # before costs
    weights: dict[pd.Timestamp, pd.Series]   # after each fill (incl. CASH)
    turnover: pd.Series                  # one-sided traded weight per fill day
    costs: pd.DataFrame                  # per fill day: commission, stamp, transfer, impact (fractions of NAV)
    unfilled: pd.Series                  # weight that could not be traded (restrictions + liquidity)
    blocked: dict[pd.Timestamp, dict[str, str]] = field(default_factory=dict)   # code -> reason

    @property
    def nav(self) -> pd.Series:
        return (1 + self.returns).cumprod()


def _vol20(panel: MarketPanel, d: pd.Timestamp, codes: list[str]) -> pd.Series:
    win = panel.window(d, 21)
    return panel.ret.loc[win, codes].std().fillna(0.02).clip(lower=0.005)


def _grow(w: pd.Series, r: pd.Series, rf: float) -> tuple[float, pd.Series]:
    """Apply per-name returns ``r`` (missing -> 0) and the cash rate; return (portfolio return, drifted weights)."""
    g = (1 + r.reindex(w.index).fillna(0.0))
    if CASH in g.index:
        g[CASH] = 1 + rf
    port = float((w * g).sum()) - 1
    return port, w * g / (1 + port)


def simulate(panel: MarketPanel, strategy: Strategy, rebalance_dates: list[pd.Timestamp],
             execution: Execution | None = None, end: pd.Timestamp | None = None,
             rf_daily: pd.Series | None = None,
             on_rebalance: Callable[[int, int], None] | None = None) -> SimResult:
    ex = execution or Execution()
    rebal = sorted({panel.dates[panel.loc_date(d)] for d in rebalance_dates})
    last = panel.dates[panel.loc_date(end)] if end is not None else panel.dates[-1]
    days = panel.dates[(panel.dates >= rebal[0]) & (panel.dates <= last)]
    pos = {d: i for i, d in enumerate(panel.dates)}
    # signal on t -> fill on t (close) or on the next trading day (next_open)
    fill_of = {}
    for t in rebal:
        i = pos[t] + (1 if ex.timing == "next_open" else 0)
        if i < len(panel.dates) and panel.dates[i] <= last:
            fill_of[panel.dates[i]] = t
    alive_after = panel.close[::-1].notna().cumsum()[::-1]       # >0 while the name still has a future bar

    w = pd.Series({CASH: 1.0})
    nav = 1.0
    out_r, out_g, turns, unfilled = {}, {}, {}, {}
    cost_rows, weights, blocked = {}, {}, {}
    pending_target: pd.Series | None = None
    done = 0

    for d in days:
        rf = float(rf_daily.get(d, 0.0)) if rf_daily is not None else 0.0
        stocks = [c for c in w.index if c != CASH]
        trading = panel.tradestatus.loc[d, stocks] > 0 if stocks else pd.Series(dtype=bool)
        fill_open = d in fill_of and ex.timing == "next_open"
        # ---- 1) the part of the day before any trade: overnight on an open-fill day, the whole
        #         day for close fills (trades happen at the close), nothing otherwise
        if fill_open:
            r_on = (panel.open.loc[d, stocks] / panel.preclose.loc[d, stocks] - 1).where(trading) if stocks else pd.Series(dtype=float)
            pre_ret, w = _grow(w, r_on, 0.0)
        elif ex.timing == "close":
            pre_ret, w = _grow(w, panel.ret.loc[d, stocks] if stocks else pd.Series(dtype=float), rf)
        else:
            pre_ret = 0.0
        cost_frac = 0.0
        # ---- 2) the strategy signal (after the close of t)
        if d in rebal:
            ts = panel.tradestatus.loc[d]
            up, dn = panel.limit_flags(d)
            ctx_t = RebalanceContext(asof=d, panel=panel, drift_weights=w.drop(CASH, errors="ignore"),
                                     suspended=set(ts.index[ts <= 0]), no_buy=set(up.index[up]), no_sell=set(dn.index[dn]))
            pending_target = strategy.target_weights(ctx_t)
            done += 1
            if on_rebalance:
                on_rebalance(done, len(rebal))
        # ---- 3) the fill
        if d in fill_of and pending_target is not None:
            nav_yuan = ex.aum * nav * (1 + pre_ret)
            ts = panel.tradestatus.loc[d]
            up, dn = panel.limit_flags(d, at="open" if ex.timing == "next_open" else "close")
            delisted = {c for c in w.index if c != CASH and alive_after.at[d, c] == 0} if len(w) > 1 else set()
            ctx = RebalanceContext(asof=d, panel=panel, drift_weights=w.drop(CASH, errors="ignore"),
                                   suspended=set(ts.index[ts <= 0]) - delisted, no_buy=set(up.index[up]), no_sell=set(dn.index[dn]))
            held = w.drop(CASH, errors="ignore")
            cash0 = float(w.get(CASH, 0.0))
            # delisted names: liquidate at the last price
            tgt = pending_target.drop([c for c in delisted if c in pending_target.index], errors="ignore")
            tgt = tgt / tgt.sum() if tgt.sum() > 0 else tgt
            # a blocked order simply fails (as in Qlib's Exchange): the name stays at its current
            # weight and nothing is reallocated; buys are later limited by the cash available
            names = tgt.index.union(held.index)
            feasible = tgt.reindex(names).fillna(0.0)
            cur = held.reindex(names).fillna(0.0)
            for c in names:
                if c in ctx.suspended or (c in ctx.no_buy and feasible[c] > cur[c]) or (c in ctx.no_sell and feasible[c] < cur[c]):
                    feasible[c] = cur[c]
            delta = feasible - cur
            reasons = {}
            for c in names:
                if c in ctx.suspended and abs(pending_target.get(c, 0) - held.get(c, 0)) > 1e-6:
                    reasons[c] = "停牌"
                elif c in ctx.no_buy and pending_target.get(c, 0) > held.get(c, 0) + 1e-6:
                    reasons[c] = "涨停"
                elif c in ctx.no_sell and pending_target.get(c, 0) < held.get(c, 0) - 1e-6:
                    reasons[c] = "跌停"
            # liquidity cap (fraction of NAV tradable per name)
            amt = panel.amount.loc[d].reindex(names).fillna(0.0)
            if ex.participation_cap is not None:
                cap = (ex.participation_cap * amt / nav_yuan)
                cap[list(delisted & set(names))] = np.inf
                capped = delta.clip(lower=-cap, upper=cap)
                for c in names[(capped - delta).abs() > 1e-9]:
                    reasons.setdefault(c, "流动性")
            else:
                capped = delta
            sells = (-capped).clip(lower=0)
            buys = capped.clip(lower=0)
            cash_avail = cash0 + float(sells.sum())
            if buys.sum() > cash_avail + 1e-12:
                buys = buys * (cash_avail / buys.sum())
            exec_delta = buys - sells
            new = held.reindex(names).fillna(0.0) + exec_delta
            new = new[new > 1e-10]
            cash = cash_avail - float(buys.sum())
            # costs (fractions of NAV)
            traded = exec_delta.abs()
            traded = traded[traded > 1e-10]
            stamp_bps = ex.stamp_before_bps if d < STAMP_DUTY_CUT else ex.stamp_after_bps
            commission = float(np.maximum(traded * ex.commission_bps / 1e4, ex.min_commission / nav_yuan).sum()) if len(traded) else 0.0
            stamp = float(sells.sum() * stamp_bps / 1e4)
            transfer = float(traded.sum() * ex.transfer_bps / 1e4)
            impact = 0.0
            if ex.impact_y > 0 and len(traded):
                sig = _vol20(panel, d, list(traded.index.difference(list(delisted))))
                part = (traded * nav_yuan / amt.reindex(traded.index).replace(0, np.nan)).reindex(sig.index).fillna(1.0)
                impact = float((traded.reindex(sig.index) * ex.impact_y * sig * np.sqrt(part.clip(upper=1.0))).sum())
            cost_frac = commission + stamp + transfer + impact
            w = pd.concat([new, pd.Series({CASH: cash})])
            w = w / w.sum()
            turns[d] = float(traded.sum()) / 2 if len(held) else float(traded.sum())
            unfilled[d] = float((delta - exec_delta).abs().sum())
            cost_rows[d] = {"commission": commission, "stamp": stamp, "transfer": transfer, "impact": impact}
            weights[d] = w.copy()
            if reasons:
                blocked[d] = reasons
            pending_target = None
        # ---- 4) the rest of the day: intraday after an open fill, the whole day on other days
        stocks = [c for c in w.index if c != CASH]
        if fill_open:
            tr = panel.tradestatus.loc[d, stocks] > 0
            post, w = _grow(w, (panel.close.loc[d, stocks] / panel.open.loc[d, stocks] - 1).where(tr), rf)
        elif ex.timing == "next_open":
            post, w = _grow(w, panel.ret.loc[d, stocks] if stocks else pd.Series(dtype=float), rf)
        else:
            post = 0.0
        gross = (1 + pre_ret) * (1 + post) - 1
        net = (1 + pre_ret) * (1 - cost_frac) * (1 + post) - 1
        nav *= 1 + net
        out_g[d], out_r[d] = gross, net

    costs = pd.DataFrame.from_dict(cost_rows, orient="index") if cost_rows else pd.DataFrame(columns=["commission", "stamp", "transfer", "impact"])
    return SimResult(name=getattr(strategy, "name", "strategy"), returns=pd.Series(out_r), returns_gross=pd.Series(out_g),
                     weights=weights, turnover=pd.Series(turns), costs=costs, unfilled=pd.Series(unfilled), blocked=blocked)
