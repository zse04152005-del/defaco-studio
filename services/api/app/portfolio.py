"""Live portfolio optimisation for the 组合构建 / 风险解析 pages (F5/F6).

The research panel (``$DEFACO_DATA``, default ``data``) is loaded once at startup — only the
last few years of bars, enough for a 250-day window plus the 252-day listing rule — and every
request then fits one risk model on the 250-day window and solves the long-only QP:

    POST /api/portfolio/optimize
    {"model": "pca_factor", "objective": "minvar", "max_weight": 0.10, "industry_cap": 0.35,
     "universe": ["600519.SH", ...] | null, "asof": "2025-12-31" | null}

The response has the same shape as one entry of ``web/public/demo-data/portfolio.json``
``models[...]`` (plus ``asof``, ``universe_size``, ``constraints``, ``corr`` and timing).

    POST /api/portfolio/rebalance   (the 调仓 page)
    {...the optimize fields..., "holdings": [{"code", "shares"}], "cash": 200000,
     "prices": {"600519.SH": 1450.0} | null, "frozen_buy": [...], "frozen_sell": [...]}

turns the continuous target into whole-lot holdings with ``quantkit.optim.lots.lot_portfolio``: from an empty
account spending may be 90%-110% of the cash; otherwise at most the account value, starting from the current
holdings, with each lot's fees charged against the risk it removes (so lot rounding alone never trades).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantkit.data.panel import MarketPanel
from quantkit.data.store import DataStore
from quantkit.features.characteristics import ALL_CHARS
from quantkit.ipca import CharacteristicPanel
from quantkit.data.sw import sw_name
from quantkit.optim.allocators import bayes_stein_mean, hrp, mean_variance, risk_parity
from quantkit.optim.lots import fee, lot_portfolio
from quantkit.optim.minvar import Constraints, min_variance
from quantkit.pipeline import load_panel
from quantkit.riskmodels import CovarianceRepr, RiskModelInput, build
from quantkit.universe import UniverseRule, eligible

log = logging.getLogger("api.portfolio")

MODELS = {"defaco": "DeFaCo（论文 M4）", "lw_nonlinear": "非线性收缩 NLS", "lw_linear": "线性收缩 LW", "pca_factor": "PCA 因子模型",
          "sample": "样本协方差", "ewma": "EWMA"}
CHAR_MODELS = {"defaco"}      # risk models that need the characteristic panel
OBJECTIVES = {"minvar": "最小方差", "rp": "风险平价", "hrp": "HRP", "mv": "均值-方差"}
WINDOW_DAYS = 250
HISTORY_YEARS = 3          # bars kept in memory: window + listing rule + some room for older asof
DEFAULT_RULE = UniverseRule(indices=("hs300",), top_n_by_mcap=100)
ANY_RULE = UniverseRule(indices=None)   # user-supplied universe: same tradability filters, no index


class OptimizeError(ValueError):
    """Bad request (infeasible constraints, unknown codes, ...): reported as HTTP 4xx."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


class _RecentBars(DataStore):
    """DataStore that only reads bars from ``start`` on (a filtered parquet scan, ~0.5 s)."""

    def __init__(self, data_dir: Path, start: pd.Timestamp):
        super().__init__(data_dir)
        self.start = start

    def read(self, *parts: str, columns: list[str] | None = None) -> pd.DataFrame:
        if parts[-1] == "bars.parquet":
            return pd.read_parquet(self.root.joinpath(*parts), columns=columns, filters=[("date", ">=", self.start)])
        return super().read(*parts, columns=columns)


@dataclass
class PortfolioEngine:
    panel: MarketPanel
    names: dict[str, str] = field(default_factory=dict)
    dataset: str = "data"
    data_dir: Path | None = None
    _chars: CharacteristicPanel | None = None
    _cov_cache: "OrderedDict[tuple, tuple[CovarianceRepr, dict]]" = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_store(cls, data_dir: Path, years: int = HISTORY_YEARS) -> "PortfolioEngine":
        t0 = time.perf_counter()
        last = pd.read_parquet(data_dir / "curated" / "trade_dates.parquet")["date"].max()
        store = _RecentBars(data_dir, pd.Timestamp(last) - pd.DateOffset(years=years))
        panel = load_panel(store)
        names = {}
        if panel.members is not None and "name" in panel.members:
            m = panel.members.sort_values("snapshot")
            names = m.drop_duplicates("code", keep="last").set_index("code")["name"].to_dict()
        log.info("portfolio panel loaded from %s: %d dates x %d codes in %.1fs",
                 data_dir.name, len(panel.dates), len(panel.codes), time.perf_counter() - t0)
        return cls(panel=panel, names=names, dataset=data_dir.name, data_dir=data_dir)

    # ------------------------------------------------------------------ characteristics (DeFaCo, exposures)
    @property
    def chars(self) -> CharacteristicPanel | None:
        """Month-end characteristic snapshots covering the loaded bars (lazy; None if unavailable)."""
        with self._lock:
            if self._chars is None:
                from quantkit.config import load_settings
                root = self.data_dir or load_settings().data_dir
                path = Path(root) / "curated" / "characteristics_raw.parquet"
                if not path.exists():
                    return None
                start = self.panel.dates[0] - pd.DateOffset(months=2)
                raw = pd.read_parquet(path, filters=[("date", ">=", start)])
                self._chars = CharacteristicPanel(raw, ALL_CHARS)
            return self._chars

    def char_exposures(self, asof: pd.Timestamp, w: pd.Series) -> dict | None:
        """Weighted average of the characteristics' ranks within the universe (-0.5 .. 0.5), as in the thesis (Section 7.5)."""
        ch = self.chars
        if ch is None:
            return None
        snap = ch.snapshot_for(asof, strict=False)
        if snap is None:
            return None
        codes = list(w.index)
        Z = ch.matrix(snap, codes)[:, 1:]
        return {"names": list(ch.cols), "values": [round(float(x), 4) for x in w.to_numpy() @ Z]}

    # ------------------------------------------------------------------ universe
    def resolve_asof(self, asof: str | None) -> pd.Timestamp:
        p = self.panel
        if asof is None:
            return p.dates[-1]
        try:
            i = p.loc_date(pd.Timestamp(asof))
        except ValueError as e:
            raise OptimizeError(f"asof {asof} 早于已加载的数据（{p.dates[0]:%Y-%m-%d} 起）") from e
        if i < WINDOW_DAYS + DEFAULT_RULE.min_listing_days:
            raise OptimizeError(f"asof {asof} 太早：内存中只保留 {p.dates[0]:%Y-%m-%d} 之后的行情，"
                                f"最早可用 {p.dates[WINDOW_DAYS + DEFAULT_RULE.min_listing_days]:%Y-%m-%d}")
        return p.dates[i]

    def universe(self, asof: pd.Timestamp, codes: list[str] | None) -> tuple[list[str], list[str]]:
        """(investable codes, dropped codes). ``None`` = CSI300 top-100 by float market cap."""
        if not codes:
            return eligible(self.panel, asof, DEFAULT_RULE), []
        wanted = list(dict.fromkeys(c.strip().upper() for c in codes if c.strip()))
        known = [c for c in wanted if c in self.panel.codes]
        ok = set(eligible(self.panel.subset(known, self.panel.dates[0]), asof, ANY_RULE)) if known else set()
        return [c for c in wanted if c in ok], [c for c in wanted if c not in ok]

    def industries(self, asof: pd.Timestamp, codes: list[str]) -> pd.Series:
        return self.panel.industry_at(asof, codes).map(lambda x: "NA" if x == "NA" else sw_name(x))

    # ------------------------------------------------------------------ risk model (cached)
    def covariance(self, model: str, asof: pd.Timestamp, codes: list[str]) -> tuple[CovarianceRepr, dict]:
        key = (model, asof, tuple(codes))
        with self._lock:
            if key in self._cov_cache:
                self._cov_cache.move_to_end(key)
                return self._cov_cache[key]
        p = self.panel
        win = p.window(asof, WINDOW_DAYS)
        extras = {}
        if model in CHAR_MODELS:
            if self.chars is None:
                raise OptimizeError("DeFaCo 需要特征数据（curated/characteristics_raw.parquet）", status=503)
            extras["chars"] = self.chars
        data = RiskModelInput(asof=asof, returns=p.ret.loc[win, codes], industries=p.industry_at(asof, codes),
                              market=p.market_ret.loc[win] if p.market_ret is not None else None, extras=extras)
        try:
            rm = build(model).fit(data)
        except FileNotFoundError as e:   # DeFaCo models not exported
            raise OptimizeError("DeFaCo 的训练参数未安装（data/models/defaco_m4.json）：开源版本不含论文模型参数，"
                                "将在论文答辩后发布；其余风险模型可正常使用", status=503) from e
        out = (rm.covariance(), rm.explain())
        with self._lock:
            self._cov_cache[key] = out
            while len(self._cov_cache) > 32:
                self._cov_cache.popitem(last=False)
        return out

    # ------------------------------------------------------------------ optimise
    def optimize(self, model: str = "pca_factor", objective: str = "minvar", max_weight: float = 0.10,
                 industry_cap: float | None = 0.35, universe: list[str] | None = None,
                 asof: str | None = None, risk_aversion: float = 5.0, compare: bool = False) -> dict:
        t0 = time.perf_counter()
        if model not in MODELS:
            raise OptimizeError(f"未知风险模型 {model!r}，可选：{', '.join(MODELS)}")
        if objective not in OBJECTIVES:
            raise OptimizeError(f"未知目标 {objective!r}，可选：{', '.join(OBJECTIVES)}")
        if not 0 < max_weight <= 1:
            raise OptimizeError("单股上限需在 (0, 1] 之间")
        if industry_cap is not None and not 0 < industry_cap <= 1:
            raise OptimizeError("行业上限需在 (0, 1] 之间")

        d = self.resolve_asof(asof)
        codes, dropped = self.universe(d, universe)
        if len(codes) < 2:
            raise OptimizeError(f"可投资股票不足 2 只（剔除 {len(dropped)} 只停牌 / ST / 上市不足一年 / 无数据的股票）")
        ind = self.industries(d, codes)
        _check_feasible(len(codes), ind, max_weight, industry_cap)

        cov, explain = self.covariance(model, d, codes)
        t_fit = time.perf_counter()
        cons = Constraints(max_weight=max_weight, groups=ind if industry_cap is not None else None,
                           group_cap=industry_cap)
        notes: list[str] = []
        extra: dict = {}
        if model == "defaco":
            notes.append(f"DeFaCo 为论文训练的 M4 模型：使用 {explain['model_year']} 年测试年的参数（κ = {explain['kappa_bp']:g} bp，"
                         f"{explain['seeds']} 个种子平均），参数在年内固定"
                         + ("；当前日期晚于样本外区间（2016–2025），结果仅作演示" if explain.get("out_of_sample_period") else ""))
        if objective == "minvar":
            w = min_variance(cov, cons)
        elif objective == "rp":
            w = risk_parity(cov, cap=max_weight)
        elif objective == "hrp":
            w = hrp(cov, cap=max_weight)
        else:   # mean-variance with Bayes-Stein shrunk expected returns
            R = self.panel.ret.loc[self.panel.window(d, WINDOW_DAYS), codes].to_numpy()
            mu, delta = bayes_stein_mean(R, cov.dense())
            w = mean_variance(cov, mu, risk_aversion, cons)
            extra["expected_returns"] = {"method": "Bayes-Stein（Jorion 1986）", "shrinkage": round(delta, 4),
                                         "risk_aversion": risk_aversion,
                                         "portfolio_ann_return": round(float(w.reindex(cov.codes).fillna(0) @ mu) * 252, 6)}
            notes.append(f"预期收益为 250 日样本均值向最小方差组合收益收缩（收缩强度 {delta:.0%}），风险厌恶系数 γ={risk_aversion:g}")
        if objective in ("rp", "hrp"):
            if industry_cap is not None:
                notes.append("风险平价 / HRP 不含行业上限约束（已忽略）；单股上限通过截断后按比例再分配实现")
        t_solve = time.perf_counter()

        out = describe(cov, w, ind, self.names, MODELS[model], explain)
        if compare:   # characteristic exposures of this portfolio and of the NLS portfolio under the same constraints
            ex = self.char_exposures(d, w.reindex(codes).fillna(0.0))
            if ex is not None:
                if model != "lw_nonlinear" and objective == "minvar":
                    cn, _ = self.covariance("lw_nonlinear", d, codes)
                    wn = min_variance(cn, cons).reindex(codes).fillna(0.0)
                    ex["compare"] = {"label": MODELS["lw_nonlinear"], "values": self.char_exposures(d, wn)["values"]}
                out["char_exposures"] = ex
        out.update({
            "model": model, "objective": objective, "asof": d.strftime("%Y-%m-%d"), "universe_size": len(codes),
            "universe_source": "custom" if universe else "default", "dropped": dropped, "notes": notes, **extra,
            "constraints": {"max_weight": max_weight, "industry_cap": industry_cap},
            "dataset": self.dataset, "window_days": WINDOW_DAYS,
            "timing_ms": {"fit": round((t_fit - t0) * 1e3), "solve": round((t_solve - t_fit) * 1e3),
                          "total": round((time.perf_counter() - t0) * 1e3)},
        })
        return out


    # ------------------------------------------------------------------ whole-lot rebalance
    def rebalance(self, holdings: list[dict], cash: float, prices: dict[str, float] | None = None,
                  frozen_buy: list[str] | None = None, frozen_sell: list[str] | None = None, **opt) -> dict:
        """Order list that takes the account to the best whole-lot holdings for the optimiser's target."""
        t0 = time.perf_counter()
        opt["compare"] = False
        r = self.optimize(**opt)
        d = pd.Timestamp(r["asof"])
        w_star = pd.Series({x["code"]: x["w"] for x in r["weights"]}, dtype=float)
        uni, _ = self.universe(d, opt.get("universe"))
        held = {}
        for h in holdings:
            c, n = str(h["code"]).strip().upper(), int(h.get("shares", 0))
            if n > 0:
                held[c] = held.get(c, 0) + n
        known = set(self.panel.codes)
        outside = {c: n for c, n in held.items() if c not in known}      # no research data: kept as they are
        codes = list(dict.fromkeys([*uni, *(c for c in held if c in known)]))
        px = self.panel.close.loc[d].reindex(codes).astype(float)
        for c, v in (prices or {}).items():
            if c in px.index and v and v > 0:
                px[c] = float(v)
        nb = set(frozen_buy or []) | set(px.index[~(px > 0)])
        ns = set(frozen_sell or []) | set(px.index[~(px > 0)])
        px = px.fillna(0.0)
        try:
            cov, _ = self.covariance(opt.get("model", "pca_factor"), d, codes)
        except Exception as e:  # noqa: BLE001  (a held stock the model cannot fit, e.g. long suspended)
            log.warning("covariance on held stocks failed (%s); held stocks outside the universe are frozen", e)
            extra = [c for c in codes if c not in uni]
            nb |= set(extra); ns |= set(extra)
            cov, _ = self.covariance(opt.get("model", "pca_factor"), d, uni)
        M = cov.as_frame().reindex(index=codes, columns=codes).to_numpy(float)
        miss = np.where(np.isnan(np.diag(M)))[0]
        fill = float(np.nanmedian(np.diag(M)))
        M = np.nan_to_num(M)
        M[miss, miss] = fill
        S = 252.0 * M
        ind = self.industries(d, codes)
        cur = pd.Series({c: held.get(c, 0) for c in codes}, dtype=float)
        out_value = sum(n * float((prices or {}).get(c, 0) or 0) for c, n in outside.items())
        value = float(cash) + float((cur * px).sum())
        opening = cur.sum() == 0
        if value <= 0:
            raise OptimizeError("账户总资产为 0：请先填写可用现金或持仓")
        objective = opt.get("objective", "minvar")
        res = lot_portfolio(codes, S, px, value, w_star, opt.get("max_weight", 0.10), ind,
                            ind_cap=opt.get("industry_cap") or 1.0, lo=0.90, hi=1.10 if opening else 1.0,
                            track=objective != "minvar", current=None if opening else cur, no_buy=nb, no_sell=ns)
        tgt = res.shares.astype(int)
        invested = float((tgt * px).sum())

        def vol(sh: pd.Series) -> float | None:
            v = (sh * px).to_numpy(float)
            return float(math.sqrt(max(v @ S @ v, 0))) / v.sum() if v.sum() > 0 else None

        ws = w_star.reindex(codes).fillna(0.0).to_numpy()
        rows, buys, sells, commission, stamp = [], 0.0, 0.0, 0.0, 0.0
        for c in codes:
            n0, n1 = int(cur[c]), int(tgt[c])
            if n0 == 0 and n1 == 0 and w_star.get(c, 0.0) <= 1e-6:
                continue
            trade, p = n1 - n0, float(px[c])
            amt = trade * p
            if trade > 0:
                buys += amt
                commission += fee(amt, "buy")
            elif trade < 0:
                sells -= amt
                f = fee(-amt, "sell")
                stamp += -amt * 5e-4
                commission += f - (-amt * 5e-4)
            rows.append({"code": c, "name": self.names.get(c, c), "industry": str(ind.get(c, "")), "price": round(p, 3),
                         "held": n0, "target_shares": n1, "trade": trade, "value": round(amt, 2),
                         "w_cur": n0 * p / value, "w_star": float(w_star.get(c, 0.0)),
                         "w_after": n1 * p / invested if invested > 0 else 0.0,
                         "freeze": "buy" if c in nb and c not in ns else ("sell" if c in ns and c not in nb else ("both" if c in nb else None))})
        for c, n in outside.items():
            rows.append({"code": c, "name": c, "industry": "", "price": float((prices or {}).get(c, 0) or 0), "held": n,
                         "target_shares": n, "trade": 0, "value": 0.0, "w_cur": 0.0, "w_star": 0.0, "w_after": 0.0, "freeze": "nodata"})
        rows.sort(key=lambda x: (-abs(x["value"]), -x["w_after"]))
        cash_after = value - invested - commission - stamp
        return {
            "asof": r["asof"], "label": r.get("label", opt.get("model")), "model": opt.get("model"), "objective": objective,
            "mode": "opening" if opening else "rebalance", "rows": rows,
            "summary": {"nav": value + out_value, "invested": invested, "buys": buys, "sells": sells, "commission": commission,
                        "stamp": stamp, "cash_after": cash_after, "extra_capital": max(0.0, -cash_after),
                        "turnover": (buys + sells) / value, "spend_ratio": res.spend_ratio, "n_orders": sum(1 for x in rows if x["trade"]),
                        "n_holdings": int((tgt > 0).sum()), "vol_before": None if opening else vol(cur), "vol_after": vol(tgt),
                        "vol_target": float(math.sqrt(max(ws @ S @ ws, 0))), "te": res.te, "moves": res.moves},
            "notes": [("开仓：花费可在资金的 90%–110% 之间，选预测风险最低的整手组合" if opening else
                       "调仓：从当前持仓出发逐手调整，每一手的费用计入目标，只有降低的风险值得付出的费用时才交易"),
                      *(["无研究数据的持仓保持不动：" + "、".join(outside)] if outside else [])],
            "timing_ms": {"total": round((time.perf_counter() - t0) * 1e3)},
        }


def _check_feasible(n: int, ind: pd.Series, max_weight: float, industry_cap: float | None) -> None:
    if n * max_weight < 1 - 1e-9:
        raise OptimizeError(f"约束不可行：{n} 只股票 × 单股上限 {max_weight:.0%} < 100%，"
                            f"请把单股上限调到 ≥ {math.ceil(100 / n)}% 或扩大股票池")
    if industry_cap is not None:
        room = sum(min(industry_cap, k * max_weight) for k in ind.value_counts())
        if room < 1 - 1e-9:
            raise OptimizeError(f"约束不可行：在单股上限 {max_weight:.0%}、行业上限 {industry_cap:.0%} 下，"
                                f"{ind.nunique()} 个行业最多只能装下 {room:.0%} 的仓位，请放宽行业上限或扩大股票池")


def describe(cov: CovarianceRepr, w: pd.Series, ind: pd.Series, names: dict[str, str], label: str,
             explain: dict | None = None, corr_top: int = 40) -> dict:
    """One ``portfolio.json`` ``models[...]`` entry (+ ``corr``) for weights ``w`` under ``cov``."""
    S = cov.dense()
    wv = w.reindex(cov.codes).fillna(0.0).to_numpy()
    tot = float(wv @ S @ wv)
    rc = pd.Series((S @ wv) * wv / tot, index=cov.codes)       # risk contributions, sum to 1
    risk: dict = {"total_vol": math.sqrt(tot * 252)}
    if cov.is_factor:
        fac = float(wv @ (cov.B @ cov.sigma_f @ cov.B.T) @ wv)
        risk.update({"factor_share": fac / tot, "idio_share": 1 - fac / tot,
                     "exposures": [round(float(x), 6) for x in cov.B.T @ wv]})
    elif cov.parts:
        sh = {k: float(wv @ v @ wv) / tot for k, v in cov.parts.items()}
        risk.update({"factor_share": sh.get("factor", 0.0), "industry_share": sh.get("industry", 0.0),
                     "idio_share": sh.get("specific", 0.0)})
    top = w[w > 1e-4].sort_values(ascending=False)
    weights = [{"code": c, "name": names.get(c, c), "industry": ind.get(c, "NA"),
                "w": round(float(v), 6), "rc": round(float(rc.get(c, 0.0)), 6)} for c, v in top.items()]
    ind_w = top.groupby(ind.reindex(top.index).fillna("NA")).sum().sort_values(ascending=False)

    # correlation of the largest holdings, ordered by industry (block structure for the Risk page)
    pos = {c: i for i, c in enumerate(cov.codes)}
    sel = sorted(top.index[:corr_top], key=lambda c: (ind.get(c, ""), c))
    ix = [pos[c] for c in sel]
    sub = S[np.ix_(ix, ix)]
    sd = np.sqrt(np.diag(sub))
    corr = sub / np.outer(sd, sd) if len(ix) else sub

    return {"label": label, "weights": weights,
            "industry_weights": {k: round(float(v), 6) for k, v in ind_w.items()},
            "risk": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in risk.items()},
            "n_holdings": int(len(top)), "hhi": round(float((top ** 2).sum()), 6),
            "explain": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in (explain or {}).items()},
            "corr": {"codes": sel, "names": [names.get(c, c) for c in sel],
                     "industries": [ind.get(c, "") for c in sel], "matrix": np.round(corr, 3).tolist()}}
