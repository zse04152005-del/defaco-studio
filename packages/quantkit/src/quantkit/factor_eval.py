"""Single-factor research in the style of alphalens (Quantopian, Apache-2.0), vectorised over
the whole factor library.

Each month end t, for the stocks in the chosen universe (historical CSI300 or CSI800 members,
tradable, not ST, listed ≥ 1 year):
  1. winsorise each factor at median ± 5·MAD and z-score it;
  2. optionally neutralise: residual of an OLS on industry dummies (+ log float market cap) —
     done for all factors at once with one projection matrix;
  3. relate it to the forward return from t to t+h months (total return, close to close):
     IC (Pearson), Rank IC (Spearman), quintile portfolios (equal weight), long-short.
Mean IC t-statistics use Newey-West standard errors with h-1 lags (overlapping returns for h>1).
Data: data/curated/factor_base.parquet + factors_alpha.parquet + characteristics_raw.parquet.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.store import DataStore
from .features.alphas import CATALOG
from .features.characteristics import ALL_CHARS
from .metrics.tests import hac

CHAR_CN = {
    "size": "市值", "mom_12_2": "12-2 月动量", "mom_6_1": "6-1 月动量", "rev_1m": "1 月反转", "vol_1m": "1 月波动率",
    "vol_12m": "12 月波动率", "maxret_1m": "上月最大日收益", "turn_1m": "1 月换手", "turn_6m": "6 月换手", "abturn": "异常换手",
    "illiq": "非流动性 Amihud", "beta_12m": "12 月 β", "ivol_3m": "3 月特质波动", "high52": "距 52 周高点", "ep": "E/P",
    "bm": "B/M", "sp": "S/P", "cfp": "CF/P", "roe": "ROE", "gross_margin": "毛利率", "revenue_yoy": "营收同比",
    "profit_yoy": "利润同比", "asset_growth": "资产增长", "leverage": "杠杆",
}


def _mad_z(X: np.ndarray) -> np.ndarray:
    """Column-wise MAD winsorisation (±5 MAD) and z-score; NaN kept (all-NaN columns stay NaN)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _mad_z_impl(X)


def _mad_z_impl(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    mad = np.where(mad > 0, mad, np.nanstd(X, axis=0) + 1e-12)
    X = np.clip(X, med - 5 * mad, med + 5 * mad)
    sd = np.nanstd(X, axis=0)
    return (X - np.nanmean(X, axis=0)) / np.where(sd > 0, sd, np.nan)


def _rank_cols(X: np.ndarray) -> np.ndarray:
    return pd.DataFrame(X).rank(pct=True).to_numpy()


def _corr_cols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of X with y over rows where both are finite."""
    m = np.isfinite(X) & np.isfinite(y)[:, None]
    n = m.sum(axis=0)
    Xm = np.where(m, X, 0.0)
    Ym = np.where(m, y[:, None], 0.0)
    mx, my = Xm.sum(0) / np.maximum(n, 1), Ym.sum(0) / np.maximum(n, 1)
    cov = (Xm * Ym).sum(0) / np.maximum(n, 1) - mx * my
    vx = (Xm * Xm).sum(0) / np.maximum(n, 1) - mx ** 2
    vy = (Ym * Ym).sum(0) / np.maximum(n, 1) - my ** 2
    r = cov / np.sqrt(np.maximum(vx * vy, 1e-30))
    return np.where(n >= 30, r, np.nan)


@dataclass
class Catalogue:
    name: str
    family: str
    formula: str
    desc: str
    modified: bool = False


def catalogue() -> dict[str, Catalogue]:
    out = {c: Catalogue(c, "基本面 / 价量特征", CHAR_CN.get(c, c), "项目自建的点时特征（截面分位前的原值）") for c in ALL_CHARS}
    for k, f in CATALOG.items():
        out[k] = Catalogue(k, f.family, f.formula, f.desc, f.modified)
    return out


@dataclass
class FactorLab:
    months: pd.DatetimeIndex
    codes: pd.Index
    values: dict[str, np.ndarray]                     # factor -> (months x codes), float32
    base: pd.DataFrame                                # long month-end table
    _prep: dict = field(default_factory=dict)         # (universe, neutral) -> per-month arrays

    @classmethod
    def load(cls, store: DataStore) -> "FactorLab":
        base = store.read("curated", "factor_base.parquet")
        base["date"] = pd.to_datetime(base["date"])
        months = pd.DatetimeIndex(sorted(base["date"].unique()))
        alpha = store.read("curated", "factors_alpha.parquet")
        chars = store.read("curated", "characteristics_raw.parquet")
        codes = pd.Index(sorted(base["code"].unique()))
        values = {}
        for df, cols in ((chars, ALL_CHARS), (alpha, [c for c in alpha.columns if c not in ("date", "code")])):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"].isin(months) & df["code"].isin(codes)].set_index(["date", "code"])
            for c in cols:
                values[c] = df[c].astype(np.float32).unstack().reindex(index=months, columns=codes).to_numpy()
        return cls(months=months, codes=codes, values=values, base=base)

    # ------------------------------------------------------------------ preparation
    def prepared(self, universe: str = "csi800", neutral: str = "industry_size") -> dict:
        key = (universe, neutral)
        if key in self._prep:
            return self._prep[key]
        names = list(self.values)
        cpos = {c: i for i, c in enumerate(self.codes)}
        out = {"names": names, "months": [], "Z": [], "fwd": {1: [], 3: [], 6: []}, "codes": []}
        for t, g in self.base.groupby("date"):
            if universe == "csi300":
                g = g[g["in300"]]
            if len(g) < 50:
                continue
            k = self.months.get_loc(t)
            idx = np.array([cpos[c] for c in g["code"]])
            X = np.column_stack([self.values[n][k, idx] for n in names]).astype(float)
            Z = _mad_z(X)
            if neutral != "none":   # residualise all factors at once on industry dummies (+ size)
                D = pd.get_dummies(g["industry"].fillna("NA")).to_numpy(dtype=float)
                if neutral == "industry_size":
                    lm = g["log_mcap"].to_numpy(dtype=float)
                    D = np.column_stack([D, np.nan_to_num(lm - np.nanmean(lm))])
                Zf = np.nan_to_num(Z)
                beta, *_ = np.linalg.lstsq(D, Zf, rcond=None)
                R = Zf - D @ beta
                Z = np.where(np.isnan(Z), np.nan, R)
                sd = np.nanstd(Z, axis=0)
                Z = Z / np.where(sd > 0, sd, np.nan)
            out["months"].append(t)
            out["Z"].append(Z.astype(np.float32))
            out["codes"].append(g["code"].to_numpy())
            for h in (1, 3, 6):
                out["fwd"][h].append(g[f"fwd{h}"].to_numpy(dtype=float))
        self._prep[key] = out
        return out

    # ------------------------------------------------------------------ evaluation
    def ic_matrix(self, universe="csi800", neutral="industry_size", h=1, method="rank") -> pd.DataFrame:
        P = self.prepared(universe, neutral)
        rows = []
        for Z, y in zip(P["Z"], P["fwd"][h]):
            if not np.isfinite(y).any():
                rows.append(np.full(len(P["names"]), np.nan))
                continue
            if method == "rank":
                rows.append(_corr_cols(_rank_cols(Z), pd.Series(y).rank(pct=True).to_numpy()))
            else:
                rows.append(_corr_cols(Z.astype(float), y))
        return pd.DataFrame(rows, index=P["months"], columns=P["names"])

    def summary(self, universe="csi800", neutral="industry_size", h=1) -> pd.DataFrame:
        ric = self.ic_matrix(universe, neutral, h, "rank")
        ic = self.ic_matrix(universe, neutral, h, "pearson")
        P = self.prepared(universe, neutral)
        lag = max(0, h - 1)
        rows = {}
        ls_all = self.quantile_returns_all(universe, neutral, h)
        cat = catalogue()
        for n in ric.columns:
            s = ric[n].dropna()
            if len(s) < 24:
                continue
            se = np.sqrt(hac(s.to_numpy() - s.mean(), lag)[0, 0] / len(s)) if lag else s.std(ddof=1) / np.sqrt(len(s))
            sign = 1.0 if s.mean() >= 0 else -1.0
            q = ls_all[n]
            ls = sign * (q[5] - q[1])
            ls_m = ls.dropna()
            ann = (1 + ls_m).prod() ** (12 / h / len(ls_m)) - 1 if len(ls_m) else np.nan
            means = [q[g].mean() for g in range(1, 6)]
            mono = pd.Series(means).corr(pd.Series(range(1, 6)), method="spearman")
            rows[n] = {"family": cat[n].family if n in cat else "", "rank_ic": s.mean(), "ic": ic[n].mean(),
                       "icir": s.mean() / s.std(ddof=1) * np.sqrt(12 / h), "t": s.mean() / se if se > 0 else np.nan,
                       "ic_pos": (s * sign > 0).mean(), "direction": int(sign), "ls_ann": ann,
                       "ls_sharpe": ls_m.mean() / ls_m.std(ddof=1) * np.sqrt(12 / h) if ls_m.std() > 0 else np.nan,
                       "monotonic": mono * sign, "turnover": self.top_turnover(n, universe, neutral, sign),
                       "coverage": float(np.mean([np.isfinite(Z[:, P['names'].index(n)]).mean() for Z in P["Z"]])),
                       "months": len(s)}
        return pd.DataFrame(rows).T

    def quantile_returns_all(self, universe, neutral, h, q=5) -> dict[str, dict[int, pd.Series]]:
        P = self.prepared(universe, neutral)
        F = len(P["names"])
        acc = {g: [] for g in range(1, q + 1)}
        for Z, y in zip(P["Z"], P["fwd"][h]):
            R = _rank_cols(Z)                                  # in (0, 1]
            grp = np.ceil(R * q)
            ok = np.isfinite(y)
            for g in range(1, q + 1):
                m = (grp == g) & ok[:, None]
                cnt = m.sum(0)
                acc[g].append(np.where(cnt > 0, (np.where(m, y[:, None], 0.0)).sum(0) / np.maximum(cnt, 1), np.nan))
        out = {}
        for j, n in enumerate(P["names"]):
            out[n] = {g: pd.Series([a[j] for a in acc[g]], index=P["months"]) for g in acc}
        return out

    def top_turnover(self, name, universe, neutral, sign=1.0, q=5) -> float:
        P = self.prepared(universe, neutral)
        j = P["names"].index(name)
        prev, outs = None, []
        for Z, codes in zip(P["Z"], P["codes"]):
            z = Z[:, j] * sign
            ok = np.isfinite(z)
            if ok.sum() < 20:
                prev = None
                continue
            thr = np.nanquantile(z, 1 - 1 / q)
            top = set(codes[ok & (z >= thr)])
            if prev:
                outs.append(1 - len(top & prev) / max(len(top), 1))
            prev = top
        return float(np.mean(outs)) if outs else np.nan

    def detail(self, name: str, universe="csi800", neutral="industry_size", h=1) -> dict:
        P = self.prepared(universe, neutral)
        j = P["names"].index(name)
        ric = self.ic_matrix(universe, neutral, h, "rank")[name]
        ic = self.ic_matrix(universe, neutral, h, "pearson")[name]
        sign = 1.0 if ric.mean() >= 0 else -1.0
        qr = self.quantile_returns_all(universe, neutral, h)[name]
        # IC decay: factor at t vs the single-month return k months later
        decay = []
        for k in range(1, 7):
            vals = []
            for i in range(len(P["months"]) - k + 1):
                Z, codes = P["Z"][i][:, j], P["codes"][i]
                if i + k - 1 >= len(P["months"]):
                    break
                # month-k return = fwd1 at month t+k-1 for the same codes
                y_map = dict(zip(P["codes"][i + k - 1], P["fwd"][1][i + k - 1]))
                y = np.array([y_map.get(c, np.nan) for c in codes])
                ok = np.isfinite(Z) & np.isfinite(y)
                if ok.sum() >= 30:
                    vals.append(pd.Series(Z[ok]).rank().corr(pd.Series(y[ok]).rank()))
            decay.append(float(np.nanmean(vals)) if vals else None)
        # autocorrelation of the factor (rank) between consecutive months
        ac = []
        for i in range(1, len(P["months"])):
            a = dict(zip(P["codes"][i - 1], P["Z"][i - 1][:, j]))
            pairs = [(a[c], v) for c, v in zip(P["codes"][i], P["Z"][i][:, j]) if c in a and np.isfinite(v) and np.isfinite(a[c])]
            if len(pairs) > 30:
                x, y = zip(*pairs)
                ac.append(pd.Series(x).rank().corr(pd.Series(y).rank()))
        ls = (sign * (qr[5] - qr[1])).dropna()
        cum = {g: ((1 + qr[g].fillna(0)).cumprod() if h == 1 else None) for g in qr}
        yearly = ric.groupby(ric.index.year).mean()
        s = ric.dropna()
        lag = max(0, h - 1)
        se = np.sqrt(hac(s.to_numpy() - s.mean(), lag)[0, 0] / len(s)) if lag else s.std(ddof=1) / np.sqrt(len(s))
        cat = catalogue().get(name)
        return {
            "name": name, "family": cat.family if cat else "", "formula": cat.formula if cat else "", "desc": cat.desc if cat else "",
            "modified": bool(cat.modified) if cat else False, "universe": universe, "neutral": neutral, "h": h,
            "stats": {"rank_ic": float(s.mean()), "ic": float(ic.mean()), "icir": float(s.mean() / s.std(ddof=1) * np.sqrt(12 / h)),
                      "t": float(s.mean() / se), "ic_pos": float((s * sign > 0).mean()), "direction": int(sign),
                      "ls_ann": float((1 + ls).prod() ** (12 / h / len(ls)) - 1) if len(ls) else None,
                      "ls_vol": float(ls.std(ddof=1) * np.sqrt(12 / h)) if len(ls) > 1 else None,
                      "ls_sharpe": float(ls.mean() / ls.std(ddof=1) * np.sqrt(12 / h)) if len(ls) > 1 else None,
                      "ls_maxdd": float(((1 + ls).cumprod() / (1 + ls).cumprod().cummax() - 1).min()) if len(ls) else None,
                      "turnover": self.top_turnover(name, universe, neutral, sign), "autocorr": float(np.nanmean(ac)) if ac else None,
                      "months": int(len(s))},
            "ic_series": [[str(d.date()), None if not np.isfinite(v) else round(float(v), 4)] for d, v in ric.items()],
            "ic_pearson": [[str(d.date()), None if not np.isfinite(v) else round(float(v), 4)] for d, v in ic.items()],
            "quantile_mean": {g: float(qr[g].mean()) for g in qr},
            "quantile_cum": {g: [[str(d.date()), round(float(v), 4)] for d, v in c.items()] for g, c in cum.items() if c is not None},
            "ls_cum": [[str(d.date()), round(float(v), 4)] for d, v in (1 + ls).cumprod().items()],
            "decay": decay, "yearly_ic": {int(y): round(float(v), 4) for y, v in yearly.items() if np.isfinite(v)},
        }

    def corr(self, names: list[str], universe="csi800", neutral="industry_size") -> pd.DataFrame:
        P = self.prepared(universe, neutral)
        js = [P["names"].index(n) for n in names]
        acc = np.zeros((len(js), len(js)))
        cnt = 0
        for Z in P["Z"]:
            R = pd.DataFrame(Z[:, js]).rank()
            C = R.corr().to_numpy()
            if np.isfinite(C).all():
                acc += C
                cnt += 1
        return pd.DataFrame(acc / max(cnt, 1), index=names, columns=names)
