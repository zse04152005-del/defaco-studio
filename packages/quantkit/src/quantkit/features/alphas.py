"""Formulaic price-volume factor library: Qlib Alpha158 (MIT), WorldQuant Alpha101
(Kakushadze, 2016, "101 Formulaic Alphas") and Guotai Junan's 191 short-horizon alphas.

All operators work on numpy arrays shaped (T days, N stocks), vectorised over stocks with
bottleneck moving-window functions. Inputs (see ``Inputs``):

* time-series operators use *backward-adjusted* prices (raw x cumulative total-return factor),
  so dividends and splits do not create jumps;
* same-day cross-sectional comparisons of price *levels* use raw prices;
* formulas whose original definition depends on the price level across stocks (e.g. ranking
  ``delta(close, 1)`` or ``open - delay(high, 1)`` across stocks) are implemented in relative form
  (divided by the price); those factors are flagged ``modified=True`` in the catalogue.

Every factor value at date t uses data up to t only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import bottleneck as bn
import numpy as np
import pandas as pd

EPS = 1e-12


# ----------------------------------------------------------------------------- operators
def delay(x: np.ndarray, d: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if d < len(x):
        out[d:] = x[:-d] if d > 0 else x
    return out


def delta(x, d):
    return x - delay(x, d)


def _mc(d: int) -> int:
    return max(1, int(np.ceil(d * 0.75)))


def ts_mean(x, d):
    return bn.move_mean(x, d, min_count=_mc(d), axis=0)


def ts_sum(x, d):
    return bn.move_sum(x, d, min_count=_mc(d), axis=0)


def ts_std(x, d):
    return bn.move_std(x, d, min_count=_mc(d), axis=0, ddof=1)


def ts_min(x, d):
    return bn.move_min(x, d, min_count=_mc(d), axis=0)


def ts_max(x, d):
    return bn.move_max(x, d, min_count=_mc(d), axis=0)


def ts_rank(x, d):
    """Rank of today's value within the last d values, in [0, 1]."""
    return (bn.move_rank(x, d, min_count=_mc(d), axis=0) + 1) / 2


def ts_argmax(x, d):
    """Position of the max in the window, 1 = oldest ... d = today (Alpha101 / Qlib IdxMax)."""
    return d - bn.move_argmax(x, d, min_count=_mc(d), axis=0)


def ts_argmin(x, d):
    return d - bn.move_argmin(x, d, min_count=_mc(d), axis=0)


def ts_cov(x, y, d):
    xy = ts_mean(x * y, d)
    return (xy - ts_mean(x, d) * ts_mean(y, d)) * d / (d - 1)


def ts_corr(x, y, d):
    sx, sy = ts_std(x, d), ts_std(y, d)
    c = ts_cov(x, y, d) / (sx * sy)
    c[(sx < 1e-10) | (sy < 1e-10)] = np.nan
    return np.clip(c, -1, 1)


def ts_quantile(x, d, q):
    return pd.DataFrame(x).rolling(d, min_periods=_mc(d)).quantile(q).to_numpy()


def _time_index(x):
    return np.arange(len(x), dtype=float)[:, None] * np.ones((1, x.shape[1]))


def ts_slope(y, d):
    t = _time_index(y)
    t = np.where(np.isnan(y), np.nan, t)
    var_t = (d * d - 1) / 12.0
    return (ts_mean(t * y, d) - ts_mean(t, d) * ts_mean(y, d)) / var_t


def ts_rsquare(y, d):
    b = ts_slope(y, d)
    vy = ts_std(y, d) ** 2 * (d - 1) / d
    return np.clip(b * b * (d * d - 1) / 12.0 / (vy + EPS), 0, 1)


def ts_resid(y, d):
    t = _time_index(y)
    b = ts_slope(y, d)
    a = ts_mean(y, d) - b * ts_mean(np.where(np.isnan(y), np.nan, t), d)
    return y - (a + b * t)


def rank(x):
    """Cross-sectional percentile rank per day (NaN kept)."""
    return pd.DataFrame(x).rank(axis=1, pct=True).to_numpy()


def sma(x, n, m):
    """GTJA SMA(A, n, m): Y_t = (m A_t + (n - m) Y_{t-1}) / n."""
    return pd.DataFrame(x).ewm(alpha=m / n, adjust=False, ignore_na=True).mean().to_numpy()


def signed_power(x, p):
    return np.sign(x) * np.abs(x) ** p


# ----------------------------------------------------------------------------- inputs
@dataclass
class Inputs:
    open: np.ndarray        # adjusted
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    vwap: np.ndarray        # adjusted
    volume: np.ndarray      # shares
    amount: np.ndarray      # yuan
    ret: np.ndarray         # daily total return
    raw_open: np.ndarray
    raw_high: np.ndarray
    raw_low: np.ndarray
    raw_close: np.ndarray

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, dates: pd.DatetimeIndex, codes: pd.Index) -> "Inputs":
        def wide(col):
            w = bars.pivot_table(index="date", columns="code", values=col, aggfunc="last")
            return w.reindex(index=dates, columns=codes).to_numpy(dtype=float)
        trading = wide("tradestatus") > 0
        raw = {c: np.where(trading, wide(c), np.nan) for c in ("open", "high", "low", "close", "volume", "amount")}
        ret = np.where(trading, wide("ret"), np.nan)
        # backward-adjusted close: compound total returns, anchored at each stock's first close
        growth = np.nancumprod(np.where(np.isnan(ret), 1.0, 1.0 + ret), axis=0)
        first = pd.DataFrame(raw["close"]).bfill().to_numpy()[0]
        adj_close = growth * first
        factor = adj_close / raw["close"]
        vol = np.where(raw["volume"] > 0, raw["volume"], np.nan)
        vwap = raw["amount"] / vol
        vwap = np.where((vwap > raw["low"] * 0.9) & (vwap < raw["high"] * 1.1), vwap, (raw["high"] + raw["low"] + raw["close"]) / 3)
        return cls(open=raw["open"] * factor, high=raw["high"] * factor, low=raw["low"] * factor, close=np.where(trading, adj_close, np.nan),
                   vwap=vwap * factor, volume=vol, amount=raw["amount"], ret=ret,
                   raw_open=raw["open"], raw_high=raw["high"], raw_low=raw["low"], raw_close=raw["close"])


# ----------------------------------------------------------------------------- catalogue
@dataclass
class Factor:
    name: str
    family: str            # Alpha158 | Alpha101 | GTJA191
    formula: str
    fn: Callable[[Inputs], np.ndarray]
    desc: str = ""
    modified: bool = False


CATALOG: dict[str, Factor] = {}


def factor(name, family, formula, desc="", modified=False):
    def deco(fn):
        CATALOG[name] = Factor(name, family, formula, fn, desc, modified)
        return fn
    return deco


# ---- Qlib Alpha158 (expressions from qlib/contrib/data/loader.py, MIT) --------------------
@factor("KMID", "Alpha158", "($close-$open)/$open", "当日实体相对开盘价")
def _kmid(i): return i.close / i.open - 1
@factor("KLEN", "Alpha158", "($high-$low)/$open", "当日振幅")
def _klen(i): return (i.high - i.low) / i.open
@factor("KMID2", "Alpha158", "($close-$open)/($high-$low+1e-12)", "实体占振幅比例")
def _kmid2(i): return (i.close - i.open) / (i.high - i.low + EPS)
@factor("KUP", "Alpha158", "($high-Greater($open,$close))/$open", "上影线")
def _kup(i): return (i.high - np.fmax(i.open, i.close)) / i.open
@factor("KLOW", "Alpha158", "(Less($open,$close)-$low)/$open", "下影线")
def _klow(i): return (np.fmin(i.open, i.close) - i.low) / i.open
@factor("KSFT", "Alpha158", "(2*$close-$high-$low)/$open", "收盘在区间中的位置偏移")
def _ksft(i): return (2 * i.close - i.high - i.low) / i.open
@factor("VWAP0", "Alpha158", "$vwap/$close", "均价相对收盘价")
def _vwap0(i): return i.vwap / i.close


def _a158_rolling(d: int):
    c, h, l, v = "close", "high", "low", "volume"  # noqa: E741
    specs = {
        f"ROC{d}": (f"Ref($close,{d})/$close", "过去收益（倒数形式）", lambda i: delay(i.close, d) / i.close),
        f"MA{d}": (f"Mean($close,{d})/$close", "均线相对价格", lambda i: ts_mean(i.close, d) / i.close),
        f"STD{d}": (f"Std($close,{d})/$close", "价格波动", lambda i: ts_std(i.close, d) / i.close),
        f"BETA{d}": (f"Slope($close,{d})/$close", "价格线性趋势斜率", lambda i: ts_slope(i.close, d) / i.close),
        f"RSQR{d}": (f"Rsquare($close,{d})", "趋势线性度", lambda i: ts_rsquare(i.close, d)),
        f"RESI{d}": (f"Resi($close,{d})/$close", "偏离趋势线的残差", lambda i: ts_resid(i.close, d) / i.close),
        f"MAX{d}": (f"Max($high,{d})/$close", "区间最高价相对价格", lambda i: ts_max(i.high, d) / i.close),
        f"MIN{d}": (f"Min($low,{d})/$close", "区间最低价相对价格", lambda i: ts_min(i.low, d) / i.close),
        f"QTLU{d}": (f"Quantile($close,{d},0.8)/$close", "80% 分位价相对价格", lambda i: ts_quantile(i.close, d, 0.8) / i.close),
        f"QTLD{d}": (f"Quantile($close,{d},0.2)/$close", "20% 分位价相对价格", lambda i: ts_quantile(i.close, d, 0.2) / i.close),
        f"RANK{d}": (f"Rank($close,{d})", "当前价在区间中的分位", lambda i: ts_rank(i.close, d)),
        f"RSV{d}": (f"($close-Min($low,{d}))/(Max($high,{d})-Min($low,{d})+1e-12)", "随机指标 RSV",
                    lambda i: (i.close - ts_min(i.low, d)) / (ts_max(i.high, d) - ts_min(i.low, d) + EPS)),
        f"IMAX{d}": (f"IdxMax($high,{d})/{d}", "最高点出现的位置", lambda i: ts_argmax(i.high, d) / d),
        f"IMIN{d}": (f"IdxMin($low,{d})/{d}", "最低点出现的位置", lambda i: ts_argmin(i.low, d) / d),
        f"IMXD{d}": (f"(IdxMax($high,{d})-IdxMin($low,{d}))/{d}", "高低点先后", lambda i: (ts_argmax(i.high, d) - ts_argmin(i.low, d)) / d),
        f"CORR{d}": (f"Corr($close,Log($volume+1),{d})", "价量相关", lambda i: ts_corr(i.close, np.log(i.volume + 1), d)),
        f"CORD{d}": (f"Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),{d})", "价量变化相关",
                     lambda i: ts_corr(i.close / delay(i.close, 1), np.log(i.volume / delay(i.volume, 1) + 1), d)),
        f"CNTP{d}": (f"Mean($close>Ref($close,1),{d})", "上涨天数占比", lambda i: ts_mean((i.close > delay(i.close, 1)).astype(float), d)),
        f"CNTD{d}": (f"Mean($close>Ref($close,1),{d})-Mean($close<Ref($close,1),{d})", "涨跌天数差",
                     lambda i: ts_mean((i.close > delay(i.close, 1)).astype(float) - (i.close < delay(i.close, 1)).astype(float), d)),
        f"SUMP{d}": (f"Sum(Greater($close-Ref($close,1),0),{d})/(Sum(Abs($close-Ref($close,1)),{d})+1e-12)", "RSI 型上涨占比",
                     lambda i: ts_sum(np.fmax(delta(i.close, 1), 0), d) / (ts_sum(np.abs(delta(i.close, 1)), d) + EPS)),
        f"VMA{d}": (f"Mean($volume,{d})/($volume+1e-12)", "均量相对当日量", lambda i: ts_mean(i.volume, d) / (i.volume + EPS)),
        f"VSTD{d}": (f"Std($volume,{d})/($volume+1e-12)", "成交量波动", lambda i: ts_std(i.volume, d) / (i.volume + EPS)),
        f"WVMA{d}": (f"Std(Abs($close/Ref($close,1)-1)*$volume,{d})/(Mean(Abs($close/Ref($close,1)-1)*$volume,{d})+1e-12)", "量加权波动的变异系数",
                     lambda i: ts_std(np.abs(i.close / delay(i.close, 1) - 1) * i.volume, d) / (ts_mean(np.abs(i.close / delay(i.close, 1) - 1) * i.volume, d) + EPS)),
        f"VSUMD{d}": (f"(Sum(Greater($volume-Ref($volume,1),0),{d})-Sum(Greater(Ref($volume,1)-$volume,0),{d}))/(Sum(Abs($volume-Ref($volume,1)),{d})+1e-12)", "放量 / 缩量占比差",
                      lambda i: (ts_sum(np.fmax(delta(i.volume, 1), 0), d) - ts_sum(np.fmax(-delta(i.volume, 1), 0), d)) / (ts_sum(np.abs(delta(i.volume, 1)), d) + EPS)),
    }
    for name, (formula, desc, fn) in specs.items():
        CATALOG[name] = Factor(name, "Alpha158", formula, fn, desc)


for _d in (5, 20, 60):
    _a158_rolling(_d)
for _n in [k for k in list(CATALOG) if k.endswith("5") and k[:-1] in ("RSQR", "RESI", "QTLU", "QTLD", "IMXD", "CORD", "WVMA", "VSUMD", "VSTD")]:
    CATALOG.pop(_n)   # keep the 5-day window only where it is informative


# ---- WorldQuant Alpha101 (Kakushadze 2016) ------------------------------------------------
@factor("WQ001", "Alpha101", "rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2),5))-0.5")
def _wq1(i): return rank(ts_argmax(signed_power(np.where(i.ret < 0, ts_std(i.ret, 20), i.close / ts_mean(i.close, 20)), 2), 5)) - 0.5
@factor("WQ002", "Alpha101", "-1*correlation(rank(delta(log(volume),2)),rank((close-open)/open),6)", "量变与日内收益的相关")
def _wq2(i): return -ts_corr(rank(delta(np.log(i.volume), 2)), rank(i.raw_close / i.raw_open - 1), 6)
@factor("WQ003", "Alpha101", "-1*correlation(rank(open),rank(volume),10)")
def _wq3(i): return -ts_corr(rank(i.raw_open), rank(i.volume), 10)
@factor("WQ004", "Alpha101", "-1*Ts_Rank(rank(low),9)")
def _wq4(i): return -ts_rank(rank(i.raw_low), 9)
@factor("WQ006", "Alpha101", "-1*correlation(open,volume,10)", "开盘价与成交量相关")
def _wq6(i): return -ts_corr(i.open, i.volume, 10)
@factor("WQ007", "Alpha101", "(adv20<volume)?(-1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7))):-1")
def _wq7(i):
    d7 = delta(i.close, 7)
    return np.where(ts_mean(i.volume, 20) < i.volume, -ts_rank(np.abs(d7), 60) * np.sign(d7), -1.0)
@factor("WQ009", "Alpha101", "条件反转：基于 5 日内日收益的符号（delta(close,1) 以收益率代替）", modified=True)
def _wq9(i):
    r = i.ret
    return np.where(ts_min(r, 5) > 0, r, np.where(ts_max(r, 5) < 0, r, -r))
@factor("WQ013", "Alpha101", "-1*rank(covariance(rank(close),rank(volume),5))")
def _wq13(i): return -rank(ts_cov(rank(i.raw_close), rank(i.volume), 5))
@factor("WQ014", "Alpha101", "(-1*rank(delta(returns,3)))*correlation(open,volume,10)")
def _wq14(i): return -rank(delta(i.ret, 3)) * ts_corr(i.open, i.volume, 10)
@factor("WQ015", "Alpha101", "-1*sum(rank(correlation(rank(high),rank(volume),3)),3)")
def _wq15(i): return -ts_sum(rank(ts_corr(rank(i.raw_high), rank(i.volume), 3)), 3)
@factor("WQ016", "Alpha101", "-1*rank(covariance(rank(high),rank(volume),5))")
def _wq16(i): return -rank(ts_cov(rank(i.raw_high), rank(i.volume), 5))
@factor("WQ019", "Alpha101", "(-1*sign((close-delay(close,7))+delta(close,7)))*(1+rank(1+sum(returns,250)))")
def _wq19(i): return -np.sign(2 * delta(i.close, 7)) * (1 + rank(1 + ts_sum(i.ret, 250)))
@factor("WQ020", "Alpha101", "-rank(open/delay(high,1)-1)*rank(open/delay(close,1)-1)*rank(open/delay(low,1)-1)（相对价格版）", modified=True)
def _wq20(i): return -rank(i.open / delay(i.high, 1) - 1) * rank(i.open / delay(i.close, 1) - 1) * rank(i.open / delay(i.low, 1) - 1)
@factor("WQ022", "Alpha101", "-1*delta(correlation(high,volume,5),5)*rank(stddev(close,20)/close)（相对波动版）", modified=True)
def _wq22(i): return -delta(ts_corr(i.high, i.volume, 5), 5) * rank(ts_std(i.close, 20) / i.close)
@factor("WQ026", "Alpha101", "-1*ts_max(correlation(ts_rank(volume,5),ts_rank(high,5),5),3)")
def _wq26(i): return -ts_max(ts_corr(ts_rank(i.volume, 5), ts_rank(i.high, 5), 5), 3)
@factor("WQ033", "Alpha101", "rank(-1*(1-open/close))")
def _wq33(i): return rank(i.raw_open / i.raw_close - 1)
@factor("WQ034", "Alpha101", "rank((1-rank(stddev(returns,2)/stddev(returns,5)))+(1-rank(returns)))（delta(close,1) 以收益率代替）", modified=True)
def _wq34(i): return rank((1 - rank(ts_std(i.ret, 2) / ts_std(i.ret, 5))) + (1 - rank(i.ret)))
@factor("WQ038", "Alpha101", "(-1*rank(Ts_Rank(close,10)))*rank(close/open)")
def _wq38(i): return -rank(ts_rank(i.close, 10)) * rank(i.raw_close / i.raw_open)
@factor("WQ040", "Alpha101", "(-1*rank(stddev(high,10)/high))*correlation(high,volume,10)（相对波动版）", modified=True)
def _wq40(i): return -rank(ts_std(i.high, 10) / i.high) * ts_corr(i.high, i.volume, 10)
@factor("WQ041", "Alpha101", "((high*low)^0.5)/vwap-1（相对价格版）", modified=True)
def _wq41(i): return np.sqrt(i.raw_high * i.raw_low) / (i.vwap * i.raw_close / i.close) - 1
@factor("WQ042", "Alpha101", "rank(vwap-close)/rank(vwap+close)（同日价格水平，用未复权价）")
def _wq42(i):
    vwap_raw = i.vwap * i.raw_close / i.close
    return rank(vwap_raw - i.raw_close) / rank(vwap_raw + i.raw_close)
@factor("WQ043", "Alpha101", "ts_rank(volume/adv20,20)*ts_rank(-1*delta(close,7),8)")
def _wq43(i): return ts_rank(i.volume / ts_mean(i.volume, 20), 20) * ts_rank(-delta(i.close, 7), 8)
@factor("WQ044", "Alpha101", "-1*correlation(high,rank(volume),5)")
def _wq44(i): return -ts_corr(i.high, rank(i.volume), 5)
@factor("WQ053", "Alpha101", "-1*delta(((close-low)-(high-close))/(close-low),9)")
def _wq53(i): return -delta(((i.close - i.low) - (i.high - i.close)) / (i.close - i.low + EPS), 9)
@factor("WQ054", "Alpha101", "(-1*((low-close)*(open^5)))/((low-high)*(close^5))")
def _wq54(i): return -((i.raw_low - i.raw_close) * i.raw_open ** 5) / ((i.raw_low - i.raw_high - EPS) * i.raw_close ** 5)
@factor("WQ055", "Alpha101", "-1*correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))),rank(volume),6)")
def _wq55(i): return -ts_corr(rank((i.close - ts_min(i.low, 12)) / (ts_max(i.high, 12) - ts_min(i.low, 12) + EPS)), rank(i.volume), 6)
@factor("WQ101", "Alpha101", "(close-open)/((high-low)+.001)")
def _wq101(i): return (i.raw_close - i.raw_open) / (i.raw_high - i.raw_low + 0.001)


# ---- Guotai Junan 191 (国泰君安《基于短周期价量特征的多因子选股体系》) ----------------------
@factor("GTJA002", "GTJA191", "-1*DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),1)")
def _g2(i): return -delta(((i.close - i.low) - (i.high - i.close)) / (i.high - i.low + EPS), 1)
@factor("GTJA020", "GTJA191", "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100")
def _g20(i): return (i.close / delay(i.close, 6) - 1) * 100
@factor("GTJA031", "GTJA191", "(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100")
def _g31(i): return (i.close / ts_mean(i.close, 12) - 1) * 100
@factor("GTJA046", "GTJA191", "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)")
def _g46(i): return (ts_mean(i.close, 3) + ts_mean(i.close, 6) + ts_mean(i.close, 12) + ts_mean(i.close, 24)) / (4 * i.close)
@factor("GTJA053", "GTJA191", "COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100")
def _g53(i): return ts_sum((i.close > delay(i.close, 1)).astype(float), 12) / 12 * 100
@factor("GTJA066", "GTJA191", "(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100")
def _g66(i): return (i.close / ts_mean(i.close, 6) - 1) * 100
@factor("GTJA071", "GTJA191", "(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100")
def _g71(i): return (i.close / ts_mean(i.close, 24) - 1) * 100
@factor("GTJA080", "GTJA191", "(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100")
def _g80(i): return (i.volume / delay(i.volume, 5) - 1) * 100
@factor("GTJA088", "GTJA191", "(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100")
def _g88(i): return (i.close / delay(i.close, 20) - 1) * 100
@factor("GTJA095", "GTJA191", "STD(AMOUNT,20)")
def _g95(i): return ts_std(i.amount, 20)
@factor("GTJA100", "GTJA191", "STD(VOLUME,20)")
def _g100(i): return ts_std(i.volume, 20)
@factor("GTJA109", "GTJA191", "SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)")
def _g109(i):
    s = sma(i.high - i.low, 10, 2)
    return s / sma(s, 10, 2)
@factor("GTJA161", "GTJA191", "MEAN(MAX(MAX(HIGH-LOW,ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)/CLOSE（ATR 相对价格版）", modified=True)
def _g161(i):
    pc = delay(i.close, 1)
    tr = np.fmax(np.fmax(i.high - i.low, np.abs(pc - i.high)), np.abs(pc - i.low))
    return ts_mean(tr, 12) / i.close
@factor("GTJA189", "GTJA191", "MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)/CLOSE（相对价格版）", modified=True)
def _g189(i): return ts_mean(np.abs(i.close - ts_mean(i.close, 6)), 6) / i.close


def compute(inputs: Inputs, names: list[str] | None = None, rows: np.ndarray | None = None,
            on_progress: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:
    """Evaluate factors; ``rows`` (indices of dates to keep, e.g. month ends) trims the output."""
    out = {}
    with np.errstate(all="ignore"):
        for n in names or list(CATALOG):
            v = np.array(CATALOG[n].fn(inputs), dtype=float)   # copy: some operators return read-only views
            v[~np.isfinite(v)] = np.nan
            out[n] = v[rows] if rows is not None else v
            if on_progress:
                on_progress(n)
    return out
