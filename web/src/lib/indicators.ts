// Technical indicators and simple candlestick patterns, computed client-side for the demo.
// (The backend will later serve them from quantkit / myhhub-style pattern module.)
import type { Bar } from "./data";

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount: number;
}

export const toCandles = (bars: Bar[]): Candle[] =>
  bars.map(([time, open, high, low, close, volume, amount]) => ({ time, open, high, low, close, volume, amount }));

/** Aggregate daily candles to weekly ("W") or monthly ("M") candles. */
export function resample(c: Candle[], freq: "D" | "W" | "M"): Candle[] {
  if (freq === "D") return c;
  const key = (t: string) => {
    if (freq === "M") return t.slice(0, 7);
    const d = new Date(t + "T00:00:00Z");
    const day = (d.getUTCDay() + 6) % 7; // Monday = 0
    d.setUTCDate(d.getUTCDate() - day);
    return d.toISOString().slice(0, 10);
  };
  const out: Candle[] = [];
  for (const x of c) {
    const last = out[out.length - 1];
    if (last && key(last.time) === key(x.time)) {
      last.high = Math.max(last.high, x.high);
      last.low = Math.min(last.low, x.low);
      last.close = x.close;
      last.volume += x.volume;
      last.amount += x.amount;
      last.time = x.time; // stamp with the period's last trading day
    } else {
      out.push({ ...x });
    }
  }
  return out;
}

export function sma(v: number[], n: number): (number | null)[] {
  const out: (number | null)[] = [];
  let s = 0;
  v.forEach((x, i) => {
    s += x;
    if (i >= n) s -= v[i - n];
    out.push(i >= n - 1 ? s / n : null);
  });
  return out;
}

export function ema(v: number[], n: number): number[] {
  const k = 2 / (n + 1);
  const out: number[] = [];
  v.forEach((x, i) => out.push(i === 0 ? x : x * k + out[i - 1] * (1 - k)));
  return out;
}

export function macd(close: number[]) {
  const e26 = ema(close, 26);
  const dif = ema(close, 12).map((x, i) => x - e26[i]);
  const dea = ema(dif, 9);
  return { dif, dea, hist: dif.map((x, i) => 2 * (x - dea[i])) };
}

export function rsi(close: number[], n = 14): (number | null)[] {
  const out: (number | null)[] = [null];
  let up = 0, dn = 0;
  for (let i = 1; i < close.length; i++) {
    const d = close[i] - close[i - 1];
    const u = Math.max(d, 0), w = Math.max(-d, 0);
    if (i <= n) {
      up += u / n;
      dn += w / n;
      out.push(i === n ? 100 - 100 / (1 + up / (dn || 1e-12)) : null);
    } else {
      up = (up * (n - 1) + u) / n;
      dn = (dn * (n - 1) + w) / n;
      out.push(100 - 100 / (1 + up / (dn || 1e-12)));
    }
  }
  return out;
}

export function kdj(c: Candle[], n = 9) {
  const K: number[] = [], D: number[] = [], J: number[] = [];
  c.forEach((x, i) => {
    const w = c.slice(Math.max(0, i - n + 1), i + 1);
    const hi = Math.max(...w.map((y) => y.high)), lo = Math.min(...w.map((y) => y.low));
    const rsv = hi === lo ? 50 : ((x.close - lo) / (hi - lo)) * 100;
    const k = (2 / 3) * (K[i - 1] ?? 50) + rsv / 3;
    const d = (2 / 3) * (D[i - 1] ?? 50) + k / 3;
    K.push(k); D.push(d); J.push(3 * k - 2 * d);
  });
  return { K, D, J };
}

export interface PatternHit {
  time: string;
  kind: "bull" | "bear";
  name: string;
  desc: string;
}

/** A few classic single/two-candle patterns (illustrative; thresholds are conventional). */
export function detectPatterns(c: Candle[]): PatternHit[] {
  const hits: PatternHit[] = [];
  const ma20 = sma(c.map((x) => x.close), 20);
  for (let i = 21; i < c.length; i++) {
    const x = c[i], p = c[i - 1];
    const body = Math.abs(x.close - x.open), range = x.high - x.low || 1e-9;
    const lower = Math.min(x.open, x.close) - x.low, upper = x.high - Math.max(x.open, x.close);
    const downtrend = (ma20[i] ?? x.close) > x.close;
    if (downtrend && lower > 2 * body && upper < body && body / range < 0.35)
      hits.push({ time: x.time, kind: "bull", name: "锤子线", desc: "下跌中出现长下影线，下方承接较强" });
    else {
      // engulfing: both bodies must be meaningful (> 0.8% / 0.4% of price) to avoid noise
      const pb = Math.abs(p.close - p.open);
      const big = body > 0.008 * x.close && pb > 0.004 * p.close && body > 1.3 * pb;
      if (big && p.close < p.open && x.close > x.open && x.open <= p.close && x.close >= p.open && downtrend)
        hits.push({ time: x.time, kind: "bull", name: "看涨吞没", desc: "下跌中阳线实体完全包住前一根阴线" });
      else if (big && p.close > p.open && x.close < x.open && x.open >= p.close && x.close <= p.open && !downtrend)
        hits.push({ time: x.time, kind: "bear", name: "看跌吞没", desc: "上涨中阴线实体完全包住前一根阳线" });
    }
  }
  return hits;
}
