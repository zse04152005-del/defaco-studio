// Data access. Today it reads static JSON exported by scripts/export_demo_data.py;
// later the same hooks will call the FastAPI service (06 design doc, section 4.5).
import { useEffect, useState } from "react";

export interface Meta {
  dataset: string;
  survivorship_biased: boolean;
  note: string;
}

export interface Security {
  code: string;
  name: string;
  industry: string;
  mcap: number; // 亿元 (float market cap)
  close: number;
  ret_1m: number;
  board: "STAR" | "main";
}

export interface ScreenerRow {
  code: string;
  name: string;
  industry: string;
  mcap: number | null; // null for stocks no longer listed at asof
  ret_1m: number;
  [factor: string]: number | string | null;
}

export interface Screener {
  meta: Meta;
  asof: string;
  factors: string[];
  items: ScreenerRow[];
}

export type Bar = [string, number, number, number, number, number, number]; // t,o,h,l,c,v,amount

export interface StockDetail {
  meta: Meta;
  code: string;
  name: string;
  industry: string;
  mcap: number;
  bars: Bar[];
  profile: Record<string, number>;
}

export interface Overview {
  meta: Meta;
  asof: string;
  indices: Record<string, [string, number][]>;
  industries: { ind: string; ret: number; amount: number; n: number }[];
  breadth: { up: number; down: number; flat: number; hist: number[] };
  vol_regime: [string, number][];
  vol_threshold: number;
}

export interface Holding {
  code: string;
  name: string;
  industry: string;
  w: number;
  rc: number;
}

export interface ModelPortfolio {
  label: string;
  weights: Holding[];
  industry_weights: Record<string, number>;
  risk: { total_vol: number; factor_share?: number; industry_share?: number; idio_share?: number; exposures?: number[] };
  n_holdings: number;
  hhi: number;
  explain: Record<string, unknown>;
}

export interface Portfolio {
  meta: Meta;
  asof: string;
  universe_size: number;
  models: Record<string, ModelPortfolio>;
  corr: { codes: string[]; names: string[]; industries: string[]; matrix: number[][] };
}

export interface BacktestRun {
  label: string;
  nav: [string, number][];
  drawdown: [string, number][];
  monthly: Record<string, number>;
  metrics: Record<string, number>;
  industry_weights: Record<string, Record<string, number>>;
}

export interface Backtest {
  meta: Meta;
  start: string;
  end: string;
  costs: string;
  results: Record<string, BacktestRun>;
}

const cache = new Map<string, Promise<unknown>>();

export function fetchJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`/demo-data/${path}`).then((r) => {
        if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path) as Promise<T>;
}

export function useData<T>(path: string | null): { data: T | null; error: string | null; loading: boolean } {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: !!path,
  });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetchJson<T>(path)
      .then((d) => alive && setState({ data: d, error: null, loading: false }))
      .catch((e: Error) => alive && setState({ data: null, error: e.message, loading: false }));
    return () => {
      alive = false;
    };
  }, [path]);
  return state;
}
