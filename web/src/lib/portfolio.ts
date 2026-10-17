// Live portfolio optimisation (POST /api/portfolio/optimize) shared by 组合构建 and 风险解析.
// Settings live in one store so both pages show the same portfolio; the Screener writes its
// candidate pool here. When the API is down the pages fall back to the precomputed
// portfolio.json (最小方差 · 单股上限 10%, default universe).
import { useEffect, useState } from "react";
import { create } from "zustand";
import { fetchJson, type ModelPortfolio, type Portfolio } from "./data";

export interface PortfolioSettings {
  model: string;
  objective: string;
  cap: number; // single-stock cap, percent
  indCap: number; // industry cap, percent
  gamma: number; // risk aversion for mean-variance
  pool: string[]; // candidate pool from the Screener
  usePool: boolean;
  set: (s: Partial<Omit<PortfolioSettings, "set">>) => void;
}

// the candidate pool persists in this browser (localStorage); older builds kept it per session
const POOL_KEY = "defaco.pool";
function loadPool(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(POOL_KEY) || sessionStorage.getItem(POOL_KEY) || "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

export const usePortfolioSettings = create<PortfolioSettings>((set) => ({
  model: "pca_factor",
  objective: "minvar",
  cap: 10,
  indCap: 35,
  gamma: 5,
  pool: loadPool(),
  usePool: loadPool().length > 0,
  set: (s) =>
    set(() => {
      if (s.pool) {
        try {
          localStorage.setItem(POOL_KEY, JSON.stringify(s.pool));
        } catch {
          /* storage may be unavailable */
        }
      }
      return s;
    }),
}));

export interface LivePortfolio extends ModelPortfolio {
  model: string;
  objective: string;
  asof: string;
  universe_size: number;
  universe_source: "default" | "custom";
  dropped: string[];
  notes?: string[];
  constraints: { max_weight: number; industry_cap: number | null };
  timing_ms: { fit: number; solve: number; total: number };
  corr: Portfolio["corr"];
  char_exposures?: { names: string[]; values: number[]; compare?: { label: string; values: number[] } };
}

export interface OptimizeParams {
  model: string;
  objective: string;
  max_weight: number;
  industry_cap: number | null;
  universe: string[] | null;
  risk_aversion?: number;
}

export type LiveSource = "live" | "fallback";

export interface LiveState {
  data: LivePortfolio | null;
  source: LiveSource | null;
  pending: boolean;
  error: string | null; // a 4xx from the optimiser (infeasible constraints, objective not implemented, ...)
}

class HttpError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const results = new Map<string, Promise<LivePortfolio>>();

// Results are cached per parameter set (moving a slider back is instant). Requests are not
// aborted, because a cached promise may be shared by both pages; stale answers are ignored.
export function optimize(p: OptimizeParams): Promise<LivePortfolio> {
  const key = JSON.stringify(p);
  const hit = results.get(key);
  if (hit) return hit;
  const req = fetch("/api/portfolio/optimize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: key,
  }).then(async (r) => {
    if (!r.ok) {
      const body = await r.json().catch(() => null);
      const detail = typeof body?.detail === "string" ? body.detail : `HTTP ${r.status}`;
      throw new HttpError(r.status, detail);
    }
    return r.json() as Promise<LivePortfolio>;
  });
  results.set(key, req);
  req.catch(() => results.delete(key));
  return req;
}

/** Precomputed result for the model, adapted to the live shape (default constraints only). */
async function fallback(model: string): Promise<LivePortfolio | null> {
  const P = await fetchJson<Portfolio>("portfolio.json");
  let m = P.models[model];
  if (!m && model in LIVE_ONLY_MODELS) {   // DeFaCo: optional local file (not part of the open-source data)
    m = (await fetchJson<{ models: Record<string, ModelPortfolio> }>("portfolio_defaco.json").catch(() => null))?.models[model];
  }
  if (!m) return null;
  return {
    ...m,
    model,
    objective: "minvar",
    asof: P.asof,
    universe_size: P.universe_size,
    universe_source: "default",
    dropped: [],
    constraints: { max_weight: 0.1, industry_cap: null },
    timing_ms: { fit: 0, solve: 0, total: 0 },
    corr: P.corr,
  };
}

/** Optimise with debounce; keeps showing the previous result while a new one is computed. */
export function useLivePortfolio(p: OptimizeParams | null, debounceMs = 300): LiveState {
  // `key` = the request the stored state answers; pending is derived instead of set in the effect
  const [state, setState] = useState<Omit<LiveState, "pending"> & { key: string | null }>({ data: null, source: null, error: null, key: null });
  const key = p ? JSON.stringify(p) : null;
  useEffect(() => {
    if (!key) return;
    const params = JSON.parse(key) as OptimizeParams;
    let stale = false;
    const t = setTimeout(() => {
      optimize(params)
        .then((data) => !stale && setState({ data, source: "live", error: null, key }))
        .catch(async (e: unknown) => {
          if (stale) return;
          // the optimiser answered but refused (infeasible caps, objective not implemented): keep the last result
          if (e instanceof HttpError && ((e.status >= 400 && e.status < 500) || e.status === 501)) {
            setState((s) => ({ ...s, error: e.message, key }));
            return;
          }
          // API down (network error, proxy 502/504, data not loaded): precomputed JSON
          const fb = await fallback(params.model).catch(() => null);
          if (!stale) setState({ data: fb, source: fb ? "fallback" : null, error: fb ? null : String(e), key });
        });
    }, debounceMs);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [key, debounceMs]);
  return { data: state.data, source: state.source, error: state.error, pending: !!key && state.key !== key };
}

/** Request parameters from the shared settings (null universe = server default). */
export function paramsFrom(s: Pick<PortfolioSettings, "objective" | "cap" | "indCap" | "pool" | "usePool" | "gamma">, model: string): OptimizeParams {
  return {
    model,
    objective: s.objective,
    max_weight: s.cap / 100,
    industry_cap: s.indCap / 100,
    universe: s.usePool && s.pool.length ? [...s.pool].sort() : null,
    ...(s.objective === "mv" ? { risk_aversion: s.gamma } : {}),
  };
}

/** Risk models served only by the live API (no precomputed fallback in portfolio.json), listed first. */
export const LIVE_ONLY_MODELS: Record<string, string> = { defaco: "DeFaCo（论文 M4）" };

let defacoCheck: Promise<boolean> | null = null;
/** Whether the trained DeFaCo parameters are installed (live API) or a local DeFaCo fallback file exists.
 * The open-source release does not ship them, so the option is hidden there. */
function checkDefaco(): Promise<boolean> {
  defacoCheck ??= fetch("/api/portfolio/options")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((o: { defaco?: boolean }) => !!o.defaco)
    .catch(() => fetchJson("portfolio_defaco.json").then(() => true, () => false));
  return defacoCheck;
}

export function useDefacoAvailable(): boolean {
  const [ok, setOk] = useState(false);
  useEffect(() => {
    let alive = true;
    checkDefaco().then((v) => alive && setOk(v));
    return () => { alive = false; };
  }, []);
  return ok;
}

export function modelOptions(models?: Record<string, { label: string }>, defaco = true): { value: string; label: string }[] {
  return [
    ...Object.entries(LIVE_ONLY_MODELS).filter(([k]) => k !== "defaco" || defaco).map(([value, label]) => ({ value, label })),
    ...Object.entries(models ?? {}).filter(([k]) => !(k in LIVE_ONLY_MODELS)).map(([value, v]) => ({ value, label: v.label })),
  ];
}
