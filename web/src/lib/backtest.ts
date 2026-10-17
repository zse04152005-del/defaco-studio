// Live backtests (F8): POST /api/backtest/run -> poll /api/backtest/jobs/{id}. The last result is
// kept in memory (and its job id in localStorage) so it survives page switches.
import { create } from "zustand";

export interface StrategyForm {
  kind: "opt" | "ew" | "vw";
  model: string;
  objective: string;
  max_weight: number;
  industry_cap: number | null;
  risk_aversion: number;
}

export interface ExecutionForm {
  timing: "next_open" | "close";
  aum: number;
  commission_bps: number;
  min_commission: number;
  impact_y: number;
  participation_cap: number | null;
}

export interface BacktestForm {
  strategies: StrategyForm[];
  include_ew: boolean;
  start: string;
  end: string;
  freq: "M" | "W" | "Q";
  top_n: number;
  use_pool: boolean;
  execution: ExecutionForm;
}

export interface Metrics {
  ann_return: number;
  ann_vol: number;
  sharpe: number;
  sharpe_ci95?: [number, number];
  psr_0?: number;
  dsr?: number;
  dsr_trials?: number;
  sr0_ann?: number;
  sortino: number;
  calmar?: number;
  max_drawdown: number;
  dd_peak?: string;
  dd_trough?: string;
  dd_recovered?: string | null;
  longest_underwater_days?: number;
  var95_daily?: number;
  cvar95_daily?: number;
  monthly_win_rate?: number;
  excess_ann_return?: number;
  tracking_error?: number;
  information_ratio?: number;
  beta?: number;
  alpha_ann?: number;
  annual_turnover?: number;
  cost_drag_ann?: number;
  cost_breakdown_ann?: Record<string, number>;
  avg_holdings?: number;
  avg_cash?: number;
  blocked_orders?: Record<string, number>;
  unfilled_avg?: number;
}

export interface StrategyResult {
  spec?: StrategyForm & { label?: string | null };
  nav: [string, number][];
  drawdown: [string, number][];
  monthly: Record<string, number>;
  turnover?: [string, number][];
  last_weights?: { code: string; name: string; w: number }[];
  metrics: Metrics;
}

export interface BacktestResult {
  start: string;
  end: string;
  rebalances: number;
  freq: string;
  universe: string;
  dataset: string;
  execution: ExecutionForm;
  strategies: Record<string, StrategyResult>;
  benchmark?: { name: string; nav: [string, number][] };
  offline?: boolean;
}

interface JobStatus {
  job: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  error: string | null;
  elapsed: number;
  result?: BacktestResult;
}

export const DEFAULT_FORM: BacktestForm = {
  strategies: [
    { kind: "opt", model: "pca_factor", objective: "minvar", max_weight: 0.1, industry_cap: 0.35, risk_aversion: 5 },
    { kind: "opt", model: "lw_nonlinear", objective: "minvar", max_weight: 0.1, industry_cap: 0.35, risk_aversion: 5 },
  ],
  include_ew: true,
  start: "2016-01-01",
  end: "2025-12-31",
  freq: "M",
  top_n: 100,
  use_pool: false,
  execution: { timing: "next_open", aum: 1e8, commission_bps: 2.5, min_commission: 5, impact_y: 0.5, participation_cap: 0.1 },
};

const JOB_KEY = "defaco.backtest.job";
const FORM_KEY = "defaco.backtest.form";

function loadForm(): BacktestForm {
  try {
    const f = JSON.parse(localStorage.getItem(FORM_KEY) || "null");
    return f ? { ...DEFAULT_FORM, ...f, execution: { ...DEFAULT_FORM.execution, ...f.execution } } : DEFAULT_FORM;
  } catch {
    return DEFAULT_FORM;
  }
}

interface BacktestState {
  form: BacktestForm;
  job: string | null;
  status: JobStatus["status"] | "idle";
  progress: number;
  message: string;
  error: string | null;
  elapsed: number;
  result: BacktestResult | null;
  setForm: (f: Partial<BacktestForm>) => void;
  run: (pool: string[]) => Promise<void>;
  resume: () => Promise<void>;
}

let timer: number | undefined;

export const useBacktest = create<BacktestState>((set, get) => {
  const poll = async (id: string) => {
    window.clearTimeout(timer);
    try {
      const r = await fetch(`/api/backtest/jobs/${id}`);
      if (r.status === 404) {
        set({ status: "idle", job: null });
        return;
      }
      const d = (await r.json()) as JobStatus;
      set({ status: d.status, progress: d.progress, message: d.message, error: d.error, elapsed: d.elapsed, ...(d.result ? { result: d.result } : {}) });
      if (d.status === "queued" || d.status === "running") timer = window.setTimeout(() => poll(id), 1200);
    } catch {
      set({ status: "error", error: "回测服务不可用（请确认 API 已启动：scripts/dev.sh）" });
    }
  };
  return {
    form: loadForm(),
    job: (() => {
      try {
        return localStorage.getItem(JOB_KEY);
      } catch {
        return null;
      }
    })(),
    status: "idle",
    progress: 0,
    message: "",
    error: null,
    elapsed: 0,
    result: null,
    setForm: (f) => {
      const form = { ...get().form, ...f };
      try {
        localStorage.setItem(FORM_KEY, JSON.stringify(form));
      } catch {
        /* ignore */
      }
      set({ form });
    },
    run: async (pool) => {
      const f = get().form;
      const body = {
        strategies: f.strategies.map((s) => ({ ...s, industry_cap: s.industry_cap })),
        include_ew: f.include_ew,
        start: f.start,
        end: f.end,
        freq: f.freq,
        top_n: f.top_n,
        universe: f.use_pool && pool.length ? pool : null,
        execution: f.execution,
      };
      set({ status: "queued", progress: 0, message: "提交任务…", error: null });
      try {
        const r = await fetch("/api/backtest/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        const d = await r.json();
        if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail));
        try {
          localStorage.setItem(JOB_KEY, d.job);
        } catch {
          /* ignore */
        }
        set({ job: d.job });
        poll(d.job);
      } catch (e) {
        set({ status: "error", error: (e as Error).message.includes("fetch") ? "回测服务不可用（请确认 API 已启动）" : (e as Error).message });
      }
    },
    resume: async () => {
      const id = get().job;
      if (id && !get().result) await poll(id);
    },
  };
});

/** Adapt the precomputed backtest.json (offline fallback) to the live result shape. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function fromStatic(d: any): BacktestResult {
  const strategies: Record<string, StrategyResult> = {};
  for (const v of Object.values(d.results) as any[]) { // eslint-disable-line @typescript-eslint/no-explicit-any
    strategies[v.label] = { nav: v.nav, drawdown: v.drawdown, monthly: v.monthly, metrics: { ...v.metrics, annual_turnover: v.metrics.turnover * 12 } };
  }
  return {
    start: d.start, end: d.end, rebalances: 120, freq: "月末", universe: "沪深300 历史成分股流通市值前 100", dataset: d.meta?.dataset ?? "data",
    execution: { ...DEFAULT_FORM.execution, timing: "close", impact_y: 0, participation_cap: null }, strategies, offline: true,
  };
}
