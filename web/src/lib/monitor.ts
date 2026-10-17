// Monitoring & push (F10). The scheduled rules run in the local API service; the browser syncs the
// user's lists to it (holdings, watchlist, candidate pool, 组合构建 settings) whenever they change.
import { useEffect } from "react";
import { usePortfolioSettings } from "./portfolio";
import { useLists } from "./securities";

export interface RuleCfg {
  name: string;
  when: string;
  enabled: boolean;
  params: Record<string, number | string | boolean>;
}

export interface ChannelCfg {
  name: string;
  enabled: boolean;
  configured: boolean;
  url_masked: string;
  has_secret: boolean;
  from_env: boolean;
}

export interface MonitorConfig {
  rules: Record<string, RuleCfg>;
  channels: Record<string, ChannelCfg>;
  schedule: { enabled: boolean; intraday_minutes: number; eod_time: string };
  last_run: string | null;
  eod_done: string | null;
  client_synced: string | null;
  session: { code: string; label: string };
}

export interface AlertEvent {
  id: string;
  at: string;
  rule: string;
  level: "info" | "warn" | "critical";
  title: string;
  body: string;
  pushed: Record<string, string>;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api/monitor/${path}`, { headers: { "Content-Type": "application/json" }, ...init });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
  return d as T;
}

export const monitorApi = {
  config: () => call<MonitorConfig>("config"),
  update: (patch: unknown) => call<MonitorConfig>("config", { method: "PUT", body: JSON.stringify(patch) }),
  events: (limit = 50) => call<{ events: AlertEvent[] }>(`events?limit=${limit}`),
  clear: () => call<{ ok: boolean }>("events", { method: "DELETE" }),
  run: (body: { rules?: string[]; push?: boolean; force?: boolean }) =>
    call<{ events: AlertEvent[]; last_run: string }>("run", { method: "POST", body: JSON.stringify(body) }),
  test: (channel?: string) => call<{ results: Record<string, string> }>("test", { method: "POST", body: JSON.stringify({ channel }) }),
};

function snapshot() {
  const l = useLists.getState();
  const p = usePortfolioSettings.getState();
  return {
    holdings: l.holdings,
    watch: l.watch,
    pool: p.pool,
    portfolio_settings: { model: p.model, max_weight: p.cap / 100, industry_cap: p.indCap / 100, use_pool: p.usePool && p.pool.length > 0 },
  };
}

/** Keep the server's copy of the user's lists current (debounced). Mounted once in the Shell. */
export function useMonitorSync() {
  useEffect(() => {
    let timer: number | undefined;
    let last = "";
    const push = () => {
      const s = JSON.stringify(snapshot());
      if (s === last) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        fetch("/api/monitor/state", { method: "POST", headers: { "Content-Type": "application/json" }, body: s })
          .then((r) => r.ok && (last = s))
          .catch(() => undefined); // API offline: retried on the next change
      }, 800);
    };
    push();
    const a = useLists.subscribe(push);
    const b = usePortfolioSettings.subscribe(push);
    return () => {
      a();
      b();
      window.clearTimeout(timer);
    };
  }, []);
}
