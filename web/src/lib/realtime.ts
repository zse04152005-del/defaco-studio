// Real-time quotes: one shared WebSocket (auto-reconnect) with REST polling as fallback.
// Display layer only — portfolio maths always uses official closes.
import { useEffect, useState } from "react";
import { create } from "zustand";

export interface Quote {
  code: string;
  name: string;
  price: number | null;
  prev_close: number | null;
  open: number | null;
  high: number | null;
  low: number | null;
  change: number | null;
  pct: number;
  volume: number;
  amount: number;
  turnover?: number;
  pe?: number | null;
  pb?: number | null;
  float_mcap?: number;
  limit_up?: number | null;
  limit_down?: number | null;
  at_limit_up?: boolean; // price at the limit-up price: buying is effectively impossible
  at_limit_down?: boolean; // at limit-down: selling is effectively impossible
  no_trade?: boolean; // no open price and no volume today (suspended once the session has opened)
  bids: [number | null, number][];
  asks: [number | null, number][];
  time: string;
  source: string;
}

export interface Session {
  code: "trading" | "auction" | "break" | "pre" | "closed";
  label: string;
}

interface RTState {
  quotes: Record<string, Quote>;
  session: Session | null;
  connected: boolean;
  mode: "ws" | "poll" | "offline";
  updatedAt: number | null;
  wanted: Map<string, number>;
}

export const useRT = create<RTState>(() => ({
  quotes: {},
  session: null,
  connected: false,
  mode: "offline",
  updatedAt: null,
  wanted: new Map(),
}));

let ws: WebSocket | null = null;
let pollTimer: number | null = null;
let retry = 0;

function apply(msg: { session?: Session; quotes?: Record<string, Quote> }) {
  useRT.setState((s) => ({
    quotes: { ...s.quotes, ...(msg.quotes ?? {}) },
    session: msg.session ?? s.session,
    updatedAt: Date.now(),
  }));
}

function wantedCodes() {
  return [...useRT.getState().wanted.keys()];
}

async function poll() {
  try {
    const codes = wantedCodes();
    const r = await fetch(`/api/realtime/quotes?codes=${encodeURIComponent(["000300.SH", "000905.SH", ...codes].join(","))}`);
    if (!r.ok) throw new Error(String(r.status));
    apply(await r.json());
    useRT.setState({ mode: "poll", connected: true });
  } catch {
    useRT.setState({ mode: "offline", connected: false });
  }
}

function startPolling() {
  if (pollTimer != null) return;
  poll();
  pollTimer = window.setInterval(poll, 5000);
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/quotes`);
  ws.onopen = () => {
    retry = 0;
    if (pollTimer != null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    useRT.setState({ connected: true, mode: "ws" });
    ws!.send(JSON.stringify({ subscribe: wantedCodes() }));
  };
  ws.onmessage = (e) => apply(JSON.parse(e.data));
  ws.onclose = () => {
    useRT.setState({ connected: false });
    startPolling(); // keep data flowing while we try to reconnect
    retry = Math.min(retry + 1, 6);
    setTimeout(connect, 1000 * 2 ** retry);
  };
  ws.onerror = () => ws?.close();
}

/** Subscribe to live quotes for ``codes`` while the calling component is mounted. */
export function useQuotes(codes: string[]): Record<string, Quote> {
  const key = codes.join(",");
  useEffect(() => {
    const wanted = useRT.getState().wanted;
    codes.forEach((c) => wanted.set(c, (wanted.get(c) ?? 0) + 1));
    connect();
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ subscribe: codes }));
    // instant first paint via REST (the socket may only push on its next tick)
    const missing = codes.filter((c) => !useRT.getState().quotes[c]);
    if (missing.length)
      fetch(`/api/realtime/quotes?codes=${encodeURIComponent(missing.join(","))}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => d && apply(d))
        .catch(() => undefined);
    return () => {
      codes.forEach((c) => {
        const n = (wanted.get(c) ?? 1) - 1;
        if (n <= 0) wanted.delete(c);
        else wanted.set(c, n);
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return useRT((s) => s.quotes);
}

export interface MinuteBar {
  t: string | null;
  p: number;
  v: number;
  avg: number;
}

/** Today's 1-minute series, refreshed every 20 s while trading. */
export function useMinutes(code: string | null) {
  const [data, setData] = useState<{ source: string; bars: MinuteBar[] } | null>(null);
  const session = useRT((s) => s.session?.code);
  useEffect(() => {
    if (!code) return;
    let alive = true;
    const load = () =>
      fetch(`/api/realtime/minute/${code}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => alive && d && setData(d))
        .catch(() => alive && setData(null));
    load();
    const id = session === "trading" || session === "auction" ? window.setInterval(load, 20000) : null;
    return () => {
      alive = false;
      if (id) clearInterval(id);
    };
  }, [code, session]);
  return data;
}

export type KBar = [string, number, number, number, number, number];

/** Daily K-line up to today from the API ({bars, source}); null when the API is unavailable. */
export function useLiveKline(code: string | null) {
  const [k, setK] = useState<{ bars: KBar[]; source: string } | null>(null);
  useEffect(() => {
    if (!code) return;
    let alive = true;
    setK(null);
    fetch(`/api/kline/${code}?years=3`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => alive && setK(d?.bars?.length ? { bars: d.bars, source: d.source } : null))
      .catch(() => alive && setK(null));
    return () => {
      alive = false;
    };
  }, [code]);
  return k;
}
