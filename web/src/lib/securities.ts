// Full A-share securities list (securities_all.json, ~5,700 rows) + search, watchlist,
// recently viewed and the user's holdings. Lists are per-browser conveniences (localStorage).
import { useEffect, useState } from "react";
import { create } from "zustand";
import { fetchJson } from "./data";

export type Board = "Main" | "ChiNext" | "STAR" | "BSE" | "index";

export interface Sec {
  code: string;
  name: string;
  py: string[]; // pinyin initials, first = phrase-aware reading
  industry: string;
  board: Board;
  research: boolean; // in the research universe (historical CSI300/CSI500): factor data available
  mcap: number | null; // float market cap at the research date, 亿元
}

export const BOARD_LABEL: Record<Board, string> = { Main: "主板", ChiNext: "创业板", STAR: "科创板", BSE: "北交所", index: "指数" };

interface Raw {
  fields: string[];
  rows: (string | number | null)[][];
}

export interface SecTable {
  list: Sec[];
  byCode: Map<string, Sec>;
}

let cache: SecTable | null = null;
let inflight: Promise<SecTable> | null = null;

function load(): Promise<SecTable> {
  if (cache) return Promise.resolve(cache);
  inflight ??= fetchJson<Raw>("securities_all.json").then((d) => {
    const list = d.rows.map(([code, name, py, industry, board, research, mcap]) => ({
      code: String(code),
      name: String(name),
      py: String(py ?? "").split("|").filter(Boolean),
      industry: String(industry ?? ""),
      board: board as Board,
      research: research === 1,
      mcap: mcap == null ? null : Number(mcap),
    }));
    const table: SecTable = { list, byCode: new Map(list.map((s) => [s.code, s])) };
    cache = table;
    return table;
  });
  return inflight;
}

/** The securities table (null until loaded). */
export function useSecurities(): SecTable | null {
  const [s, setS] = useState<SecTable | null>(cache);
  useEffect(() => {
    if (!s) load().then(setS).catch(() => undefined);
  }, [s]);
  return s;
}

/**
 * Ranked search over code, name and pinyin initials ("gzmt", "600519", "茅台", "cjdl").
 * Exact > prefix > substring; ties prefer the research universe, then larger market cap.
 */
export function searchSecurities(list: Sec[], query: string, limit = 10): Sec[] {
  const q = query.trim().toLowerCase().replace(/\s+/g, "");
  if (!q) return [];
  const digits = /^\d+$/.test(q);
  const scored: [number, Sec][] = [];
  for (const s of list) {
    const code = s.code.toLowerCase();
    const num = code.slice(0, 6);
    const name = s.name.toLowerCase();
    let score = 0;
    if (code === q || num === q) score = 100;
    else if (num.startsWith(q)) score = 90 - (num.length - q.length);
    else if (name === q) score = 95;
    else if (name.startsWith(q)) score = 85;
    else if (!digits && s.py[0] === q) score = 88; // phrase-aware reading
    else if (!digits && s.py.some((p) => p === q)) score = 72; // alternative reading of a heteronym
    else if (!digits && s.py.some((p) => p.startsWith(q))) score = s.py[0].startsWith(q) ? 80 : 74;
    else if (name.includes(q)) score = 60;
    else if (digits && num.includes(q)) score = 50;
    else if (!digits && q.length >= 2 && s.py.some((p) => p.includes(q))) score = 40;
    if (score) scored.push([score + (s.research ? 2 : 0) + Math.min(2, (s.mcap ?? 0) / 10000), s]);
  }
  return scored.sort((a, b) => b[0] - a[0]).slice(0, limit).map((x) => x[1]);
}

// ----------------------------------------------------------------------------- personal lists
export interface Holding {
  code: string;
  shares: number;
}

const DEFAULT_HOLDINGS: Holding[] = [
  { code: "600519.SH", shares: 300 },
  { code: "601318.SH", shares: 6000 },
  { code: "300750.SZ", shares: 1200 },
  { code: "600036.SH", shares: 8000 },
  { code: "000333.SZ", shares: 3000 },
];

function read<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, v: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* storage unavailable (private mode): lists then live for the session only */
  }
}

interface ListsState {
  watch: string[];
  recent: string[];
  holdings: Holding[];
  cash: number;
  toggleWatch: (code: string) => void;
  visit: (code: string) => void;
  clearRecent: () => void;
  setHoldings: (h: Holding[]) => void;
  setCash: (c: number) => void;
}

export const useLists = create<ListsState>((set, get) => ({
  watch: read<string[]>("defaco.watch", ["600519.SH", "300750.SZ", "600900.SH"]),
  recent: read<string[]>("defaco.recent", []),
  holdings: read<Holding[]>("defaco.holdings", DEFAULT_HOLDINGS),
  cash: read<number>("defaco.cash", 200000),
  toggleWatch: (code) => {
    const w = get().watch.includes(code) ? get().watch.filter((c) => c !== code) : [code, ...get().watch];
    write("defaco.watch", w);
    set({ watch: w });
  },
  visit: (code) => {
    if (get().recent[0] === code) return;
    const r = [code, ...get().recent.filter((c) => c !== code)].slice(0, 30);
    write("defaco.recent", r);
    set({ recent: r });
  },
  clearRecent: () => {
    write("defaco.recent", []);
    set({ recent: [] });
  },
  setHoldings: (holdings) => {
    write("defaco.holdings", holdings);
    set({ holdings });
  },
  setCash: (cash) => {
    write("defaco.cash", cash);
    set({ cash });
  },
}));

// ----------------------------------------------------------------------------- paper-trading account
// An external paper-trading run may export one account to demo-data/paper_holdings.json (git-ignored;
// fields as in PaperAccount). When the file exists and its version is new, it replaces the holdings and cash
// above; edits made in the browser stay until the account trades again.
export interface PaperAccount {
  version: string;
  label: string;
  opened: string;
  last_trade: string;
  valued: string;
  capital: number;
  nav: number;
  cash: number;
  holdings: Holding[];
}

export const usePaper = create<{ paper: PaperAccount | null }>(() => ({ paper: null }));

export async function syncPaperAccount(force = false): Promise<PaperAccount | null> {
  try {
    const r = await fetch("/demo-data/paper_holdings.json", { cache: "no-store" });
    if (!r.ok) return null;
    const p = (await r.json()) as PaperAccount;        // the dev server answers a missing file with HTML: JSON fails
    if (!p?.holdings?.length) return null;
    usePaper.setState({ paper: p });
    if (force || read<string>("defaco.paper.version", "") !== p.version) {
      useLists.getState().setHoldings(p.holdings);
      useLists.getState().setCash(p.cash);
      write("defaco.paper.version", p.version);
    }
    return p;
  } catch {
    return null;
  }
}

if (typeof window !== "undefined") void syncPaperAccount();
