import clsx from "clsx";
import { useMemo, useState } from "react";
import { Star, X } from "lucide-react";
import { tone } from "../lib/format";
import { useQuotes } from "../lib/realtime";
import { paramsFrom, useLivePortfolio, usePortfolioSettings } from "../lib/portfolio";
import { BOARD_LABEL, useLists, useSecurities, type Sec } from "../lib/securities";
import { Panel } from "./ui";

type Tab = "watch" | "pool" | "hold" | "port" | "recent" | "mcap";
const TABS: { key: Tab; label: string }[] = [
  { key: "watch", label: "自选" },
  { key: "pool", label: "候选" },
  { key: "hold", label: "持仓" },
  { key: "port", label: "组合" },
  { key: "recent", label: "最近" },
  { key: "mcap", label: "市值" },
];

function loadTab(): Tab {
  try {
    return (localStorage.getItem("defaco.stockTab") as Tab) || "watch";
  } catch {
    return "watch";
  }
}

interface Row {
  sec: Sec;
  extra?: string; // e.g. portfolio weight or shares
}

export function StockLists({ current, onPick }: { current: string; onPick: (c: string) => void }) {
  const secs = useSecurities();
  const { watch, recent, holdings, toggleWatch, clearRecent } = useLists();
  const settings = usePortfolioSettings();
  const [tab, setTabState] = useState<Tab>(loadTab);
  const [filter, setFilter] = useState("");
  const setTab = (t: Tab) => {
    setTabState(t);
    try {
      localStorage.setItem("defaco.stockTab", t);
    } catch {
      /* ignore */
    }
  };
  // the live portfolio is only requested while its tab is open
  const live = useLivePortfolio(tab === "port" ? paramsFrom(settings, settings.model) : null);

  const rows: Row[] = useMemo(() => {
    if (!secs) return [];
    const get = (c: string) => secs.byCode.get(c);
    let out: Row[] = [];
    if (tab === "watch") out = watch.map((c) => get(c)).filter(Boolean).map((sec) => ({ sec: sec! }));
    else if (tab === "pool") out = settings.pool.map((c) => get(c)).filter(Boolean).map((sec) => ({ sec: sec! }));
    else if (tab === "recent") out = recent.map((c) => get(c)).filter(Boolean).map((sec) => ({ sec: sec! }));
    else if (tab === "hold")
      out = holdings.flatMap((h) => (get(h.code) ? [{ sec: get(h.code)!, extra: `${h.shares} 股` }] : []));
    else if (tab === "port")
      out = (live.data?.weights ?? []).flatMap((w) => (get(w.code) ? [{ sec: get(w.code)!, extra: `${(w.w * 100).toFixed(1)}%` }] : []));
    else
      out = secs.list
        .filter((s) => s.research && s.mcap != null)
        .sort((a, b) => (b.mcap ?? 0) - (a.mcap ?? 0))
        .slice(0, 100)
        .map((sec) => ({ sec }));
    const f = filter.trim().toLowerCase();
    return f ? out.filter((r) => r.sec.name.includes(filter.trim()) || r.sec.code.toLowerCase().includes(f) || r.sec.py.some((p) => p.startsWith(f))) : out;
  }, [secs, tab, watch, recent, holdings, live.data, filter, settings.pool]);

  const quotes = useQuotes(useMemo(() => rows.slice(0, 100).map((r) => r.sec.code), [rows]));
  const empty: Record<Tab, string> = {
    watch: "还没有自选股：在个股页点击右上角 ☆ 加入",
    pool: "候选池为空：在个股页点「加入候选池」或在选股器中添加",
    hold: "还没有持仓：在「持仓调仓」页录入",
    port: live.pending ? "正在求解组合…" : "组合服务不可用",
    recent: "还没有浏览记录",
    mcap: "加载中…",
  };
  const subtitle =
    tab === "port"
      ? `${live.data?.model ?? settings.model} · 单股上限 ${settings.cap}%${live.source === "fallback" ? " · 预计算" : ""}`
      : tab === "pool"
        ? "可在选股器或组合构建中使用"
        : tab === "mcap"
        ? "研究股票池流通市值前 100"
        : tab === "hold"
          ? "来自「持仓调仓」"
          : "实时价 · 涨跌幅";

  return (
    <Panel
      title="股票列表"
      subtitle={subtitle}
      bodyClass="p-2"
      className="h-full"
      actions={
        tab === "recent" && recent.length > 0 ? (
          <button className="text-[11px] text-faint hover:text-ink" onClick={clearRecent}>
            清空
          </button>
        ) : undefined
      }
    >
      <div className="mb-2 grid grid-cols-6 gap-0.5 rounded-lg bg-bg-2 p-0.5" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={clsx("rounded-md py-1 text-[11px]", tab === t.key ? "bg-panel-2 text-ink shadow-sm" : "text-faint hover:text-ink")}
          >
            {t.label}
          </button>
        ))}
      </div>
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="筛选（代码 / 名称 / 拼音）"
        className="mb-2 h-8 w-full rounded-lg border border-line bg-bg-2 px-2.5 text-xs outline-none focus:border-accent"
      />
      <div className="max-h-[620px] space-y-0.5 overflow-y-auto pr-1">
        {rows.length === 0 && <p className="px-2 py-6 text-center text-[11px] text-faint">{filter ? "没有匹配的股票" : empty[tab]}</p>}
        {rows.map(({ sec, extra }) => {
          const q = quotes[sec.code];
          return (
            <div
              key={sec.code}
              className={clsx(
                "group flex w-full items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs transition-colors",
                sec.code === current ? "bg-panel-2 text-ink shadow-[inset_0_0_0_1px_var(--border-2)]" : "text-muted hover:bg-panel-2/60 hover:text-ink",
              )}
            >
              <button className="flex min-w-0 flex-1 flex-col items-start text-left" onClick={() => onPick(sec.code)}>
                <span className="flex w-full items-center gap-1">
                  <span className="truncate">{sec.name}</span>
                  {sec.board !== "Main" && <span className="shrink-0 text-[9px] text-faint">{BOARD_LABEL[sec.board]}</span>}
                </span>
                <span className="num text-[10px] text-faint">
                  {sec.code.slice(0, 6)}
                  {extra && <span className="ml-1.5">{extra}</span>}
                </span>
              </button>
              <span className="flex flex-col items-end">
                <span className="num text-[11px] text-ink">{q?.price != null ? q.price.toFixed(2) : "—"}</span>
                <span className={clsx("num text-[10px]", tone(q?.pct))}>
                  {q?.pct != null ? `${q.pct >= 0 ? "+" : ""}${(q.pct * 100).toFixed(2)}%` : ""}
                </span>
              </span>
              {(tab === "watch" || tab === "pool") && (
                <button
                  className="hidden text-faint hover:text-ink group-hover:block"
                  title={tab === "watch" ? "移出自选" : "移出候选池"}
                  aria-label={tab === "watch" ? "移出自选" : "移出候选池"}
                  onClick={() => (tab === "watch" ? toggleWatch(sec.code) : settings.set({ pool: settings.pool.filter((c) => c !== sec.code) }))}
                >
                  <X size={12} />
                </button>
              )}
              {tab !== "watch" && watch.includes(sec.code) && <Star size={10} className="shrink-0 fill-current text-warn" />}
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
