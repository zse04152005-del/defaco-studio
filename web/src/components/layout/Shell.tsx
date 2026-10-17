import clsx from "clsx";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  Bell,
  BriefcaseBusiness,
  CandlestickChart,
  FlaskConical,
  Gauge,
  History,
  LayoutDashboard,
  ListFilter,
  ShieldHalf,
  Sigma,
} from "lucide-react";
import { useData, type Overview } from "../../lib/data";
import { useUI } from "../../lib/store";
import { useAssistant } from "../../lib/assistant";
import { useMonitorSync } from "../../lib/monitor";
import { BOARD_LABEL, searchSecurities, useLists, useSecurities, type Sec } from "../../lib/securities";
import { AssistantDrawer } from "../assistant/AssistantDrawer";
import { useQuotes, useRT } from "../../lib/realtime";
import { AiButton, SearchIcon, Switch, ThemeSwitch } from "../ui";
import { Delta } from "../ui";

export const NAV = [
  { to: "/", label: "市场总览", icon: LayoutDashboard, key: "1" },
  { to: "/stock", label: "个股 K 线", icon: CandlestickChart, key: "2" },
  { to: "/screener", label: "选股器", icon: ListFilter, key: "3" },
  { to: "/factors", label: "因子研究", icon: Sigma, key: "4" },
  { to: "/portfolio", label: "组合构建", icon: Gauge, key: "5" },
  { to: "/risk", label: "风险解析", icon: ShieldHalf, key: "6" },
  { to: "/rebalance", label: "持仓调仓", icon: BriefcaseBusiness, key: "7" },
  { to: "/backtest", label: "回测绩效", icon: History, key: "8" },
  { to: "/alerts", label: "监控预警", icon: Bell, key: "9" },
  { to: "/lab", label: "论文实验室", icon: FlaskConical, key: "0" },
];

function Sidebar() {
  return (
    <aside className="flex w-[68px] shrink-0 flex-col items-center gap-1 border-r border-line bg-bg-2 py-3 lg:w-[188px] lg:items-stretch lg:px-3">
      <div className="mb-4 flex items-center gap-2 px-1">
        <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden>
          <rect width="32" height="32" rx="8" fill="var(--panel-2)" />
          <path d="M7 22l6-7 5 4 7-9" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="25" cy="10" r="2.5" fill="var(--up)" />
        </svg>
        <div className="hidden leading-tight lg:block">
          <div className="text-sm font-bold tracking-tight">DeFaCo Studio</div>
          <div className="text-[10px] text-faint">因子风险 · 组合管理</div>
        </div>
      </div>
      {NAV.map(({ to, label, icon: Icon, key }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          title={`${label}（${key}）`}
          className={({ isActive }) =>
            clsx(
              "group flex items-center gap-3 rounded-xl px-2.5 py-2 text-[13px] transition-colors",
              isActive ? "bg-panel-2 text-ink shadow-[inset_0_0_0_1px_var(--border-2)]" : "text-muted hover:bg-panel hover:text-ink",
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon size={18} className={isActive ? "text-accent" : ""} />
              <span className="hidden lg:inline">{label}</span>
              <span className="ml-auto hidden text-[10px] text-faint lg:inline">{key}</span>
            </>
          )}
        </NavLink>
      ))}
    </aside>
  );
}

function StockSearch() {
  const secs = useSecurities();
  const recent = useLists((s) => s.recent);
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const ref = useRef<HTMLInputElement>(null);
  const nav = useNavigate();
  const select = useUI((s) => s.select);
  const hits = useMemo(() => {
    if (!secs) return [];
    if (!q.trim()) return recent.map((c) => secs.byCode.get(c)).filter((x): x is Sec => !!x).slice(0, 8);
    return searchSecurities(secs.list, q, 10);
  }, [secs, q, recent]);
  useEffect(() => setActive(0), [q]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        ref.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const go = (code: string) => {
    select(code);
    nav(`/stock?code=${code}`);
    setQ("");
    setOpen(false);
    ref.current?.blur();
  };
  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.nativeEvent.isComposing) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, hits.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter" && hits[active]) {
      go(hits[active].code);
    } else if (e.key === "Escape") {
      setOpen(false);
      ref.current?.blur();
    }
  };
  return (
    <div className="relative w-full max-w-md">
      <div className="uv-search">
        <SearchIcon />
        <input
          ref={ref}
          value={q}
          placeholder="代码 / 名称 / 拼音首字母（如 gzmt）"
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={onKeyDown}
          role="combobox"
          aria-expanded={open && hits.length > 0}
          aria-label="搜索股票"
        />
        <kbd className="absolute right-2.5 rounded-md border border-line px-1.5 text-[10px] text-faint">⌘K</kbd>
      </div>
      {open && hits.length > 0 && (
        <div className="panel absolute left-0 right-0 top-11 z-50 overflow-hidden p-1" role="listbox">
          {!q.trim() && <div className="px-3 pb-1 pt-1.5 text-[10.5px] text-faint">最近浏览</div>}
          {hits.map((h, i) => (
            <button
              key={h.code}
              role="option"
              aria-selected={i === active}
              onMouseDown={() => go(h.code)}
              onMouseEnter={() => setActive(i)}
              className={clsx("flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[13px]", i === active && "bg-panel-2")}
            >
              <span className="num w-20 text-muted">{h.code}</span>
              <span className="flex-1 truncate">
                {h.name}
                {q.trim() && h.py[0] && <span className="ml-1.5 text-[10.5px] uppercase text-faint">{h.py[0]}</span>}
              </span>
              {h.board !== "Main" && <span className="rounded border border-line px-1 text-[10px] text-faint">{BOARD_LABEL[h.board]}</span>}
              <span className="w-16 truncate text-right text-[11px] text-faint">{h.industry}</span>
            </button>
          ))}
          {q.trim() && <div className="border-t border-line px-3 pb-1 pt-1.5 text-[10.5px] text-faint">↑↓ 选择 · Enter 打开 · Esc 关闭 · 共 {secs?.list.length ?? 0} 只证券</div>}
        </div>
      )}
    </div>
  );
}

function SessionPill() {
  const session = useRT((s) => s.session);
  const mode = useRT((s) => s.mode);
  if (!session) return <span className="rounded-full border border-line px-2.5 py-1 text-[11px] text-faint">{mode === "offline" ? "实时服务未连接" : "连接中…"}</span>;
  const live = session.code === "trading" || session.code === "auction";
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-[11px]" title="A 股交易时段（北京时间）">
      <i className={clsx("h-1.5 w-1.5 rounded-full", live ? "animate-pulse bg-[#22b07d]" : "bg-faint")} />
      {session.label}
    </span>
  );
}

function TopBar() {
  const { theme, setTheme, upDown, setUpDown, toggleAi } = useUI();
  const { data } = useData<Overview>("overview.json");
  const quotes = useQuotes(["000300.SH"]);
  const live = quotes["000300.SH"];
  const idx = data?.indices.sh000300;
  const value = live?.price ?? (idx ? idx[idx.length - 1][1] : null);
  const chg = live ? live.pct : idx && idx.length > 1 ? idx[idx.length - 1][1] / idx[idx.length - 2][1] - 1 : null;
  const vol = data?.vol_regime.at(-1)?.[1];
  const high = vol != null && data ? vol > data.vol_threshold : null;
  return (
    <header className="relative z-30 flex h-14 shrink-0 items-center gap-4 border-b border-line bg-bg/80 px-4 backdrop-blur">
      <StockSearch />
      <div className="ml-auto hidden items-center gap-4 text-xs md:flex">
        {value != null && (
          <span className="flex items-center gap-2">
            <span className="text-muted">沪深300</span>
            <span className="num font-semibold">{value.toFixed(2)}</span>
            <Delta value={chg} />
          </span>
        )}
        <SessionPill />
        {high != null && (
          <span
            className="hidden items-center gap-1.5 rounded-full px-2.5 py-1 xl:flex"
            style={{ background: "color-mix(in srgb, var(--m-reg) 14%, transparent)", color: "var(--m-reg)" }}
            title="20 日年化波动率相对历史 75% 分位"
          >
            <i className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--m-reg)" }} />
            区制：{high ? "高波动" : "低波动"}
          </span>
        )}
      </div>
      <Switch checked={upDown === "green"} onChange={(v) => setUpDown(v ? "green" : "red")} label={<span className="hidden xl:inline">绿涨红跌</span>} />
      <ThemeSwitch light={theme === "light"} onChange={(l) => setTheme(l ? "light" : "dark")} />
      <AiButton onClick={() => toggleAi()}>AI 助手</AiButton>
    </header>
  );
}

function StatusBar() {
  const { data } = useData<Overview>("overview.json");
  const mode = useRT((s) => s.mode);
  const updatedAt = useRT((s) => s.updatedAt);
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);
  const ago = updatedAt ? Math.max(0, Math.round((Date.now() - updatedAt) / 1000)) : null;
  return (
    <footer className="flex h-7 shrink-0 items-center gap-4 border-t border-line bg-bg-2 px-4 text-[11px] text-faint">
      <span>研究数据截至 {data?.asof ?? "…"}</span>
      <span>
        实时行情：{mode === "ws" ? "WebSocket 推送" : mode === "poll" ? "轮询" : "未连接"}
        {ago != null && ` · ${ago}s 前更新`} · 腾讯 / 新浪（可能延迟）
      </span>
      <span className="ml-auto">⚠ 仅供研究与教学，不构成投资建议</span>
    </footer>
  );
}

const PAGE_KEY: Record<string, string> = {
  "/": "overview", "/stock": "stock", "/screener": "screener", "/portfolio": "portfolio", "/risk": "risk",
  "/rebalance": "rebalance", "/backtest": "backtest", "/alerts": "alerts", "/lab": "lab", "/factors": "factors",
};

export function Shell({ children }: { children: ReactNode }) {
  const nav = useNavigate();
  const { pathname } = useLocation();
  const setCtx = useAssistant((s) => s.setCtx);
  useEffect(() => setCtx({ page: PAGE_KEY[pathname] ?? "overview" }), [pathname, setCtx]);
  useMonitorSync(); // the scheduled alert rules use the user's holdings / watchlist / pool
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
      const item = NAV.find((n) => n.key === e.key);
      if (item) nav(item.to);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nav]);
  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="min-h-0 flex-1 overflow-y-auto px-5 py-5">{children}</main>
        <StatusBar />
      </div>
      <AssistantDrawer />
    </div>
  );
}
