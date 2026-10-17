import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Check, Plus, Star } from "lucide-react";
import type { EChartsOption } from "echarts";
import { useData, type Screener, type StockDetail } from "../lib/data";
import { usePortfolioSettings } from "../lib/portfolio";
import { BOARD_LABEL, useLists, useSecurities } from "../lib/securities";
import { StockLists } from "../components/StockLists";
import { toCandles, type Candle } from "../lib/indicators";
import { useLiveKline, useMinutes, useQuotes, useRT } from "../lib/realtime";
import { IntradayChart, OrderBook } from "../components/charts/IntradayChart";
import { cssVar, useUI } from "../lib/store";
import { useAssistant } from "../lib/assistant";
import { KLineChart } from "../components/charts/KLineChart";
import { EChart } from "../components/charts/EChart";
import { AiButton, Badge, DataBanner, Delta, InfinityLoader, Panel, RimButton, Segmented } from "../components/ui";

const PROFILE = [
  { key: "size", label: "规模" },
  { key: "bm", label: "价值" },
  { key: "mom_12_2", label: "动量" },
  { key: "roe", label: "盈利" },
  { key: "vol_12m", label: "波动" },
  { key: "turn_1m", label: "换手" },
  { key: "beta_12m", label: "Beta" },
  { key: "asset_growth", label: "成长" },
];

export default function Stock() {
  const [params, setParams] = useSearchParams();
  const selected = useUI((s) => s.selectedCode);
  const select = useUI((s) => s.select);
  const code = params.get("code") ?? selected;
  const { data, loading } = useData<StockDetail>(`stocks/${code}.json`);
  const secs = useSecurities();
  const sec = secs?.byCode.get(code);
  // factor percentiles: the exported detail file if any, else the screener cross-section (all research stocks)
  const { data: scr } = useData<Screener>(!loading && !data && sec?.research ? "screener.json" : null);
  const profile: Record<string, number> | null = useMemo(() => {
    if (data?.profile) return Object.fromEntries(Object.entries(data.profile).map(([k, v]) => [k, v + 0.5]));
    const row = scr?.items.find((r) => r.code === code);
    return row ? Object.fromEntries(Object.entries(row).filter(([, v]) => typeof v === "number")) as Record<string, number> : null;
  }, [data, scr, code]);
  const { watch, toggleWatch, visit } = useLists();
  const watched = watch.includes(code);
  const { pool, set: setPortfolio } = usePortfolioSettings();
  const inPool = pool.includes(code);
  useEffect(() => {
    if (sec) visit(code);
  }, [code, sec, visit]);
  const toggleAi = useUI((st) => st.toggleAi);
  const report = useAssistant((st) => st.report);
  const setCtx = useAssistant((st) => st.setCtx);
  const quotes = useQuotes([code]);
  const q = quotes[code];
  const session = useRT((s) => s.session);
  const [view, setView] = useState<"intraday" | "kline">("kline");
  const minutes = useMinutes(view === "intraday" ? code : null);
  const live = useLiveKline(code);
  const candles: Candle[] = useMemo(() => {
    if (live) return live.bars.map(([time, open, high, low, close, volume]) => ({ time, open, high, low, close, volume, amount: close * volume }));
    return data ? toCandles(data.bars) : [];
  }, [live, data]);
  const klineSource = live
    ? live.source.startsWith("sina")
      ? "实时接口（新浪，不复权，含今日；腾讯源缺少该代码的历史日线）"
      : "实时接口（腾讯，前复权，含今日）"
    : data
      ? "本地演示数据（截至导出日）"
      : "";
  // flash the price when it ticks
  const [flash, setFlash] = useState<"" | "up" | "down">("");
  const lastPrice = useRef<number | null>(null);
  useEffect(() => {
    if (q?.price == null) return;
    if (lastPrice.current != null && q.price !== lastPrice.current) {
      setFlash(q.price > lastPrice.current ? "up" : "down");
      const id = setTimeout(() => setFlash(""), 700);
      lastPrice.current = q.price;
      return () => clearTimeout(id);
    }
    lastPrice.current = q.price;
  }, [q?.price]);

  const pick = (c: string) => {
    select(c);
    setParams({ code: c });
  };

  const last = candles[candles.length - 1];
  const prev = candles[candles.length - 2];
  const yr = candles[Math.max(0, candles.length - 244)];
  const price = q?.price ?? last?.close ?? null;
  const pctNow = q ? q.pct : last && prev ? last.close / prev.close - 1 : null;
  const name = q?.name ?? data?.name ?? sec?.name ?? code;
  const industry = data?.industry ?? (sec?.industry || undefined);
  useEffect(() => setCtx({ code, name }), [code, name, setCtx]);

  const radar: EChartsOption | null = profile
    ? {
        tooltip: {},
        radar: {
          indicator: PROFILE.map((p) => ({ name: p.label, max: 100, min: 0 })),
          radius: "66%",
          axisName: { color: cssVar("--muted"), fontSize: 11 },
          splitLine: { lineStyle: { color: cssVar("--grid") } },
          splitArea: { areaStyle: { color: ["transparent"] } },
          axisLine: { lineStyle: { color: cssVar("--grid") } },
        },
        series: [
          {
            type: "radar",
            data: [
              {
                value: PROFILE.map((p) => Math.round((profile[p.key] ?? 0.5) * 100)),
                name: "截面分位",
                areaStyle: { color: cssVar("--m-cil") + "40" },
                lineStyle: { color: cssVar("--m-cil") },
                itemStyle: { color: cssVar("--m-cil") },
              },
            ],
          },
        ],
      }
    : null;

  return (
    <div className="grid gap-4 xl:grid-cols-[200px_minmax(0,1fr)_300px]">
      <div className="hidden xl:block">
        <StockLists current={code} onPick={pick} />
      </div>
      <div className="min-w-0">
        <DataBanner note={data?.meta.survivorship_biased ? data.meta.note : undefined} />
        <Panel
          title={
            <span className="flex items-center gap-2">
              <span className="text-base">{name}</span>
              <span className="num text-xs font-normal text-faint">{code}</span>
              {industry && <Badge>{industry}</Badge>}
              {sec && sec.board !== "Main" && <Badge color="var(--faint)">{BOARD_LABEL[sec.board]}</Badge>}
            </span>
          }
          actions={
            <>
              <Segmented name="view" value={view} onChange={setView} options={[{ value: "intraday", label: "分时" }, { value: "kline", label: "K 线" }]} />
              <RimButton onClick={() => setPortfolio({ pool: inPool ? pool.filter((c) => c !== code) : [...pool, code] })}>
                {inPool ? <Check size={14} /> : <Plus size={14} />} {inPool ? `已在候选池（${pool.length}）` : "加入候选池"}
              </RimButton>
              <button
                className={clsx("rounded-lg p-2 hover:bg-panel-2", watched ? "text-warn" : "text-muted hover:text-warn")}
                title={watched ? "移出自选" : "加入自选"}
                aria-label={watched ? "移出自选" : "加入自选"}
                aria-pressed={watched}
                onClick={() => toggleWatch(code)}
              >
                <Star size={16} className={watched ? "fill-current" : undefined} />
              </button>
            </>
          }
        >
          {price != null && (
            <div className="mb-3 flex flex-wrap items-end gap-x-6 gap-y-2">
              <div
                className={clsx(
                  "num rounded-lg px-1 text-3xl font-semibold transition-colors duration-500",
                  (pctNow ?? 0) >= 0 ? "text-up" : "text-down",
                  flash === "up" && "bg-up/20",
                  flash === "down" && "bg-down/20",
                )}
              >
                {price.toFixed(2)}
              </div>
              <Delta value={pctNow} className="text-sm" />
              {yr && last && <div className="text-xs text-muted">近一年 <Delta value={last.close / yr.close - 1} digits={1} /></div>}
              {q && (
                <div className="flex items-center gap-1.5 text-[11px] text-faint">
                  <i className={clsx("h-1.5 w-1.5 rounded-full", session?.code === "trading" ? "animate-pulse bg-[#22b07d]" : "bg-faint")} />
                  {session?.label ?? ""} · {q.time.slice(5, 16)} · {q.source === "tencent" ? "腾讯" : "新浪"}行情
                </div>
              )}
            </div>
          )}
          {view === "intraday" ? (
            minutes?.bars?.length ? (
              <IntradayChart bars={minutes.bars} prevClose={q?.prev_close ?? null} height={520} />
            ) : (
              <div className="grid h-[520px] place-items-center">{minutes ? <span className="text-sm text-muted">暂无分时数据（实时服务未启动或非交易日）</span> : <InfinityLoader label="加载分时" />}</div>
            )
          ) : candles.length ? (
            <>
              <KLineChart candles={candles} height={540} />
              <div className="mt-1 text-[10px] text-faint">K 线来源：{klineSource}</div>
            </>
          ) : (
            <div className="grid h-[520px] place-items-center">{loading ? <InfinityLoader label="加载 K 线" /> : <span className="text-sm text-muted">暂无 K 线数据</span>}</div>
          )}
        </Panel>
      </div>
      <div className="flex flex-col gap-4">
        <Panel title="五档盘口" subtitle={q ? `更新于 ${q.time.slice(11)}` : "实时服务未连接"} bodyClass="p-3">
          {q ? (
            <>
              <OrderBook bids={q.bids} asks={q.asks} prevClose={q.prev_close} />
              <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1 border-t border-line pt-2 text-[11px]">
                {[
                  ["今开", q.open?.toFixed(2)], ["昨收", q.prev_close?.toFixed(2)],
                  ["最高", q.high?.toFixed(2)], ["最低", q.low?.toFixed(2)],
                  ["涨停", q.limit_up?.toFixed(2) ?? "—"], ["跌停", q.limit_down?.toFixed(2) ?? "—"],
                  ["成交额", `${(q.amount / 1e8).toFixed(2)} 亿`], ["换手", q.turnover != null ? `${(q.turnover * 100).toFixed(2)}%` : "—"],
                  ["市盈率", q.pe?.toFixed(2) ?? "—"], ["市净率", q.pb?.toFixed(2) ?? "—"],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between"><span className="text-muted">{k}</span><span className="num">{v ?? "—"}</span></div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-xs text-muted">启动 API 服务后显示实时五档盘口：<code className="text-[10px]">uvicorn app.main:app --app-dir services/api</code></p>
          )}
        </Panel>
        <Panel title="因子画像" subtitle="月末截面分位（0–100），与 CIL 模块的特征一致" accent="var(--m-cil)">
          {radar ? (
            <EChart option={radar} height={240} />
          ) : (
            <p className="py-10 text-center text-xs text-muted">
              {sec && !sec.research ? "该股票不在研究股票池（沪深300 / 中证500 历史成分股），暂无因子数据" : "加载因子数据…"}
            </p>
          )}
          {profile && (
            <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
              {PROFILE.map((p) => (
                <div key={p.key} className="flex justify-between">
                  <span className="text-muted">{p.label}</span>
                  <span className="num">{Math.round((profile[p.key] ?? 0.5) * 100)}</span>
                </div>
              ))}
            </div>
          )}
        </Panel>
        <Panel
          title="AI 研报"
          subtitle="多智能体：数据收集 → 多空分析师 → 风控主编（不给价位与仓位）"
          actions={<AiButton onClick={() => { toggleAi(true); report(code, name); }}>生成</AiButton>}
        >
          <div className="space-y-2 text-xs text-muted">
            <p>点击「生成」，研报在右侧助手中流式输出：</p>
            {[
              ["1", "数据收集", "个股技术指标、24 个因子分位、实时行情、行业可比、模型组合权重（不调用大模型）", "var(--m-fce)"],
              ["2", "多空研判", "多头与空头分析师并行，只能引用给定数据", "var(--up)"],
              ["3", "风控主编", "整合为固定结构的简报；每个数字自动核验出处，并做合规检查", "var(--m-dpl)"],
            ].map(([n, t, d, c]) => (
              <div key={n} className="flex gap-2 rounded-lg border border-line p-2">
                <span className="num grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] text-white" style={{ background: c }}>{n}</span>
                <div><span className="font-semibold text-ink">{t}</span> · {d}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
