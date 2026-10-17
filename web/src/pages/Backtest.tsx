import clsx from "clsx";
import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Download, FileText, Play, Plus, Trash2 } from "lucide-react";
import { useData } from "../lib/data";
import { DEFAULT_FORM, fromStatic, useBacktest, type BacktestResult, type Metrics, type StrategyForm } from "../lib/backtest";
import { useDefacoAvailable, usePortfolioSettings } from "../lib/portfolio";
import { cssVar } from "../lib/store";
import { pct } from "../lib/format";
import { EChart } from "../components/charts/EChart";
import { Badge, DataBanner, GlowButton, PageHeader, Panel, RimButton, Segmented, Switch } from "../components/ui";

const MODELS: Record<string, string> = { defaco: "DeFaCo（论文 M4）", pca_factor: "PCA 因子", lw_nonlinear: "非线性收缩 NLS", lw_linear: "线性收缩 LW", sample: "样本协方差", ewma: "EWMA" };
const OBJ: Record<string, string> = { minvar: "最小方差", rp: "风险平价", hrp: "HRP", mv: "均值-方差" };
const PALETTE = ["#4c8dff", "#f5a524", "#22c3a0", "#b37feb", "#e0524a", "#7a8699", "#1aa3d3"];
const AUM: [number, string][] = [[1e7, "1 千万"], [1e8, "1 亿"], [1e9, "10 亿"], [1e10, "100 亿"]];

type Row = { k: keyof Metrics; label: string; f: (v: never) => string; better?: "low" | "high"; hint?: string };
const ROWS: Row[] = [
  { k: "ann_return", label: "年化收益", f: (v: number) => pct(v), better: "high" },
  { k: "ann_vol", label: "年化波动", f: (v: number) => pct(v), better: "low" },
  { k: "sharpe", label: "Sharpe", f: (v: number) => v.toFixed(2), better: "high", hint: "扣除 SHIBOR 3M 无风险利率" },
  { k: "sharpe_ci95", label: "Sharpe 95% CI", f: (v: [number, number]) => `[${v[0].toFixed(2)}, ${v[1].toFixed(2)}]`, hint: "考虑偏度与峰度的标准误（Mertens 2002）" },
  { k: "psr_0", label: "PSR", f: (v: number) => pct(v, 0), better: "high", hint: "Sharpe>0 的概率（Bailey & López de Prado 2012）" },
  { k: "dsr", label: "DSR", f: (v: number) => pct(v, 0), better: "high", hint: "对“同时比较多个策略”做校正后的 PSR（Bailey & López de Prado 2014）" },
  { k: "max_drawdown", label: "最大回撤", f: (v: number) => pct(v, 1), better: "high" },
  { k: "calmar", label: "Calmar", f: (v: number) => v.toFixed(2), better: "high" },
  { k: "cvar95_daily", label: "日 CVaR95", f: (v: number) => pct(v), better: "high" },
  { k: "excess_ann_return", label: "超额年化", f: (v: number) => pct(v), better: "high", hint: "相对沪深300" },
  { k: "information_ratio", label: "IR", f: (v: number) => v.toFixed(2), better: "high" },
  { k: "beta", label: "Beta", f: (v: number) => v.toFixed(2) },
  { k: "annual_turnover", label: "年换手", f: (v: number) => `${v.toFixed(2)}×`, better: "low" },
  { k: "cost_drag_ann", label: "成本拖累", f: (v: number) => pct(v, 2), better: "low", hint: "佣金 + 印花税 + 过户费 + 冲击成本（年化）" },
  { k: "avg_holdings", label: "持股数", f: (v: number) => v.toFixed(0) },
];

function NumInput({ value, onChange, step = 1, min, max, suffix, w = "w-14" }: { value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; suffix?: string; w?: string }) {
  return (
    <span className="flex items-center gap-1">
      <input type="number" value={value} step={step} min={min} max={max} onChange={(e) => onChange(+e.target.value)}
        className={clsx("num h-7 rounded-md border border-line bg-bg-2 px-1.5 text-right text-[11px] text-ink outline-none focus:border-accent", w)} />
      {suffix && <span className="text-faint">{suffix}</span>}
    </span>
  );
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: Record<string, string> }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="h-7 rounded-md border border-line bg-bg-2 px-1.5 text-[11px] text-ink outline-none">
      {Object.entries(options).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
    </select>
  );
}

function Settings({ running }: { running: boolean }) {
  const defacoOn = useDefacoAvailable();
  const { form, setForm, run } = useBacktest();
  const pool = usePortfolioSettings((s) => s.pool);
  const ex = form.execution;
  const setS = (i: number, p: Partial<StrategyForm>) => setForm({ strategies: form.strategies.map((s, j) => (j === i ? { ...s, ...p } : s)) });
  return (
    <Panel title="回测设置" subtitle="信号只用当日及以前的数据；默认 T+1 开盘成交" bodyClass="p-3 space-y-3 text-[11px]">
      <div>
        <div className="mb-1.5 flex items-center justify-between text-muted">
          <span>策略（最多 6 个，另可加等权基准）</span>
          <button className="flex items-center gap-1 text-accent disabled:opacity-40" disabled={form.strategies.length >= 6}
            onClick={() => setForm({ strategies: [...form.strategies, { ...DEFAULT_FORM.strategies[0], model: "lw_nonlinear", objective: "hrp" }] })}>
            <Plus size={12} /> 添加
          </button>
        </div>
        <div className="space-y-1.5">
          {form.strategies.map((s, i) => (
            <div key={i} className="flex flex-wrap items-center gap-1.5 rounded-lg border border-line p-1.5">
              <i className="h-2.5 w-2.5 rounded-full" style={{ background: PALETTE[i % 7] }} />
              <Select value={s.model} onChange={(v) => setS(i, { model: v })} options={defacoOn ? MODELS : Object.fromEntries(Object.entries(MODELS).filter(([k]) => k !== "defaco"))} />
              <Select value={s.objective} onChange={(v) => setS(i, { objective: v })} options={OBJ} />
              <span className="text-faint">单股≤</span>
              <NumInput value={Math.round(s.max_weight * 100)} onChange={(v) => setS(i, { max_weight: Math.max(1, v) / 100 })} min={1} max={100} suffix="%" w="w-11" />
              {s.objective === "mv" && (<><span className="text-faint">γ</span><NumInput value={s.risk_aversion} onChange={(v) => setS(i, { risk_aversion: Math.max(0.5, v) })} w="w-11" /></>)}
              {(s.objective === "minvar" || s.objective === "mv") && (
                <><span className="text-faint">行业≤</span><NumInput value={Math.round((s.industry_cap ?? 1) * 100)} onChange={(v) => setS(i, { industry_cap: v >= 100 ? null : Math.max(5, v) / 100 })} suffix="%" w="w-11" /></>
              )}
              <button className="ml-auto text-faint hover:text-down disabled:opacity-30" disabled={form.strategies.length <= 1}
                onClick={() => setForm({ strategies: form.strategies.filter((_, j) => j !== i) })} aria-label="删除策略"><Trash2 size={12} /></button>
            </div>
          ))}
        </div>
        <div className="mt-1.5 flex items-center gap-2 text-muted"><Switch checked={form.include_ew} onChange={(v) => setForm({ include_ew: v })} /> 同时回测等权组合（同一股票池）</div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <label className="flex items-center gap-1.5 text-muted">开始 <input type="date" value={form.start} min="2014-01-01" max="2025-12-31" onChange={(e) => setForm({ start: e.target.value })} className="num h-7 rounded-md border border-line bg-bg-2 px-1 text-ink" /></label>
        <label className="flex items-center gap-1.5 text-muted">结束 <input type="date" value={form.end} min="2014-06-01" max="2025-12-31" onChange={(e) => setForm({ end: e.target.value })} className="num h-7 rounded-md border border-line bg-bg-2 px-1 text-ink" /></label>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-muted">
        调仓 <Segmented name="freq" value={form.freq} onChange={(v) => setForm({ freq: v })} options={[{ value: "W", label: "周" }, { value: "M", label: "月" }, { value: "Q", label: "季" }]} />
      </div>
      <div className="flex flex-wrap items-center gap-2 text-muted">
        股票池
        <Segmented name="univ" value={form.use_pool && pool.length ? "pool" : "top"} onChange={(v) => setForm({ use_pool: v === "pool" })}
          options={[{ value: "top", label: "沪深300 前 N" }, ...(pool.length ? [{ value: "pool", label: `候选池 ${pool.length}` }] : [])]} />
        {!(form.use_pool && pool.length) && <NumInput value={form.top_n} onChange={(v) => setForm({ top_n: Math.min(300, Math.max(20, v)) })} step={10} suffix="只" />}
      </div>
      <div className="rounded-lg border border-line p-2">
        <div className="mb-1.5 text-muted">成交与成本（参照 Qlib Exchange 与 A 股规则）</div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <Segmented name="timing" value={ex.timing} onChange={(v) => setForm({ execution: { ...ex, timing: v } })} options={[{ value: "next_open", label: "T+1 开盘" }, { value: "close", label: "当日收盘" }]} />
          <span className="flex items-center gap-1 text-faint">资金 <Select value={String(ex.aum)} onChange={(v) => setForm({ execution: { ...ex, aum: +v } })} options={Object.fromEntries(AUM.map(([a, l]) => [String(a), l]))} /></span>
          <span className="flex items-center gap-1 text-faint">佣金 <NumInput value={ex.commission_bps} step={0.5} min={0} onChange={(v) => setForm({ execution: { ...ex, commission_bps: v } })} suffix="bp" w="w-11" /></span>
          <span className="flex items-center gap-1 text-faint" title="平方根冲击：Y × 日波动 × √(成交额/当日成交额)">冲击 Y <NumInput value={ex.impact_y} step={0.1} min={0} onChange={(v) => setForm({ execution: { ...ex, impact_y: v } })} w="w-11" /></span>
          <span className="flex items-center gap-1 text-faint" title="单笔成交不超过当日成交额的比例">成交量上限 <NumInput value={ex.participation_cap == null ? 0 : Math.round(ex.participation_cap * 100)} min={0} max={100}
            onChange={(v) => setForm({ execution: { ...ex, participation_cap: v <= 0 ? null : v / 100 } })} suffix="%（0=不限）" w="w-11" /></span>
        </div>
        <p className="mt-1.5 text-[10px] text-faint">另含：卖出印花税 0.1%（2023-08-28 起 0.05%）、过户费 0.001%、每笔最低佣金 5 元；涨跌停 / 停牌订单失败的部分留作现金，现金按 SHIBOR 计息。</p>
      </div>
      <div className="flex items-center gap-2">
        <GlowButton onClick={() => run(pool)} disabled={running}><Play size={15} /> {running ? "回测中…" : "运行回测"}</GlowButton>
        <button className="text-[11px] text-faint hover:text-ink" onClick={() => setForm(DEFAULT_FORM)}>恢复默认</button>
      </div>
    </Panel>
  );
}

export default function Backtest() {
  const { status, progress, message, error, elapsed, result, job, resume } = useBacktest();
  const { data: staticData } = useData<unknown>("backtest.json");
  useEffect(() => {
    resume();
  }, [resume]);
  const running = status === "queued" || status === "running";
  const res: BacktestResult | null = result ?? (staticData ? fromStatic(staticData) : null);
  const names = useMemo(() => (res ? Object.keys(res.strategies) : []), [res]);
  const [focus, setFocus] = useState<string | null>(null);
  const [logScale, setLogScale] = useState(false);
  const f = focus && names.includes(focus) ? focus : names[0];

  return (
    <div>
      <PageHeader
        title="回测绩效"
        desc={res ? `${res.start} → ${res.end} · ${res.freq}调仓 ${res.rebalances} 次 · ${res.universe} · ${res.execution.timing === "next_open" ? "T+1 开盘成交" : "当日收盘成交"}` : "实时回测"}
        actions={
          res && !res.offline && job ? (
            <>
              <RimButton onClick={() => window.open(`/api/backtest/jobs/${job}/report.html`, "_blank")}><FileText size={14} /> 导出 HTML 报告</RimButton>
              <RimButton onClick={() => window.open(`/api/backtest/jobs/${job}/returns.csv`, "_blank")}><Download size={14} /> 日收益 CSV</RimButton>
            </>
          ) : undefined
        }
      />
      <DataBanner note={res?.offline ? "当前显示的是离线预计算结果（收盘成交、无冲击成本）；运行一次回测即可得到实时结果和可导出的报告。" : undefined} />
      <div className="grid gap-4 xl:grid-cols-[420px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Settings running={running} />
          {(running || status === "error") && (
            <Panel title={running ? "运行中" : "出错"} bodyClass="p-3 text-[12px]">
              {running ? (
                <>
                  <div className="mb-1.5 h-2 overflow-hidden rounded-full bg-panel-2"><div className="h-full rounded-full bg-accent transition-all" style={{ width: `${Math.max(3, progress * 100)}%` }} /></div>
                  <div className="text-muted">{message || "排队中…"} · {elapsed.toFixed(0)} 秒</div>
                  <div className="text-[10.5px] text-faint">首次运行需加载 2012 年以来的研究数据（约 15 秒）；每个策略每次调仓都重新估计风险模型并求解。</div>
                </>
              ) : (
                <span className="text-warn">{error}</span>
              )}
            </Panel>
          )}
        </div>
        {res && <Results res={res} names={names} focus={f} setFocus={setFocus} logScale={logScale} setLogScale={setLogScale} />}
      </div>
    </div>
  );
}

function Results({ res, names, focus, setFocus, logScale, setLogScale }: {
  res: BacktestResult; names: string[]; focus: string; setFocus: (k: string) => void; logScale: boolean; setLogScale: (v: boolean) => void;
}) {
  const S = res.strategies;
  const color = (k: string) => PALETTE[names.indexOf(k) % 7];
  const navOption: EChartsOption = {
    grid: { left: 50, right: 16, top: 34, bottom: 30 },
    legend: { top: 0, left: 0, type: "scroll" },
    tooltip: { trigger: "axis", valueFormatter: (v) => (v as number).toFixed(3) },
    xAxis: { type: "time" },
    yAxis: { type: logScale ? "log" : "value", scale: true },
    dataZoom: [{ type: "inside" }],
    series: [
      ...names.map((k) => ({ name: k, type: "line" as const, showSymbol: false, data: S[k].nav, lineStyle: { width: k === focus ? 2.2 : 1.2, color: color(k) }, itemStyle: { color: color(k) } })),
      ...(res.benchmark ? [{ name: "沪深300", type: "line" as const, showSymbol: false, data: res.benchmark.nav, lineStyle: { width: 1, type: "dashed" as const, color: cssVar("--muted") }, itemStyle: { color: cssVar("--muted") } }] : []),
    ],
  };
  const ddOption: EChartsOption = {
    grid: { left: 50, right: 16, top: 10, bottom: 30 },
    tooltip: { trigger: "axis", valueFormatter: (v) => pct(v as number, 1) },
    xAxis: { type: "time" },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: names.map((k) => ({ name: k, type: "line" as const, showSymbol: false, data: S[k].drawdown, lineStyle: { width: k === focus ? 1.6 : 0.8, color: color(k) },
      ...(k === focus ? { areaStyle: { color: color(k) + "22" } } : {}) })),
  };
  const months = Object.entries(S[focus].monthly);
  const years = [...new Set(months.map(([k]) => k.slice(0, 4)))];
  const calOption: EChartsOption = {
    grid: { left: 44, right: 10, top: 10, bottom: 24 },
    tooltip: { formatter: (p: unknown) => { const v = (p as { value: [number, number, number] }).value; return `${years[v[1]]}-${String(v[0] + 1).padStart(2, "0")}：${pct(v[2], 2)}`; } },
    xAxis: { type: "category", data: Array.from({ length: 12 }, (_, i) => `${i + 1}月`), splitLine: { show: false } },
    yAxis: { type: "category", data: years, splitLine: { show: false } },
    visualMap: { show: false, min: -0.08, max: 0.08, inRange: { color: [cssVar("--down"), cssVar("--panel-2"), cssVar("--up")] } },
    series: [{ type: "heatmap", data: months.map(([k, v]) => [+k.slice(5, 7) - 1, years.indexOf(k.slice(0, 4)), v]),
      label: { show: true, fontSize: 9, color: cssVar("--text"), formatter: (p) => ((p.value as number[])[2] * 100).toFixed(1) },
      itemStyle: { borderColor: cssVar("--panel"), borderWidth: 2, borderRadius: 3 } }],
  };
  const yr = (m: Record<string, number>) => {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(m)) out[k.slice(0, 4)] = (1 + (out[k.slice(0, 4)] ?? 0)) * (1 + v) - 1;
    return out;
  };
  const yearOption: EChartsOption = {
    grid: { left: 44, right: 10, top: 30, bottom: 24 },
    legend: { top: 0, type: "scroll" },
    tooltip: { trigger: "axis", valueFormatter: (v) => pct(v as number, 1) },
    xAxis: { type: "category", data: years },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: names.map((k) => ({ name: k, type: "bar" as const, data: years.map((y) => yr(S[k].monthly)[y] ?? null), itemStyle: { color: color(k) } })),
  };
  const sig = names.filter((k) => (S[k].metrics.sharpe_ci95?.[0] ?? -1) > 0);
  const best = (k: keyof Metrics, better?: "low" | "high") => {
    if (!better) return null;
    const vals = names.map((n) => S[n].metrics[k] as number).filter((v) => typeof v === "number" && Number.isFinite(v));
    return better === "low" ? Math.min(...vals) : Math.max(...vals);
  };
  const m0 = S[names[0]].metrics;
  return (
    <div className="min-w-0 space-y-4">
      {!res.offline && (
        <div className="rounded-xl border border-line bg-panel-2/60 px-4 py-2.5 text-[12px]">
          <b>统计结论：</b>
          {sig.length ? `${sig.join("、")} 的 Sharpe 95% 置信区间不含 0。` : "所有策略的 Sharpe 95% 置信区间都包含 0：这段样本里收益差异在统计上不显著，比较应以波动、回撤等风险指标为主。"}
          {m0.dsr_trials && m0.dsr_trials > 1 && ` 同时比较了 ${m0.dsr_trials} 个策略，纯靠运气的最好策略年化 Sharpe 也可达约 ${(m0.sr0_ann ?? 0).toFixed(2)}，DSR 已对此校正。`}
        </div>
      )}
      <Panel title="净值与回撤（扣费后）" actions={<div className="flex items-center gap-2 text-[11px] text-muted">对数坐标 <Switch checked={logScale} onChange={setLogScale} /></div>} bodyClass="p-2 pt-3">
        <EChart option={navOption} height={300} />
        <EChart option={ddOption} height={130} />
      </Panel>
      <Panel title="绩效与统计检验" subtitle="蓝色加粗为该列最优；点击行切换下方图表" bodyClass="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-[11.5px]">
            <thead className="text-left text-faint">
              <tr className="border-b border-line">
                <th className="px-3 py-2 font-medium">策略</th>
                {ROWS.map((c) => <th key={c.k} className="whitespace-nowrap px-2 py-2 text-right font-medium" title={c.hint}>{c.label}{c.hint ? " ⓘ" : ""}</th>)}
              </tr>
            </thead>
            <tbody>
              {names.map((k) => (
                <tr key={k} onClick={() => setFocus(k)} className={clsx("cursor-pointer border-b border-line/60 hover:bg-panel-2/40", k === focus && "bg-panel-2/60")}>
                  <td className="whitespace-nowrap px-3 py-2"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: color(k) }} />{k}</td>
                  {ROWS.map((c) => {
                    const v = S[k].metrics[c.k] as never;
                    const b = best(c.k, c.better);
                    const isBest = b != null && typeof v === "number" && Math.abs((v as number) - b) < 1e-12;
                    return <td key={c.k} className={clsx("num whitespace-nowrap px-2 py-2 text-right", isBest && "font-semibold text-accent")}>{v == null || (typeof v === "number" && !Number.isFinite(v)) ? "—" : c.f(v)}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="grid gap-4 2xl:grid-cols-2">
        <Panel title={`月度收益 · ${focus}`} subtitle="单位 %" bodyClass="p-2"><EChart option={calOption} height={Math.max(220, years.length * 30 + 40)} /></Panel>
        <Panel title="逐年收益" bodyClass="p-2"><EChart option={yearOption} height={Math.max(220, years.length * 30 + 40)} /></Panel>
      </div>
      {!res.offline && (
        <div className="grid gap-4 2xl:grid-cols-2">
          <Panel title="交易成本与成交约束" subtitle="年化成本拆分；被阻挡订单的次数" bodyClass="p-3 text-[11.5px]">
            <table className="w-full">
              <thead className="text-faint"><tr><th className="text-left font-medium">策略</th><th className="text-right font-medium">佣金</th><th className="text-right font-medium">印花税</th><th className="text-right font-medium">冲击</th><th className="pl-3 text-left font-medium">被阻挡</th></tr></thead>
              <tbody>
                {names.map((k) => {
                  const cb = S[k].metrics.cost_breakdown_ann ?? {};
                  const bl = S[k].metrics.blocked_orders ?? {};
                  return (
                    <tr key={k} className="border-t border-line/60">
                      <td className="py-1.5">{k}</td>
                      <td className="num text-right">{pct(cb.commission ?? 0, 3)}</td>
                      <td className="num text-right">{pct(cb.stamp ?? 0, 3)}</td>
                      <td className="num text-right">{pct(cb.impact ?? 0, 3)}</td>
                      <td className="pl-3">{Object.entries(bl).map(([a, n]) => <Badge key={a} color={a === "流动性" ? "var(--accent)" : "var(--warn)"}>{a} {n}</Badge>)}{!Object.keys(bl).length && "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Panel>
          <Panel title={`最新持仓 · ${focus}`} subtitle={`调仓日 ${res.end} 前最后一次`} bodyClass="p-3">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11.5px]">
              {(S[focus].last_weights ?? []).map((w) => (
                <div key={w.code} className="flex justify-between"><span className="truncate">{w.name}</span><span className="num text-muted">{pct(w.w, 1)}</span></div>
              ))}
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
