import clsx from "clsx";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ShieldHalf, X, Zap, ZapOff } from "lucide-react";
import type { EChartsOption } from "echarts";
import { useAssistant } from "../lib/assistant";
import { useData, type Portfolio as PortfolioData } from "../lib/data";
import { modelOptions, paramsFrom, useDefacoAvailable, useLivePortfolio, usePortfolioSettings } from "../lib/portfolio";
import { cssVar } from "../lib/store";
import { pct } from "../lib/format";
import { EChart } from "../components/charts/EChart";
import { AiButton, Badge, DataBanner, InfinityLoader, PageHeader, Panel, RimButton, Segmented, Stat, Switch } from "../components/ui";
import { useUI } from "../lib/store";

const PALETTE = ["#4c8dff", "#22c3c3", "#f5a524", "#b37feb", "#e0524a", "#2e8b57", "#e07b2a", "#8b6cc7", "#d4a017", "#5b8def", "#7a8699"];

const OBJECTIVES = [
  { value: "minvar", label: "最小方差" },
  { value: "rp", label: "风险平价" },
  { value: "hrp", label: "HRP" },
  { value: "mv", label: "均值-方差" },
] as const;

export default function Portfolio() {
  const { data } = useData<PortfolioData>("portfolio.json"); // model labels + offline fallback
  const settings = usePortfolioSettings();
  const defacoOn = useDefacoAvailable();
  const { model, objective, cap, indCap, pool, usePool, set } = settings;
  const [compareTo, setCompareTo] = useState("lw_nonlinear");
  const [compare, setCompare] = useState(true);
  const nav = useNavigate();
  const toggleAi = useUI((s) => s.toggleAi);

  const live = useLivePortfolio(paramsFrom(settings, model));
  const cmp = useLivePortfolio(compare && compareTo !== model ? paramsFrom(settings, compareTo) : null);
  const m = live.data;
  const b = compare && compareTo !== model ? cmp.data : null;

  const setCtx = useAssistant((st) => st.setCtx);
  useEffect(
    () => setCtx({ model, portfolio: m ? { max_weight: cap / 100, industry_cap: indCap / 100, universe_size: m.universe_size, live: live.source === "live" } : undefined }),
    [model, cap, indCap, m, live.source, setCtx],
  );

  const merged = useMemo(() => {
    if (!m) return [];
    const bw = new Map((b?.weights ?? []).map((x) => [x.code, x.w]));
    const rows = m.weights.map((x) => ({ ...x, wb: bw.get(x.code) ?? 0 }));
    (b?.weights ?? []).forEach((x) => {
      if (!rows.find((r) => r.code === x.code)) rows.push({ ...x, w: 0, rc: 0, wb: x.w });
    });
    return rows.sort((p, q) => q.w - p.w);
  }, [m, b]);

  if (!m) {
    if (live.error) return <div className="panel grid h-96 place-items-center text-sm text-warn">{live.error}</div>;
    return <div className="grid h-96 place-items-center"><InfinityLoader label="求解多头约束 QP…" /></div>;
  }

  const offline = live.source === "fallback";
  const customised = cap !== 10 || objective !== "minvar" || (usePool && pool.length > 0);

  const indEntries = Object.entries(m.industry_weights);
  const donut: EChartsOption = {
    tooltip: { trigger: "item", valueFormatter: (v) => pct(v as number, 1) },
    series: [
      {
        type: "pie",
        radius: ["52%", "78%"],
        center: ["50%", "50%"],
        itemStyle: { borderColor: cssVar("--panel"), borderWidth: 2, borderRadius: 4 },
        label: { show: false },
        data: indEntries.map(([k, v], i) => ({ name: k, value: v, itemStyle: { color: PALETTE[i % PALETTE.length] } })),
      },
    ],
  };

  const top = merged.slice(0, 20).reverse();
  const bars: EChartsOption = {
    grid: { left: 70, right: 16, top: 26, bottom: 20 },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 8 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v) => pct(v as number, 2) },
    xAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    yAxis: { type: "category", data: top.map((r) => r.name), axisLabel: { fontSize: 11 } },
    series: [
      { name: `权重 · ${m.label}`, type: "bar", data: top.map((r) => r.w), itemStyle: { color: cssVar("--m-dpl"), borderRadius: [0, 4, 4, 0] }, barGap: "10%" },
      { name: "风险贡献", type: "bar", data: top.map((r) => r.rc), itemStyle: { color: cssVar("--m-sic"), borderRadius: [0, 4, 4, 0] } },
      ...(compare && b ? [{ name: `权重 · ${b.label}`, type: "bar" as const, data: top.map((r) => r.wb), itemStyle: { color: cssVar("--faint"), borderRadius: [0, 4, 4, 0] } }] : []),
    ],
  };

  const models = modelOptions(data?.models ?? { [model]: m }, defacoOn);
  const universeDesc = m.universe_source === "custom" ? `选股器候选池（可投资 ${m.universe_size} 只）` : `沪深300 成分股中流通市值前 100（可投资 ${m.universe_size} 只）`;
  return (
    <div>
      <PageHeader
        title="组合构建"
        desc={`股票池：${universeDesc} · 调仓日 ${m.asof} · 250 日估计窗口`}
        actions={
          <>
            <AiButton onClick={() => toggleAi(true)}>AI 解读配置</AiButton>
            <RimButton onClick={() => nav("/risk")}>
              <ShieldHalf size={14} /> 风险解析
            </RimButton>
          </>
        }
      />
      <DataBanner note={data?.meta.survivorship_biased ? data.meta.note : undefined} />
      <DataBanner note={offline && customised ? "组合 API 未连接：下方是预计算结果（最小方差 · 单股上限 10% · 默认股票池），调整的参数暂未生效" : undefined} />

      {/* ---------------- control bar ---------------- */}
      <div className="panel mb-4 flex flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">风险模型</span>
          <Segmented name="model" value={model} onChange={(v) => set({ model: v })} options={models} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">目标</span>
          <Segmented name="obj" value={objective} onChange={(v) => set({ objective: v })} options={OBJECTIVES.map((o) => ({ value: o.value, label: o.label }))} />
        </div>
        <label className="flex items-center gap-2 text-xs text-muted">
          单股上限
          <input type="range" name="cap" min={3} max={20} value={cap} onChange={(e) => set({ cap: +e.target.value })} className="w-24 accent-[var(--accent)]" />
          <span className="num w-9 text-ink">{cap}%</span>
        </label>
        {objective === "mv" && (
          <label className="flex items-center gap-2 text-xs text-muted" title="风险厌恶系数 γ：越大越接近最小方差，越小越追逐预期收益">
            风险厌恶 γ
            <input type="range" min={1} max={50} value={settings.gamma} onChange={(e) => set({ gamma: +e.target.value })} className="w-20 accent-[var(--accent)]" />
            <span className="num w-6 text-ink">{settings.gamma}</span>
          </label>
        )}
        <label className={clsx("flex items-center gap-2 text-xs text-muted", (objective === "rp" || objective === "hrp") && "opacity-40")}>
          行业上限
          <input type="range" name="indCap" min={15} max={60} value={indCap} onChange={(e) => set({ indCap: +e.target.value })} className="w-24 accent-[var(--accent)]" />
          <span className="num w-9 text-ink">{indCap}%</span>
        </label>
        {pool.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">股票池</span>
            <Segmented
              name="universe"
              value={usePool ? "pool" : "default"}
              onChange={(v) => set({ usePool: v === "pool" })}
              options={[{ value: "default", label: "沪深300 前100" }, { value: "pool", label: `候选池 ${pool.length}` }]}
            />
            <button title="清空候选池" className="text-faint hover:text-ink" onClick={() => set({ pool: [], usePool: false })}><X size={14} /></button>
          </div>
        )}
        <div className="ml-auto flex items-center gap-1.5 text-[11px]" data-testid="optimize-status">
          {live.pending ? (
            <span className="text-muted">求解中…</span>
          ) : offline ? (
            <span className="flex items-center gap-1 text-warn"><ZapOff size={12} /> 离线 · 预计算结果</span>
          ) : (
            <span className="flex items-center gap-1 text-accent"><Zap size={12} /> 实时求解 · {m.timing_ms.total} ms</span>
          )}
        </div>
        {live.error && <p className="w-full text-[11px] text-warn">{live.error}（下方仍显示上一次的结果）</p>}
        {!live.error && (m.notes?.length ?? 0) > 0 && <p className="w-full text-[11px] text-faint">注：{m.notes!.join("；")}</p>}
        {!live.error && m.dropped.length > 0 && (
          <p className="w-full text-[11px] text-faint">候选池中 {m.dropped.length} 只因停牌 / ST / 上市不足一年 / 无行情被剔除：{m.dropped.slice(0, 8).join("、")}{m.dropped.length > 8 ? " …" : ""}</p>
        )}
      </div>

      <div className={clsx("transition-opacity", live.pending && "opacity-60")}>
        <>
          {/* ---------------- KPI comparison ---------------- */}
          <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="预期年化波动" value={pct(m.risk.total_vol)} hint={compare && b ? `对比 ${pct(b.risk.total_vol)}` : undefined} />
            <Stat label="持仓数" value={m.n_holdings} hint={compare && b ? `对比 ${b.n_holdings}` : undefined} />
            <Stat label="有效持仓数 1/HHI" value={(1 / m.hhi).toFixed(1)} hint={compare && b ? `对比 ${(1 / b.hhi).toFixed(1)}` : undefined} />
            <Stat label="最大权重" value={pct(m.weights[0]?.w ?? 0, 1)} hint={compare && b ? `对比 ${pct(b.weights[0]?.w ?? 0, 1)}` : undefined} />
            <Stat label="因子风险占比" value={m.risk.factor_share != null ? pct(m.risk.factor_share, 1) : "—"} hint={m.risk.factor_share == null ? "稠密模型无分解" : m.risk.industry_share != null ? `行业块 ${pct(m.risk.industry_share, 1)}` : "PCA 因子"} />
            <Stat label="第一大行业" value={<span className="text-base">{indEntries[0]?.[0]}</span>} hint={pct(indEntries[0]?.[1], 1)} />
          </div>

          <div className="grid gap-4 xl:grid-cols-3">
            <Panel
              title="权重与风险贡献 Top 20"
              subtitle="紫色 = 权重，橙色 = 对组合方差的贡献；灰色 = 对比模型的权重"
              className="xl:col-span-2"
              accent="var(--m-dpl)"
              actions={
                <div className="flex items-center gap-3">
                  <Switch checked={compare} onChange={setCompare} label="对比" />
                  {compare && (
                    <select
                      value={compareTo}
                      onChange={(e) => setCompareTo(e.target.value)}
                      className="h-7 rounded-lg border border-line bg-bg-2 px-2 text-xs outline-none"
                    >
                      {models.filter((x) => x.value !== model).map((x) => (
                        <option key={x.value} value={x.value}>{x.label}</option>
                      ))}
                    </select>
                  )}
                </div>
              }
            >
              <EChart option={bars} height={480} />
            </Panel>
            <Panel title="行业分布" subtitle="按申万一级行业汇总权重">
              <EChart option={donut} height={260} />
              <div className="mt-3 space-y-1.5">
                {indEntries.slice(0, 6).map(([k, v], i) => (
                  <div key={k} className="flex items-center gap-2 text-xs">
                    <i className="h-2 w-2 rounded-full" style={{ background: PALETTE[i % PALETTE.length] }} />
                    <span className="flex-1 text-muted">{k}</span>
                    <div className="h-1.5 w-24 overflow-hidden rounded-full bg-panel-2">
                      <div className="h-full rounded-full" style={{ width: `${Math.min(100, (v / indEntries[0][1]) * 100)}%`, background: PALETTE[i % PALETTE.length] }} />
                    </div>
                    <span className={clsx("num w-12 text-right", v * 100 > indCap ? "text-warn" : "")}>{pct(v, 1)}</span>
                  </div>
                ))}
              </div>
            </Panel>
          </div>

          <Panel className="mt-4" title="持仓明细" subtitle={compare && b ? `Δw = ${m.label} − ${b.label}` : undefined} actions={<Badge color="var(--m-dpl)">{merged.filter((r) => r.w > 0).length} 只</Badge>} bodyClass="p-0">
            <div className="max-h-[420px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-panel text-left text-faint">
                  <tr className="border-b border-line">
                    <th className="px-4 py-2 font-medium">代码</th>
                    <th className="px-2 py-2 font-medium">名称</th>
                    <th className="px-2 py-2 font-medium">行业</th>
                    <th className="px-2 py-2 text-right font-medium">权重</th>
                    <th className="px-2 py-2 text-right font-medium">风险贡献</th>
                    {compare && <th className="px-4 py-2 text-right font-medium">Δw</th>}
                  </tr>
                </thead>
                <tbody>
                  {merged.map((r) => {
                    const d = r.w - r.wb;
                    return (
                      <tr key={r.code} className="border-b border-line/60 hover:bg-panel-2/60">
                        <td className="num px-4 py-1.5 text-muted">{r.code}</td>
                        <td className="px-2 py-1.5">{r.name}</td>
                        <td className="px-2 py-1.5 text-muted">{r.industry}</td>
                        <td className="num px-2 py-1.5 text-right">{pct(r.w)}</td>
                        <td className="num px-2 py-1.5 text-right text-muted">{pct(r.rc)}</td>
                        {compare && (
                          <td className={clsx("num px-4 py-1.5 text-right", d > 0 ? "text-up" : d < 0 ? "text-down" : "text-faint")}>
                            {d === 0 ? "—" : `${d > 0 ? "+" : ""}${(d * 100).toFixed(2)}%`}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      </div>
    </div>
  );
}
