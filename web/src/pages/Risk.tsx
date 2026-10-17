import clsx from "clsx";
import { useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { Lock } from "lucide-react";
import { useAssistant } from "../lib/assistant";
import { useData, type Portfolio } from "../lib/data";
import { modelOptions, paramsFrom, useDefacoAvailable, useLivePortfolio, usePortfolioSettings } from "../lib/portfolio";
import { cssVar } from "../lib/store";
import { pct } from "../lib/format";
import { EChart } from "../components/charts/EChart";
import { Badge, DataBanner, Empty, InfinityLoader, PageHeader, Panel, Segmented, Stat } from "../components/ui";

const CHARS = ["size", "bm", "ep", "mom_12_2", "rev_1m", "vol_1m", "turn_1m", "illiq", "beta_12m", "ivol_3m", "roe", "asset_growth"];

interface GammaShare { chars: string[]; m4: number[]; m3: (number | null)[]; fast: string[] }

export default function Risk() {
  const { data } = useData<Portfolio>("portfolio.json"); // model labels + offline fallback
  const settings = usePortfolioSettings();
  const defacoOn = useDefacoAvailable();
  const { model, cap, indCap, set } = settings;
  const live = useLivePortfolio(paramsFrom(settings, model));
  const [gammaRun, setGammaRun] = useState<"m4" | "m3">("m4");
  const m = live.data;
  const setCtx = useAssistant((st) => st.setCtx);
  useEffect(
    () => setCtx({ model, portfolio: m ? { max_weight: cap / 100, industry_cap: indCap / 100, universe_size: m.universe_size, live: live.source === "live" } : undefined }),
    [model, cap, indCap, m, live.source, setCtx],
  );
  if (!m) {
    if (live.error) return <div className="panel grid h-96 place-items-center text-sm text-warn">{live.error}</div>;
    return <div className="grid h-96 place-items-center"><InfinityLoader label="计算风险" /></div>;
  }
  const fac = m.risk.factor_share;
  const indShare = m.risk.industry_share ?? 0;

  const waterfall: EChartsOption = {
    grid: { left: 50, right: 16, top: 20, bottom: 28 },
    tooltip: { trigger: "axis", valueFormatter: (v) => pct(v as number, 1) },
    xAxis: { type: "category", data: ["因子风险", "行业块风险", "特质风险", "总风险"] },
    yAxis: { type: "value", max: 1, axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: [
      { type: "bar", stack: "w", itemStyle: { color: "transparent" }, data: [0, fac ?? 0, (fac ?? 0) + indShare, 0], tooltip: { show: false } },
      {
        type: "bar",
        stack: "w",
        barWidth: "46%",
        data: [
          { value: fac ?? 0, itemStyle: { color: cssVar("--m-cil"), borderRadius: 4 } },
          { value: indShare, itemStyle: { color: cssVar("--m-sic"), borderRadius: 4 } },
          { value: fac != null ? 1 - fac - indShare : 0, itemStyle: { color: cssVar("--faint"), borderRadius: 4 } },
          { value: 1, itemStyle: { color: cssVar("--m-dpl"), borderRadius: 4 } },
        ],
        label: { show: true, position: "top", color: cssVar("--muted"), fontSize: 11, formatter: (p) => pct(p.value as number, 1) },
      },
    ],
  };

  const exp = m.risk.exposures ?? [];
  const expOption: EChartsOption = {
    grid: { left: 40, right: 16, top: 16, bottom: 26 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: exp.map((_, i) => `F${i + 1}`) },
    yAxis: { type: "value" },
    series: [{ type: "bar", barWidth: "45%", data: exp.map((v) => ({ value: v, itemStyle: { color: v >= 0 ? cssVar("--m-cil") : cssVar("--m-sic"), borderRadius: 3 } })) }],
  };

  // DeFaCo: characteristic exposures (ranks within the universe, -0.5..0.5) vs the NLS portfolio
  const ce = model === "defaco" ? m.char_exposures : undefined;
  const ceOption: EChartsOption | null = ce
    ? {
        grid: { left: 86, right: 16, top: 26, bottom: 20 },
        legend: { top: 0, right: 0, textStyle: { color: cssVar("--muted"), fontSize: 10 }, itemWidth: 10, itemHeight: 8 },
        tooltip: { trigger: "axis", valueFormatter: (v) => (v as number).toFixed(3) },
        xAxis: { type: "value", min: -0.4, max: 0.4 },
        yAxis: { type: "category", data: ce.names, inverse: true, axisLabel: { fontSize: 10 } },
        series: [
          { name: m.label, type: "bar", data: ce.values, itemStyle: { color: cssVar("--m-cil"), borderRadius: 2 }, barGap: "10%" },
          ...(ce.compare ? [{ name: ce.compare.label, type: "bar" as const, data: ce.compare.values, itemStyle: { color: cssVar("--faint"), borderRadius: 2 } }] : []),
        ],
      }
    : null;

  // DeFaCo: weight of each characteristic in the loading map, diag(ΓΓ')/tr(ΓΓ'), averaged over seeds
  const gs = model === "defaco" ? (m.explain?.gamma_share as GammaShare | undefined) : undefined;
  const gVals = gs ? (gammaRun === "m3" && gs.m3[0] != null ? gs.m3 : gs.m4) : [];
  const gOrder = gs ? gs.chars.map((c, i) => ({ c, v: (gVals[i] as number) ?? 0 })).sort((a, b) => b.v - a.v) : [];
  const gammaOption: EChartsOption | null = gs
    ? {
        grid: { left: 86, right: 40, top: 8, bottom: 20 },
        tooltip: { trigger: "axis", valueFormatter: (v) => pct(v as number, 1) },
        xAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
        yAxis: { type: "category", data: gOrder.map((x) => x.c), inverse: true, axisLabel: { fontSize: 10 } },
        series: [{
          type: "bar", barWidth: "62%",
          data: gOrder.map((x) => ({ value: x.v, itemStyle: { color: gs.fast.includes(x.c) ? cssVar("--m-dfl") : cssVar("--m-cil"), borderRadius: 2 } })),
          label: { show: true, position: "right", fontSize: 9, color: cssVar("--muted"), formatter: (p) => pct(p.value as number, 1) },
        }],
      }
    : null;
  const fastShare = gs ? gs.chars.reduce((acc, c, i) => acc + (gs.fast.includes(c) ? ((gVals[i] as number) ?? 0) : 0), 0) : 0;

  const C = m.corr;
  const heat: EChartsOption = {
    grid: { left: 70, right: 70, top: 8, bottom: 70 },
    tooltip: {
      formatter: (p: unknown) => {
        const v = (p as { value: [number, number, number] }).value;
        return `${C.names[v[0]]} × ${C.names[v[1]]}<br/>相关系数 <b>${v[2].toFixed(2)}</b>`;
      },
    },
    xAxis: { type: "category", data: C.names, axisLabel: { rotate: 70, fontSize: 9, interval: 0 }, splitLine: { show: false } },
    yAxis: { type: "category", data: C.names, axisLabel: { fontSize: 9, interval: 0 }, splitLine: { show: false } },
    visualMap: {
      min: -0.2,
      max: 1,
      calculable: true,
      orient: "vertical",
      right: 0,
      top: "middle",
      itemHeight: 140,
      textStyle: { color: cssVar("--muted"), fontSize: 10 },
      inRange: { color: [cssVar("--panel-2"), "#3b4f80", cssVar("--m-sic"), "#ffd29b"] },
    },
    series: [
      {
        type: "heatmap",
        data: C.matrix.flatMap((row, i) => row.map((v, j) => [i, j, v])),
        progressive: 0,
      },
    ],
  };
  // industry boundaries for the block structure
  const bounds: number[] = [];
  C.industries.forEach((ind, i) => i > 0 && ind !== C.industries[i - 1] && bounds.push(i));

  const rcTop = [...m.weights].sort((a, b) => b.rc - a.rc).slice(0, 10);
  return (
    <div>
      <PageHeader
        title="风险解析"
        desc={
          <>
            组合风险从哪里来：因子、行业联动还是个股特质 · 当前组合：{m.label} · 单股上限 {pct(m.constraints.max_weight, 0)} · 行业上限{" "}
            {m.constraints.industry_cap != null ? pct(m.constraints.industry_cap, 0) : "无"} · {m.universe_source === "custom" ? "候选池" : "沪深300 前100"} {m.universe_size} 只
            <span className={clsx("ml-2 text-[11px]", live.source === "live" ? "text-accent" : "text-warn")} data-testid="optimize-status">
              {live.pending ? "求解中…" : live.source === "live" ? `实时 · ${m.timing_ms.total} ms` : "离线 · 预计算"}
            </span>
          </>
        }
        actions={<Segmented name="risk-model" value={model} onChange={(v) => set({ model: v })} options={modelOptions(data?.models ?? { [model]: m }, defacoOn)} />}
      />
      <DataBanner note={data?.meta.survivorship_biased ? data.meta.note : undefined} />
      {live.error && <p className="mb-3 text-[11px] text-warn">{live.error}（下方仍显示上一次的结果）</p>}
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="预期年化波动" value={pct(m.risk.total_vol)} />
        <Stat label="因子风险占比" value={fac != null ? pct(fac, 1) : "—"} hint={fac == null ? "稠密协方差无分解" : undefined} />
        <Stat label="最大单票风险贡献" value={pct(rcTop[0]?.rc, 1)} hint={rcTop[0]?.name} />
        <Stat label="风险贡献集中度" value={(1 / m.weights.reduce((s, x) => s + x.rc * x.rc, 0)).toFixed(1)} hint="有效风险来源数" />
      </div>
      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="风险拆解" subtitle="总风险 = 因子 + 行业块 + 特质（模块色与论文图一致）" accent="var(--m-cil)">
          {fac != null ? <EChart option={waterfall} height={260} /> : <Empty title="当前模型为稠密协方差" desc="切换到「DeFaCo」或「PCA 因子模型」查看风险拆解。" />}
          <p className="mt-2 text-[11px] text-faint">{model === "defaco" ? "行业块风险来自 DeFaCo 的 SIC 结构化残差（行业内等相关）。" : "行业块风险需要 SIC 模块（DeFaCo），其他模型中为 0。"}</p>
        </Panel>
        <Panel title={ceOption ? "特征暴露" : "因子暴露"} subtitle={ceOption ? "组合在 24 个特征（股票池内秩，−0.5…0.5）上的加权平均，与 NLS 组合对比" : "组合对 PCA 因子的暴露 B′w"} accent="var(--m-cil)">
          {ceOption ? <EChart option={ceOption} height={420} /> : exp.length ? <EChart option={expOption} height={260} /> : <Empty title="无因子暴露" desc="稠密协方差模型没有因子结构。" />}
        </Panel>
        <Panel title="风险贡献 Top 10" subtitle="对组合方差的边际贡献 × 权重">
          <div className="space-y-2">
            {rcTop.map((r) => (
              <div key={r.code} className="text-xs">
                <div className="mb-1 flex justify-between">
                  <span>{r.name} <span className="text-faint">{r.industry}</span></span>
                  <span className="num">{pct(r.rc, 1)}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-panel-2">
                  <div className="h-full rounded-full" style={{ width: `${(r.rc / rcTop[0].rc) * 100}%`, background: "var(--m-sic)" }} />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-5">
        <Panel
          className="xl:col-span-3"
          title="相关结构热力图"
          subtitle={`前 40 大持仓按行业排序（${m.label} 协方差）：对角线附近的亮块就是因子之外的行业联动，这正是 SIC 模块要刻画的结构`}
          accent="var(--m-sic)"
          actions={<Badge color="var(--m-sic)">{bounds.length + 1} 个行业块</Badge>}
          bodyClass="p-2"
        >
          <EChart option={heat} height={520} />
        </Panel>
        <Panel
          className="xl:col-span-2"
          title="Γ 特征权重"
          subtitle="载荷映射 Γ 中各特征的权重 diag(ΓΓ′)/tr(ΓΓ′)，5 个种子平均（旋转不变）；红色为 5 个快变特征"
          accent="var(--m-cil)"
          actions={gs ? <Segmented name="gamma-run" value={gammaRun} onChange={(v) => setGammaRun(v as "m4" | "m3")} options={[{ value: "m4", label: "M4（含换手项）" }, { value: "m3", label: "M3（κ = 0）" }]} /> : undefined}
        >
          {gammaOption ? (
            <>
              <EChart option={gammaOption} height={520} />
              <p className="mt-1 text-[11px] text-faint">
                {String(m.explain.model_year)} 年模型 · K = {String(m.explain.K)} · κ = {String(m.explain.kappa_bp)} bp · 快变特征合计 {pct(fastShare, 1)}（不含常数项）。换手项压低快变特征的权重，使风险结构更平滑（论文 H4b）。
              </p>
            </>
          ) : (
            <div className="relative">
              <div className="grid grid-cols-[90px_repeat(5,1fr)] gap-1 opacity-40 blur-[1px]">
                <div />
                {[1, 2, 3, 4, 5].map((k) => <div key={k} className="text-center text-[10px] text-faint">F{k}</div>)}
                {CHARS.map((c, i) => (
                  <div key={c} className="contents">
                    <div className="truncate text-[11px] text-muted">{c}</div>
                    {[0, 1, 2, 3, 4].map((k) => {
                      const v = Math.sin(i * 1.7 + k * 2.3);
                      return <div key={k} className="h-5 rounded" style={{ background: v > 0 ? `rgba(59,130,246,${Math.abs(v)})` : `rgba(224,123,42,${Math.abs(v)})` }} />;
                    })}
                  </div>
                ))}
              </div>
              <div className="absolute inset-0 grid place-items-center">
                {defacoOn ? (
                  <button type="button" onClick={() => set({ model: "defaco" })} className="panel flex max-w-[260px] flex-col items-center gap-2 p-4 text-center">
                    <Lock size={18} className="text-cil" />
                    <div className="text-sm font-semibold">切换到 DeFaCo 查看 Γ</div>
                    <div className="text-xs text-muted">Γ 只属于论文模型 DeFaCo（特征驱动载荷）；当前模型没有特征载荷。</div>
                  </button>
                ) : (
                  <div className="panel flex max-w-[280px] flex-col items-center gap-2 p-4 text-center">
                    <Lock size={18} className="text-cil" />
                    <div className="text-sm font-semibold">DeFaCo 参数未包含在开源版本中</div>
                    <div className="text-xs text-muted">训练好的 DeFaCo 参数属于论文研究成果，将在论文答辩后发布；放入 data/models/defaco_m4.json 后此处显示真实的 Γ。</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
