import clsx from "clsx";
import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { CircleCheck, CircleDashed, CircleDot } from "lucide-react";
import { useData } from "../lib/data";
import { cssVar } from "../lib/store";
import { EChart } from "../components/charts/EChart";
import { Badge, InfinityLoader, PageHeader, Panel, Segmented } from "../components/ui";

type Cell = { SD?: number | null; SR?: number | null; TO?: number | null };
type Grid = Record<string, Record<string, Cell>>;
interface Lab {
  ladder: { id: string; name: string; desc: string; status: "done" | "next" | "todo" }[];
  findings: Record<string, string[]>;
  setting: Record<string, string>;
  kinds: Record<string, { bench: Record<string, Cell>; grid: Grid; ipca_tests_sd: Record<string, Record<string, string>> }>;
  monthly: Record<string, { bench: Record<string, Cell>; grid: Grid }>;
  n_assets: { models: string[]; rows: { N: number; model: string; SD: number; SR: number; TO: number }[] };
  residual_corr: Record<string, Record<string, number>>;
  ipca_variants: { columns: string[]; rows: Record<string, (number | string | null)[]> };
  figures: Record<string, string>;
}

const MODULES = [
  { k: "CIL", c: "var(--m-cil)", t: "特征驱动的混频载荷" },
  { k: "FCE", c: "var(--m-fce)", t: "因子协方差（静态 SFC）" },
  { k: "SIC", c: "var(--m-sic)", t: "行业块结构残差" },
  { k: "DPL", c: "var(--m-dpl)", t: "可微组合层" },
  { k: "DFL", c: "var(--m-dfl)", t: "决策导向、换手感知学习" },
];
const METHOD_CN: Record<string, string> = {
  pca: "PCA", pls: "PLS", spca: "SPCA", spls: "SPLS", aen: "AEN", dae: "DAE",
  pca_raw: "PCA·原始", pls_raw: "PLS·原始", spca_raw: "SPCA·原始", spls_raw: "SPLS·原始", aen_raw: "AEN·原始", dae_raw: "DAE·原始", ipca: "IPCA（M1）", ipca_plus: "IPCA+（M1+）", ipca_plus_s3: "IPCA+·平滑（M1+）", m2: "IPCA+ · SIC（M2）", m2_pca: "PCA·原始 · SIC（M2）", m3_init: "M3 初始点 Θ⁰（未训练）", m3: "DeFaCo 端到端（M3）", m4: "DeFaCo 换手感知（M4）",
};
const KIND_CN: Record<string, string> = { mv: "多头最小方差（式 7）", gmv: "允许卖空 GMV（式 8）", tp: "换手惩罚（式 9）" };
const METRIC_CN = { SD: "月度 SD %（越低越好）", SR: "月度 Sharpe（越高越好）", TO: "月均换手 %（越低越好）" } as const;

function ResultGrid({ grid, bench, metric, specs }: { grid: Grid; bench: Record<string, Cell>; metric: keyof typeof METRIC_CN; specs: string[] }) {
  const all = Object.values(grid).flatMap((r) => specs.map((s) => r[s]?.[metric])).filter((v): v is number => v != null);
  const lo = Math.min(...all), hi = Math.max(...all);
  const better = metric === "SR" ? "high" : "low";
  const tone = (v: number | null | undefined) => {
    if (v == null) return undefined;
    const x = hi > lo ? (v - lo) / (hi - lo) : 0.5;
    const good = better === "high" ? x : 1 - x;
    return `color-mix(in srgb, var(--m-fce) ${Math.round(good * 45)}%, transparent)`;
  };
  const fmt = (v: number | null | undefined) => (v == null ? "—" : metric === "SR" ? v.toFixed(3) : v.toFixed(metric === "TO" ? 1 : 2));
  const best = better === "high" ? hi : lo;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11.5px]">
        <thead className="text-faint">
          <tr className="border-b border-line">
            <th className="px-3 py-1.5 text-left font-medium">方法</th>
            {specs.map((s) => <th key={s} className="px-3 py-1.5 text-right font-medium">{s.toUpperCase()}</th>)}
          </tr>
        </thead>
        <tbody>
          {Object.keys(METHOD_CN).filter((m) => grid[m]).map((m) => (
            <tr key={m} className={clsx("border-b border-line/50", m === "ipca" && "border-t-2 border-t-line", (m.startsWith("ipca_plus") || m.startsWith("m2") || m.startsWith("m3")) && "font-medium", (m === "m2" || m === "m3_init") && "border-t-2 border-t-line", (m === "m3" || m === "m4") && "text-accent", m.endsWith("_raw") && "text-muted")}>
              <td className="px-3 py-1">{METHOD_CN[m]}</td>
              {specs.map((s) => {
                const v = grid[m]?.[s]?.[metric];
                return (
                  <td key={s} className={clsx("num px-3 py-1 text-right", v === best && "font-semibold text-accent")} style={{ background: tone(v) }}>{fmt(v)}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex flex-wrap gap-1.5 px-3 text-[11px]">
        <span className="text-faint">基准：</span>
        {Object.entries(bench).map(([k, v]) => <span key={k} className="num rounded border border-line px-1.5">{k} {fmt(v[metric])}</span>)}
      </div>
    </div>
  );
}

const FINDING_SUB: Record<string, string> = {
  M0: "复现 Conlon, Cotter & Kynigakis (2025, EJOR)",
  M1: "IPCA 特征载荷 vs 收益潜在因子",
  "M1+": "特征载荷 + 自由载荷（修正因子爆炸后）",
  M2: "SIC 行业块结构残差（两阶段）",
  M3: "端到端决策导向训练（κ = 0）",
  M4: "换手感知训练：完整 DeFaCo（κ > 0）",
};

export default function Lab() {
  const { data, error } = useData<Lab>("lab.json");
  const [fig, setFig] = useState<"framework" | "workflow">("framework");
  const [kind, setKind] = useState<"mv" | "gmv" | "tp">("gmv");
  const [metric, setMetric] = useState<keyof typeof METRIC_CN>("SD");
  const [nMetric, setNMetric] = useState<"SD" | "SR" | "TO">("SD");

  const nOption: EChartsOption | null = useMemo(() => {
    if (!data) return null;
    const Ns = [...new Set(data.n_assets.rows.map((r) => r.N))].sort((a, b) => a - b);
    const label = (m: string) => ({ EW: "EW", lw_linear: "LINS", lw_nonlinear: "NLS" }[m] ?? METHOD_CN[m.split("-")[0]] ?? m);
    const pal = ["#7a8699", "#9aa3b2", "#c2c8d2", "#4c8dff", "#22c3c3", "#f5a524", "#b37feb", "#e07b2a", "#2e8b57", "#5b8def", "#d4a017", "#e0524a"];
    return {
      grid: { left: 44, right: 12, top: 50, bottom: 24 },
      legend: { top: 0, type: "scroll" },
      tooltip: { trigger: "axis", valueFormatter: (v) => (v as number).toFixed(3) },
      xAxis: { type: "category", data: Ns.map(String), name: "N" },
      yAxis: { type: "value", scale: true },
      series: data.n_assets.models.map((m, i) => {
        const isB = ["EW", "lw_linear", "lw_nonlinear"].includes(m);
        const isM1 = m.startsWith("ipca");
        return { name: label(m), type: "line" as const, symbolSize: 5, data: Ns.map((n) => data.n_assets.rows.find((r) => r.N === n && r.model === m)?.[nMetric] ?? null),
          lineStyle: { width: isM1 ? 3 : 1.3, type: isB ? ("dashed" as const) : ("solid" as const), color: isM1 ? cssVar("--text") : pal[i % pal.length] },
          itemStyle: { color: isM1 ? cssVar("--text") : pal[i % pal.length] } };
      }),
    };
  }, [data, nMetric]);

  const rcOption: EChartsOption | null = data ? {
    grid: { left: 150, right: 20, top: 26, bottom: 20 },
    legend: { top: 0 },
    tooltip: { trigger: "axis", valueFormatter: (v) => (v as number).toFixed(3) },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: Object.keys(data.residual_corr) },
    series: ["平均绝对相关", "行业内", "行业间"].map((k, i) => ({ name: k, type: "bar" as const, data: Object.values(data.residual_corr).map((v) => v[k]),
      itemStyle: { color: [cssVar("--accent"), cssVar("--m-sic"), cssVar("--warn")][i] } })),
  } : null;

  if (error && !data) {
    return (
      <div>
        <PageHeader title="论文实验室" desc="DeFaCo 毕业论文的方法结构、消融阶梯与实验结果" />
        <div className="panel grid min-h-[320px] place-items-center p-8 text-center">
          <div className="max-w-xl space-y-2">
            <div className="text-base font-semibold">论文实验结果未包含在开源版本中</div>
            <p className="text-sm text-muted">论文实验室展示毕业论文的实验结果和方法图，这些内容将在论文答辩后发布。其余页面（行情、选股、因子研究、组合构建、风险解析、调仓、回测、监控、AI 助手）都可以正常使用。</p>
            <p className="text-xs text-faint">已有论文结果时，把 lab.json 放到 web/public/demo-data/、方法图放到 web/public/figures/ 即可显示。</p>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div>
      <PageHeader title="论文实验室" desc="DeFaCo：决策导向、换手感知的特征因子协方差 —— 方法结构、消融阶梯与全部实验（M0–M4）的结果" />
      <div className="grid gap-4 xl:grid-cols-3">
        <Panel className="xl:col-span-2" title="方法结构"
          actions={<Segmented name="fig" value={fig} onChange={setFig} options={[{ value: "framework", label: "总体框架" }, { value: "workflow", label: "滚动方案与消融" }]} />} bodyClass="p-3">
          <div className="overflow-hidden rounded-xl bg-white p-2">
            <img src={`/figures/fig${fig === "framework" ? "1_framework" : "4_protocol"}.png`} alt="DeFaCo 方法结构图" className="w-full" />
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {MODULES.map((m) => (
              <span key={m.k} className="flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1 text-xs">
                <i className="h-2.5 w-2.5 rounded-sm" style={{ background: m.c }} /> <b>{m.k}</b> <span className="text-muted">{m.t}</span>
              </span>
            ))}
          </div>
        </Panel>
        <Panel title="消融阶梯" subtitle="每一级证明一个模块的增量贡献" bodyClass="p-3">
          {!data ? <InfinityLoader label="加载实验结果" /> : (
            <div className="relative space-y-2.5 before:absolute before:bottom-4 before:left-[15px] before:top-4 before:w-px before:bg-line">
              {data.ladder.map((s) => (
                <div key={s.id} className="relative flex gap-3">
                  <div className="z-10 grid h-8 w-8 shrink-0 place-items-center rounded-full border border-line bg-panel">
                    {s.status === "done" ? <CircleCheck size={16} className="text-[#22b07d]" /> : s.status === "next" ? <CircleDot size={16} className="text-accent" /> : <CircleDashed size={16} className="text-faint" />}
                  </div>
                  <div className={clsx("flex-1 rounded-xl border p-2.5", s.status === "next" ? "border-accent/50" : "border-line")}>
                    <div className="flex items-center gap-2 text-sm font-semibold">
                      <span className="num">{s.id}</span> {s.name}
                      {s.status === "done" && <Badge color="#22b07d">已完成</Badge>}
                      {s.status === "next" && <Badge>下一步</Badge>}
                    </div>
                    <div className="text-xs text-muted">{s.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>

      {data && (
        <>
          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            {Object.entries(data.findings).map(([k, items]) => (
              <Panel key={k} title={`${k} 主要发现`} subtitle={FINDING_SUB[k] ?? ""} bodyClass="p-3">
                <ul className="space-y-1.5 text-[12.5px] leading-relaxed">
                  {items.map((t, i) => <li key={i} className={clsx("flex gap-2", t.startsWith("→") && "font-semibold text-accent")}><span className="text-faint">{t.startsWith("→") ? "" : "•"}</span>{t.replace(/^→\s*/, "→ ")}</li>)}
                </ul>
              </Panel>
            ))}
          </div>

          <Panel className="mt-4" title="组合表现：方法 × 设定（EFM，扣除 20bp）"
            subtitle={`${data.setting.universe} · ${data.setting.window} · ${data.setting.period}；绿色越深越好，蓝色加粗为最优；灰色为原始收益版 M0`}
            actions={<div className="flex flex-wrap gap-2">
              <Segmented name="lk" value={kind} onChange={setKind} options={Object.entries(KIND_CN).map(([v, l]) => ({ value: v as "mv" | "gmv" | "tp", label: l.split("（")[0] }))} />
              <Segmented name="lm" value={metric} onChange={setMetric} options={[{ value: "SD", label: "SD" }, { value: "SR", label: "Sharpe" }, { value: "TO", label: "换手" }]} />
            </div>} bodyClass="p-2">
            <p className="mb-2 px-3 text-[11px] text-muted">{KIND_CN[kind]} · {METRIC_CN[metric]}</p>
            <ResultGrid grid={data.kinds[kind].grid} bench={data.kinds[kind].bench} metric={metric} specs={["sfc", "dfc", "dbc", "dec"]} />
          </Panel>

          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            <Panel title="稳健性：月度收益、120 个月窗口" subtitle={`${data.setting.robustness}；${kind === "tp" ? "（换手惩罚表未导出，显示 MV）" : KIND_CN[kind]}`} bodyClass="p-2">
              <ResultGrid grid={data.monthly[kind === "tp" ? "mv" : kind].grid} bench={data.monthly[kind === "tp" ? "mv" : kind].bench} metric={metric === "TO" ? "SD" : metric} specs={["sfc", "dfc", "dbc", "dec"]} />
            </Panel>
            <Panel title="资产数量 N 的影响（MV，SFC-EFM）" subtitle="与原文相反：N 越大，因子模型相对收缩估计的优势越大；黑色粗线为 IPCA"
              actions={<Segmented name="nm" value={nMetric} onChange={setNMetric} options={[{ value: "SD", label: "SD" }, { value: "SR", label: "Sharpe" }, { value: "TO", label: "换手" }]} />} bodyClass="p-2">
              {nOption && <EChart option={nOption} height={300} />}
            </Panel>
          </div>

          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            <Panel title="残差相关诊断（K=5）" subtitle="原文截面秩变换去掉了市场因子，残差仍有 +0.25 的行业间相关；IPCA 残差最干净" bodyClass="p-2">
              {rcOption && <EChart option={rcOption} height={200} />}
            </Panel>
            <Panel title="M1 诊断变体（SFC-EFM，扣除 20bp）" subtitle="固定 K、去掉快变特征、特征平滑都没有让 IPCA 追上 M0；平滑只降低部分换手" bodyClass="p-2">
              <div className="overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead className="text-faint"><tr className="border-b border-line">
                    <th className="px-2 py-1 text-left font-medium">变体</th>
                    {["MV SD", "MV TO", "GMV SD", "TP SR", "QLIKE"].map((c) => <th key={c} className="px-2 py-1 text-right font-medium">{c}</th>)}
                  </tr></thead>
                  <tbody>
                    {Object.entries(data.ipca_variants.rows).map(([r, vals]) => {
                      const get = (c: string) => vals[data.ipca_variants.columns.indexOf(c)];
                      const f = (v: number | string | null, d = 2) => (v == null ? "—" : typeof v === "number" ? v.toFixed(d) : v);
                      return (
                        <tr key={r} className="border-b border-line/50">
                          <td className="px-2 py-1">{r}</td>
                          <td className="num px-2 py-1 text-right">{f(get("MV SD"), 3)}</td>
                          <td className="num px-2 py-1 text-right">{f(get("MV TO"), 1)}</td>
                          <td className="num px-2 py-1 text-right">{f(get("GMV SD"), 3)}</td>
                          <td className="num px-2 py-1 text-right">{f(get("TP SR"), 3)}</td>
                          <td className="num px-2 py-1 text-right">{f(get("QLIKE"), 1)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>

          {data.figures.ipca_importance && (
            <Panel className="mt-4" title="IPCA 特征重要性随时间（Γ 行范数占比）" subtitle="K=1 的年份常数项（市场）占主导；其余年份由 β、波动、换手、B/M、动量分担 —— 支持 CIL 中的行组稀疏" bodyClass="p-3">
              <div className="overflow-hidden rounded-xl bg-white p-2"><img src={data.figures.ipca_importance} alt="IPCA 特征重要性" className="w-full" /></div>
            </Panel>
          )}
        </>
      )}

      <Panel className="mt-4" title="2×2 换手实验（M3/M4 阶段）" subtitle="训练时惩罚 vs 测试时惩罚：前者应让风险–换手前沿外移（假设 H4）">
        <div className="grid max-w-xl grid-cols-[120px_1fr_1fr] overflow-hidden rounded-xl border border-line text-xs">
          <div className="bg-panel-2 p-2" />
          <div className="bg-panel-2 p-2 text-center font-semibold">测试：不加惩罚</div>
          <div className="bg-panel-2 p-2 text-center font-semibold">测试：+κ‖w−w⁺‖₁</div>
          <div className="border-t border-line p-2 font-semibold">训练 κ = 0</div>
          <div className="border-t border-line p-2 text-center text-muted">基线</div>
          <div className="border-t border-line p-2 text-center text-muted">基准论文式 (9)</div>
          <div className="border-t border-line p-2 font-semibold">训练 λ &gt; 0</div>
          <div className="border-t border-line bg-dfl/10 p-2 text-center font-semibold text-dfl">DeFaCo</div>
          <div className="border-t border-line p-2 text-center text-muted">两者叠加</div>
        </div>
        <p className="mt-3 text-[11px] text-faint">M0 已给出“测试时惩罚”的一列（TP）：换手从 30–50% 降到 4–7%，扣费 Sharpe 约翻倍；训练时惩罚待 M3/M4 实现。</p>
      </Panel>
    </div>
  );
}
