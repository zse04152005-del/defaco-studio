import clsx from "clsx";
import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Check, GitCompare, Search } from "lucide-react";
import { cssVar } from "../lib/store";
import { pct } from "../lib/format";
import { EChart } from "../components/charts/EChart";
import { Badge, InfinityLoader, PageHeader, Panel, Segmented, Stat } from "../components/ui";

interface Row {
  name: string; family: string; formula: string; desc: string; modified: boolean;
  rank_ic: number; ic: number; icir: number; t: number; ic_pos: number; direction: number;
  ls_ann: number | null; ls_sharpe: number | null; monotonic: number | null; turnover: number | null; coverage: number;
}
interface Summary { universe: string; neutral: string; h: number; start: string; end: string; n_factors: number; rows: Row[] }
interface Detail {
  name: string; family: string; formula: string; desc: string; modified: boolean;
  stats: { rank_ic: number; ic: number; icir: number; t: number; ic_pos: number; direction: number; ls_ann: number | null; ls_vol: number | null;
           ls_sharpe: number | null; ls_maxdd: number | null; turnover: number | null; autocorr: number | null; months: number };
  ic_series: [string, number | null][];
  quantile_mean: Record<string, number>;
  quantile_cum: Record<string, [string, number][]>;
  ls_cum: [string, number][];
  decay: (number | null)[];
  yearly_ic: Record<string, number>;
}

const FAMILIES = ["全部", "基本面 / 价量特征", "Alpha158", "Alpha101", "GTJA191"];
const SOURCES: Record<string, string> = {
  "Alpha158": "Qlib（Microsoft，MIT）qlib/contrib/data/loader.py",
  "Alpha101": "Kakushadze (2016) 101 Formulaic Alphas, Wilmott",
  "GTJA191": "国泰君安《基于短周期价量特征的多因子选股体系》（2017）",
  "基本面 / 价量特征": "项目自建点时特征（财报按实际披露日对齐）",
};

async function get<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
  return d as T;
}

const icColor = (v: number) => {
  const a = Math.min(1, Math.abs(v) / 0.06);
  return v > 0 ? `color-mix(in srgb, var(--up) ${a * 100}%, var(--muted))` : `color-mix(in srgb, var(--down) ${a * 100}%, var(--muted))`;
};

export default function Factors() {
  const [universe, setUniverse] = useState<"csi800" | "csi300">("csi800");
  const [neutral, setNeutral] = useState<"none" | "industry" | "industry_size">("industry_size");
  const [h, setH] = useState<"1" | "3">("1");
  const [sum, setSum] = useState<Summary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [fam, setFam] = useState("全部");
  const [sortKey, setSortKey] = useState<"icir" | "rank_ic" | "ls_sharpe" | "turnover" | "name">("icir");
  const [sel, setSel] = useState<string>("turn_1m");
  const [picked, setPicked] = useState<string[]>([]);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [corr, setCorr] = useState<{ names: string[]; matrix: number[][] } | null>(null);
  const qs = `universe=${universe}&neutral=${neutral}&h=${h}`;

  useEffect(() => {
    setSum(null);
    get<Summary>(`/api/factors/summary?${qs}`).then((d) => { setSum(d); setErr(null); }).catch((e) => setErr((e as Error).message));
  }, [qs]);
  useEffect(() => {
    setDetail(null);
    get<Detail>(`/api/factors/detail/${sel}?${qs}`).then(setDetail).catch(() => setDetail(null));
  }, [sel, qs]);
  useEffect(() => {
    if (picked.length < 2) return setCorr(null);
    get<{ names: string[]; matrix: number[][] }>("/api/factors/corr", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ names: picked, universe, neutral }) })
      .then(setCorr).catch(() => setCorr(null));
  }, [picked, universe, neutral]);

  const rows = useMemo(() => {
    if (!sum) return [];
    const f = q.trim().toLowerCase();
    return sum.rows
      .filter((r) => fam === "全部" || r.family === fam)
      .filter((r) => !f || r.name.toLowerCase().includes(f) || r.formula.toLowerCase().includes(f) || r.desc.includes(q.trim()))
      .sort((a, b) => sortKey === "name" ? a.name.localeCompare(b.name) : sortKey === "turnover" ? (a.turnover ?? 9) - (b.turnover ?? 9)
        : Math.abs((b[sortKey] as number) ?? 0) - Math.abs((a[sortKey] as number) ?? 0));
  }, [sum, q, fam, sortKey]);

  const strong = sum ? sum.rows.filter((r) => Math.abs(r.t) > 3).length : 0;
  return (
    <div>
      <PageHeader
        title="因子研究"
        desc={sum ? `${sum.n_factors} 个因子 · ${sum.universe} · ${sum.neutral} · 未来 ${sum.h} 个月收益 · ${sum.start} → ${sum.end}（月末截面）` : "IC / 分层收益 / 衰减 / 换手 / 相关性（参照 alphalens）"}
        actions={
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <Segmented name="fu" value={universe} onChange={setUniverse} options={[{ value: "csi800", label: "沪深300+中证500" }, { value: "csi300", label: "沪深300" }]} />
            <Segmented name="fn" value={neutral} onChange={setNeutral} options={[{ value: "none", label: "原始" }, { value: "industry", label: "行业中性" }, { value: "industry_size", label: "行业+市值中性" }]} />
            <Segmented name="fh" value={h} onChange={setH} options={[{ value: "1", label: "1 个月" }, { value: "3", label: "3 个月" }]} />
          </div>
        }
      />
      {err && <div className="mb-4 rounded-xl border border-warn/40 bg-warn/10 px-4 py-2.5 text-[12px]">{err}</div>}
      {sum && (
        <div className="mb-4 rounded-xl border border-line bg-panel-2/60 px-4 py-2.5 text-[12px]">
          方法：每月末截面 MAD 去极值 + 标准化{neutral !== "none" ? (neutral === "industry" ? " + 申万一级行业中性化" : " + 行业与对数市值中性化") : ""}；Rank IC 与下一期收益的 Spearman 相关；t 值{sum.h > 1 ? "用 Newey-West 修正重叠收益，" : ""}检验 IC 均值是否为 0；分层为等权五分组，多空方向取 IC 符号。
          <b className="ml-1">{strong}</b> 个因子 |t| &gt; 3（考虑到同时检验 {sum.n_factors} 个因子，建议以 |t| &gt; 3 为显著门槛，参考 Harvey, Liu &amp; Zhu 2016）。
        </div>
      )}
      <div className="grid gap-4 2xl:grid-cols-[minmax(0,620px)_minmax(0,1fr)] xl:grid-cols-[minmax(0,560px)_minmax(0,1fr)]">
        <Panel title="因子库" subtitle="点击查看详情；勾选多个看相关性" bodyClass="p-0">
          <div className="flex flex-wrap items-center gap-2 border-b border-line p-2">
            <div className="flex flex-1 items-center gap-1.5 rounded-lg border border-line bg-bg-2 px-2 py-1">
              <Search size={13} className="text-faint" />
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="名称 / 公式 / 说明" className="flex-1 bg-transparent text-[12px] outline-none" />
            </div>
            <select value={fam} onChange={(e) => setFam(e.target.value)} className="h-7 rounded-md border border-line bg-bg-2 px-1.5 text-[11px]">
              {FAMILIES.map((f) => <option key={f}>{f}</option>)}
            </select>
          </div>
          {!sum ? (
            <div className="grid h-72 place-items-center"><InfinityLoader label="计算因子 IC" /></div>
          ) : (
            <div className="max-h-[720px] overflow-y-auto">
              <table className="w-full text-[11.5px]">
                <thead className="sticky top-0 z-10 bg-panel text-left text-faint">
                  <tr className="border-b border-line">
                    <th className="w-6 px-2" />
                    {([["name", "因子"], ["rank_ic", "RankIC"], ["icir", "ICIR"], ["", "t"], ["ls_sharpe", "多空SR"], ["turnover", "换手"]] as const).map(([k, l]) => (
                      <th key={l} onClick={() => k && setSortKey(k)} className={clsx("px-2 py-2 font-medium", k && "cursor-pointer hover:text-ink", k !== "name" && "text-right", sortKey === k && "text-accent")}>{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.name} onClick={() => setSel(r.name)} className={clsx("cursor-pointer border-b border-line/50 hover:bg-panel-2/50", r.name === sel && "bg-panel-2")}>
                      <td className="px-2" onClick={(e) => { e.stopPropagation(); setPicked(picked.includes(r.name) ? picked.filter((x) => x !== r.name) : [...picked, r.name].slice(-12)); }}>
                        <span className={clsx("grid h-3.5 w-3.5 place-items-center rounded border", picked.includes(r.name) ? "border-accent bg-accent text-white" : "border-line")}>
                          {picked.includes(r.name) && <Check size={10} />}
                        </span>
                      </td>
                      <td className="max-w-[230px] px-2 py-1.5">
                        <div className="flex items-center gap-1.5">
                          <span className="font-medium text-ink">{r.name}</span>
                          <span className="text-[9.5px] text-faint">{r.family === "基本面 / 价量特征" ? "自建" : r.family}</span>
                          {r.modified && <span className="text-[9.5px] text-warn" title="原式依赖跨股票的价格水平，已改为相对价格版本">改</span>}
                        </div>
                        <div className="truncate text-[10px] text-faint" title={r.formula}>{r.desc || r.formula}</div>
                      </td>
                      <td className="num px-2 text-right" style={{ color: icColor(r.rank_ic) }}>{r.rank_ic.toFixed(3)}</td>
                      <td className="num px-2 text-right">{r.icir.toFixed(2)}</td>
                      <td className={clsx("num px-2 text-right", Math.abs(r.t) > 3 ? "font-semibold text-ink" : "text-faint")}>{r.t.toFixed(1)}</td>
                      <td className="num px-2 text-right">{r.ls_sharpe == null ? "—" : r.ls_sharpe.toFixed(2)}</td>
                      <td className="num px-2 text-right text-muted">{r.turnover == null ? "—" : pct(r.turnover, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
        <div className="min-w-0 space-y-4">
          {detail ? <DetailView d={detail} h={+h} /> : <div className="grid h-96 place-items-center rounded-xl border border-line"><InfinityLoader label="计算因子详情" /></div>}
          {corr && <CorrView c={corr} />}
          {!corr && picked.length < 2 && (
            <p className="flex items-center gap-1.5 text-[11px] text-faint"><GitCompare size={13} /> 在左侧勾选 2 个以上因子，查看截面秩相关矩阵（判断因子是否重复）。</p>
          )}
        </div>
      </div>
    </div>
  );
}

function DetailView({ d, h }: { d: Detail; h: number }) {
  const s = d.stats;
  const ic = d.ic_series.filter((x) => x[1] != null) as [string, number][];
  const roll = ic.map((x, i) => [x[0], i >= 11 ? ic.slice(i - 11, i + 1).reduce((a, b) => a + b[1], 0) / 12 : null] as [string, number | null]);
  let c = 0;
  const cum = ic.map(([t, v]) => [t, (c += v)] as [string, number]);
  const icOption: EChartsOption = {
    grid: { left: 44, right: 44, top: 30, bottom: 26 },
    legend: { top: 0 },
    tooltip: { trigger: "axis", valueFormatter: (v) => (v == null ? "—" : (v as number).toFixed(3)) },
    xAxis: { type: "time" },
    yAxis: [{ type: "value" }, { type: "value", splitLine: { show: false } }],
    series: [
      { name: "Rank IC", type: "bar", data: ic, itemStyle: { color: (p) => ((p.value as [string, number])[1] >= 0 ? cssVar("--up") : cssVar("--down")) + "aa" } },
      { name: "12 月滚动均值", type: "line", showSymbol: false, data: roll, lineStyle: { color: cssVar("--accent"), width: 1.6 }, itemStyle: { color: cssVar("--accent") } },
      { name: "累计 IC（右轴）", type: "line", yAxisIndex: 1, showSymbol: false, data: cum, lineStyle: { color: cssVar("--warn"), width: 1.2, type: "dashed" }, itemStyle: { color: cssVar("--warn") } },
    ],
  };
  const qColors = ["#2e8b57", "#7ab68d", "#9aa3b2", "#e8a07a", "#e0524a"];
  const qOption: EChartsOption = {
    grid: { left: 44, right: 12, top: 30, bottom: 26 },
    legend: { top: 0 },
    tooltip: { trigger: "axis", valueFormatter: (v) => (v as number).toFixed(3) },
    xAxis: { type: "time" },
    yAxis: { type: "log" },
    series: [
      ...Object.entries(d.quantile_cum).map(([g, v], i) => ({ name: `Q${g}`, type: "line" as const, showSymbol: false, data: v, lineStyle: { width: 1.2, color: qColors[i] }, itemStyle: { color: qColors[i] } })),
      { name: "多空", type: "line" as const, showSymbol: false, data: d.ls_cum, lineStyle: { width: 2, color: cssVar("--accent") }, itemStyle: { color: cssVar("--accent") } },
    ],
  };
  const qmOption: EChartsOption = {
    grid: { left: 44, right: 10, top: 14, bottom: 22 },
    tooltip: { trigger: "axis", valueFormatter: (v) => pct(v as number, 2) },
    xAxis: { type: "category", data: ["Q1 最低", "Q2", "Q3", "Q4", "Q5 最高"] },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(1)}%` } },
    series: [{ type: "bar", data: [1, 2, 3, 4, 5].map((g, i) => ({ value: d.quantile_mean[g], itemStyle: { color: qColors[i] } })) }],
  };
  const decOption: EChartsOption = {
    grid: { left: 44, right: 10, top: 14, bottom: 22 },
    tooltip: { trigger: "axis", valueFormatter: (v) => (v as number).toFixed(3) },
    xAxis: { type: "category", data: d.decay.map((_, i) => `第 ${i + 1} 月`) },
    yAxis: { type: "value" },
    series: [{ type: "bar", data: d.decay, itemStyle: { color: cssVar("--accent") } }],
  };
  return (
    <>
      <Panel title={<span className="flex items-center gap-2">{d.name} <Badge>{d.family}</Badge>{d.modified && <Badge color="var(--warn)">相对价格版</Badge>}</span>}
        subtitle={SOURCES[d.family]} bodyClass="p-3">
        <code className="block whitespace-pre-wrap rounded-lg bg-panel-2 px-3 py-2 text-[11.5px] text-ink">{d.formula}</code>
        {d.desc && <p className="mt-1.5 text-[11px] text-muted">{d.desc}</p>}
        <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4 2xl:grid-cols-8">
          <Stat label="Rank IC 均值" value={<span style={{ color: icColor(s.rank_ic) }}>{s.rank_ic.toFixed(3)}</span>} hint={`Pearson ${s.ic.toFixed(3)}`} />
          <Stat label="ICIR（年化）" value={s.icir.toFixed(2)} />
          <Stat label="t 值" value={s.t.toFixed(1)} hint={Math.abs(s.t) > 3 ? "显著（|t|>3）" : Math.abs(s.t) > 2 ? "弱显著" : "不显著"} />
          <Stat label="IC 同向月份" value={pct(s.ic_pos, 0)} hint={s.direction > 0 ? "正向因子" : "反向因子"} />
          <Stat label="多空年化" value={s.ls_ann == null ? "—" : pct(s.ls_ann, 1)} hint={`回撤 ${s.ls_maxdd == null ? "—" : pct(s.ls_maxdd, 0)}`} />
          <Stat label="多空 Sharpe" value={s.ls_sharpe == null ? "—" : s.ls_sharpe.toFixed(2)} hint="未扣交易成本" />
          <Stat label="头部组换手" value={s.turnover == null ? "—" : pct(s.turnover, 0)} hint="每月" />
          <Stat label="因子自相关" value={s.autocorr == null ? "—" : s.autocorr.toFixed(2)} hint="相邻月秩相关" />
        </div>
      </Panel>
      <Panel title={`Rank IC 时间序列（未来 ${h} 个月）`} bodyClass="p-2"><EChart option={icOption} height={240} /></Panel>
      <div className="grid gap-4 2xl:grid-cols-2">
        {Object.keys(d.quantile_cum).length > 0 && <Panel title="分层累计收益（等权，对数坐标）" subtitle="Q5 = 因子值最高组；多空方向按 IC 符号" bodyClass="p-2"><EChart option={qOption} height={240} /></Panel>}
        <div className="grid gap-4">
          <Panel title="各组平均月收益" bodyClass="p-2"><EChart option={qmOption} height={110} /></Panel>
          <Panel title="IC 衰减（因子与之后第 k 个月收益）" bodyClass="p-2"><EChart option={decOption} height={110} /></Panel>
        </div>
      </div>
      <Panel title="逐年 Rank IC" bodyClass="p-3">
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(d.yearly_ic).map(([y, v]) => (
            <span key={y} className="num rounded-md border border-line px-2 py-1 text-[11px]">{y} <b style={{ color: icColor(v) }}>{v.toFixed(3)}</b></span>
          ))}
        </div>
      </Panel>
    </>
  );
}

function CorrView({ c }: { c: { names: string[]; matrix: number[][] } }) {
  const option: EChartsOption = {
    grid: { left: 80, right: 16, top: 10, bottom: 70 },
    tooltip: { formatter: (p: unknown) => { const v = (p as { value: [number, number, number] }).value; return `${c.names[v[0]]} × ${c.names[v[1]]}：${v[2].toFixed(2)}`; } },
    xAxis: { type: "category", data: c.names, axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: "category", data: c.names, axisLabel: { fontSize: 10 } },
    visualMap: { show: false, min: -1, max: 1, inRange: { color: [cssVar("--down"), cssVar("--panel-2"), cssVar("--up")] } },
    series: [{ type: "heatmap", data: c.matrix.flatMap((row, i) => row.map((v, j) => [j, i, v])),
      label: { show: c.names.length <= 8, fontSize: 10, color: cssVar("--text"), formatter: (p) => ((p.value as number[])[2]).toFixed(2) } }],
  };
  return <Panel title="因子截面秩相关（月度平均）" subtitle="相关 > 0.7 的因子信息高度重复" bodyClass="p-2"><EChart option={option} height={Math.max(260, c.names.length * 32 + 90)} /></Panel>;
}
