import { useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import type { EChartsOption } from "echarts";
import { useMemo } from "react";
import { useData, type Overview as OverviewData, type Portfolio, type Security } from "../lib/data";
import { useQuotes } from "../lib/realtime";
import { cssVar } from "../lib/store";
import { pct, yi } from "../lib/format";
import { EChart } from "../components/charts/EChart";
import { Badge, DataBanner, Delta, InfinityLoader, PageHeader, Panel, Stat } from "../components/ui";

const INDEX_NAMES: Record<string, string> = { sh000300: "沪深300", sh000905: "中证500" };
const INDEX_CODES: Record<string, string> = { sh000300: "000300.SH", sh000905: "000905.SH" };

function IndexCard({ code, series: hist }: { code: string; series: [string, number][] }) {
  const quotes = useQuotes([INDEX_CODES[code]]);
  const q = quotes[INDEX_CODES[code]];
  // append today's live value when it is newer than the stored history
  const today = q?.time?.slice(0, 10);
  const series: [string, number][] = q?.price && today && today > hist[hist.length - 1][0] ? [...hist, [today, q.price]] : hist;
  const last = series[series.length - 1][1];
  const prev = q?.prev_close ?? series[series.length - 2][1];
  const yearAgo = series[Math.max(0, series.length - 244)][1];
  const up = last >= series[Math.max(0, series.length - 60)][1];
  const tail = series.slice(-120);
  const color = up ? cssVar("--up") : cssVar("--down");
  const option: EChartsOption = {
    grid: { left: 0, right: 0, top: 4, bottom: 0 },
    xAxis: { type: "category", show: false, data: tail.map((d) => d[0]) },
    yAxis: { type: "value", show: false, scale: true },
    series: [
      {
        type: "line",
        data: tail.map((d) => d[1]),
        showSymbol: false,
        lineStyle: { width: 1.6, color },
        areaStyle: {
          color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: color + "55" }, { offset: 1, color: color + "00" }] },
        },
      },
    ],
    tooltip: { show: false },
  };
  return (
    <div className="panel flex items-center gap-4 px-4 py-3 fade-up">
      <div className="min-w-[120px]">
        <div className="text-xs text-muted">{INDEX_NAMES[code] ?? code}</div>
        <div className="num mt-0.5 text-xl font-semibold">{last.toFixed(2)}</div>
        <div className="mt-0.5 flex gap-3 text-[11px]">
          <Delta value={last / prev - 1} />
          <span className="text-faint">
            近一年 <Delta value={last / yearAgo - 1} digits={1} />
          </span>
        </div>
      </div>
      <div className="h-16 min-w-0 flex-1">
        <EChart option={option} height={64} />
      </div>
    </div>
  );
}

export default function Overview() {
  const { data: raw, loading } = useData<OverviewData>("overview.json");
  const { data: port } = useData<Portfolio>("portfolio.json");
  const { data: secs } = useData<{ items: Security[] }>("securities.json");
  const nav = useNavigate();
  // live snapshot: the 300 largest stocks still trading at the research cut-off
  const liveCodes = useMemo(
    () => (secs?.items ?? []).filter((s) => s.mcap > 0).sort((a, b) => b.mcap - a.mcap).slice(0, 300).map((s) => s.code),
    [secs],
  );
  const quotes = useQuotes(liveCodes);
  const data = useMemo(() => {
    if (!raw || !secs) return raw;
    const ind = new Map(secs.items.map((s) => [s.code, s.industry]));
    const qs = liveCodes.map((c) => quotes[c]).filter((q) => q && q.price && q.prev_close);
    if (qs.length < 100) return raw;
    const agg: Record<string, { w: number; r: number; n: number }> = {};
    const hist = new Array(20).fill(0);
    let up = 0, down = 0, flat = 0;
    for (const q of qs) {
      const k = ind.get(q.code) ?? "其他";
      const a = agg[k] ?? (agg[k] = { w: 0, r: 0, n: 0 });
      a.w += q.amount; a.r += q.pct * q.amount; a.n += 1;
      if (q.pct > 0) up++; else if (q.pct < 0) down++; else flat++;
      hist[Math.max(0, Math.min(19, Math.floor((Math.max(-0.1, Math.min(0.0999, q.pct)) + 0.1) / 0.01)))]++;
    }
    return {
      ...raw,
      live: { n: qs.length, time: qs[0].time },
      industries: Object.entries(agg).map(([k, v]) => ({ ind: k, ret: v.w ? v.r / v.w : 0, amount: v.w / 1e8, n: v.n })),
      breadth: { up, down, flat, hist },
    };
  }, [raw, secs, quotes, liveCodes]);
  if (loading || !data) return <div className="grid h-96 place-items-center"><InfinityLoader label="加载市场数据" /></div>;
  const live = (data as OverviewData & { live?: { n: number; time: string } }).live;

  const up = cssVar("--up"), down = cssVar("--down"), muted = cssVar("--panel-2");
  const maxAbs = Math.max(0.02, ...data.industries.map((d) => Math.abs(d.ret)));
  // ECharts cannot parse color-mix(): blend hex colours here
  const blend = (a: string, b: string, t: number) => {
    const pa = a.slice(1).match(/\w\w/g)!.map((h) => parseInt(h, 16));
    const pb = b.slice(1).match(/\w\w/g)!.map((h) => parseInt(h, 16));
    return `rgb(${pa.map((v, i) => Math.round(v * t + pb[i] * (1 - t))).join(",")})`;
  };
  const treemap: EChartsOption = {
    tooltip: {
      formatter: (p: unknown) => {
        const d = (p as { data: { name: string; value: number; ret: number; n: number } }).data;
        return `<b>${d.name}</b><br/>涨跌 ${(d.ret * 100).toFixed(2)}%<br/>成交额 ${d.value.toFixed(0)} 亿 · ${d.n} 只`;
      },
    },
    series: [
      {
        type: "treemap",
        roam: false,
        nodeClick: false,
        breadcrumb: { show: false },
        width: "100%",
        height: "100%",
        itemStyle: { borderColor: cssVar("--panel"), borderWidth: 2, gapWidth: 2, borderRadius: 6 },
        label: {
          show: true,
          formatter: (p: unknown) => {
            const d = (p as { data: { name: string; ret: number } }).data;
            return `{n|${d.name}}\n{r|${d.ret > 0 ? "+" : ""}${(d.ret * 100).toFixed(2)}%}`;
          },
          rich: { n: { fontSize: 12, fontWeight: 600 }, r: { fontSize: 11, padding: [3, 0, 0, 0] } },
        },
        data: data.industries.map((d) => ({
          name: d.ind,
          value: d.amount,
          ret: d.ret,
          n: d.n,
          itemStyle: { color: blend(d.ret >= 0 ? up : down, muted, 0.3 + 0.7 * Math.min(1, Math.abs(d.ret) / maxAbs)) },
          // pale tiles (small moves) need dark text in the light theme
          label: { color: 0.3 + 0.7 * Math.min(1, Math.abs(d.ret) / maxAbs) > 0.6 ? "#fff" : cssVar("--text") },
        })),
      },
    ],
  };

  const bins = data.breadth.hist.map((_, i) => -10 + i);
  const breadth: EChartsOption = {
    grid: { left: 30, right: 8, top: 10, bottom: 24 },
    xAxis: { type: "category", data: bins.map((b) => `${b}%`), axisLabel: { interval: 4 } },
    yAxis: { type: "value" },
    tooltip: { trigger: "axis" },
    series: [{ type: "bar", data: data.breadth.hist.map((v, i) => ({ value: v, itemStyle: { color: bins[i] >= 0 ? up : down, borderRadius: [3, 3, 0, 0] } })), barWidth: "70%" }],
  };

  const reg = data.vol_regime;
  const regOption: EChartsOption = {
    grid: { left: 40, right: 12, top: 16, bottom: 24 },
    tooltip: { trigger: "axis", valueFormatter: (v) => `${((v as number) * 100).toFixed(1)}%` },
    xAxis: { type: "category", data: reg.map((d) => d[0]), axisLabel: { formatter: (v: string) => v.slice(0, 7) } },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: [
      {
        type: "line",
        data: reg.map((d) => d[1]),
        showSymbol: false,
        lineStyle: { color: cssVar("--m-reg"), width: 1.5 },
        areaStyle: { color: cssVar("--m-reg") + "22" },
        markLine: {
          symbol: "none",
          lineStyle: { color: cssVar("--faint"), type: "dashed" },
          label: { formatter: "高波动阈值", color: cssVar("--muted"), fontSize: 10 },
          data: [{ yAxis: data.vol_threshold }],
        },
      },
    ],
  };

  const pf = port?.models.pca_factor;
  return (
    <div>
      <PageHeader
        title="市场总览"
        desc={live ? `实时：${live.n} 只成分股报价 · ${live.time.slice(5, 16)} · 区制监测基于收盘数据` : `截至 ${data.asof} · 行业、宽度与市场区制`}
      />
      <DataBanner note={data.meta.survivorship_biased ? data.meta.note : undefined} />
      <div className="grid gap-4 md:grid-cols-2">
        {Object.entries(data.indices).map(([k, s]) => (
          <IndexCard key={k} code={k} series={s} />
        ))}
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Panel title={live ? "行业热力图（实时）" : "行业热力图"} subtitle="面积 = 成交额，颜色 = 成交额加权涨跌（申万一级）" className="xl:col-span-2" bodyClass="p-2">
          <EChart option={treemap} height={360} />
        </Panel>
        <div className="flex flex-col gap-4">
          <Panel title={live ? "涨跌分布（实时）" : "涨跌分布"} subtitle={live ? "市值前 300 只个股当日涨跌幅" : "样本内个股当日收益"}>
            <div className="mb-2 flex gap-4 text-xs">
              <span className="text-up">▲ 上涨 {data.breadth.up}</span>
              <span className="text-down">▼ 下跌 {data.breadth.down}</span>
              <span className="text-muted">平 {data.breadth.flat}</span>
            </div>
            <EChart option={breadth} height={150} />
          </Panel>
          <Panel
            title="我的组合（演示）"
            subtitle="PCA 因子模型 · 最小方差"
            actions={
              <button className="flex items-center gap-1 text-xs text-accent hover:underline" onClick={() => nav("/portfolio")}>
                详情 <ArrowRight size={13} />
              </button>
            }
          >
            {pf ? (
              <div className="grid grid-cols-2 gap-2">
                <Stat label="预期年化波动" value={pct(pf.risk.total_vol)} />
                <Stat label="持仓数" value={pf.n_holdings} hint={`HHI ${pf.hhi.toFixed(3)}`} />
                <Stat label="因子风险占比" value={pct(pf.risk.factor_share ?? null, 1)} />
                <Stat label="第一大行业" value={<span className="text-base">{Object.keys(pf.industry_weights)[0]}</span>} hint={pct(Object.values(pf.industry_weights)[0], 1)} />
              </div>
            ) : (
              <div className="skeleton h-28" />
            )}
          </Panel>
        </div>
      </div>
      <Panel
        className="mt-4"
        title="市场区制监测"
        subtitle="沪深300 20 日年化波动率；超过历史 75% 分位视为高波动区制（附录模块 R 的简化版）"
        accent="var(--m-reg)"
        actions={<Badge color="var(--m-reg)">区制</Badge>}
      >
        <EChart option={regOption} height={220} />
        <p className="mt-2 text-[11px] text-faint">样本内成交额合计 {yi(data.industries.reduce((s, d) => s + d.amount, 0))}</p>
      </Panel>
    </div>
  );
}
