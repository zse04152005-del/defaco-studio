import clsx from "clsx";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Check, RotateCcw, X } from "lucide-react";
import { useData, type Screener as ScreenerData } from "../lib/data";
import { usePortfolioSettings } from "../lib/portfolio";
import { DataBanner, Delta, GlowButton, InfinityLoader, PageHeader, Panel, RimButton } from "../components/ui";
import { useUI } from "../lib/store";
import { useSecurities } from "../lib/securities";

const FACTORS: { key: string; label: string; hint: string }[] = [
  { key: "bm", label: "价值 BM", hint: "账面市值比" },
  { key: "ep", label: "盈利收益率 EP", hint: "TTM EPS / 价格" },
  { key: "roe", label: "ROE", hint: "净资产收益率" },
  { key: "mom_12_2", label: "动量 12-2", hint: "过去 12 个月剔除近 1 月" },
  { key: "size", label: "规模", hint: "流通市值（对数）" },
  { key: "vol_12m", label: "低波动", hint: "分位越高波动越大（可设上限）" },
];

const TEMPLATES: { name: string; desc: string; rules: Record<string, [number, number]> }[] = [
  { name: "低估值高盈利", desc: "BM ≥ 60 分位，ROE ≥ 60 分位", rules: { bm: [60, 100], roe: [60, 100] } },
  { name: "低波动防御", desc: "12 月波动 ≤ 40 分位", rules: { vol_12m: [0, 40] } },
  { name: "强势动量", desc: "动量 ≥ 70 分位", rules: { mom_12_2: [70, 100] } },
  { name: "大盘蓝筹", desc: "规模 ≥ 70 分位，EP ≥ 50 分位", rules: { size: [70, 100], ep: [50, 100] } },
];

export default function Screener() {
  const { data, loading } = useData<ScreenerData>("screener.json");
  const [rules, setRules] = useState<Record<string, [number, number]>>({});
  const [inds, setInds] = useState<string[]>([]);
  // the candidate pool is shared with the stock page (☆ 加入候选池) and 组合构建
  const pool = usePortfolioSettings((s) => s.pool);
  const setPortfolio = usePortfolioSettings((s) => s.set);
  const setPool = (p: string[]) => setPortfolio({ pool: p, ...(p.length ? {} : { usePool: false }) });
  const secs = useSecurities();
  const nav = useNavigate();
  const select = useUI((s) => s.select);

  const industries = useMemo(() => [...new Set((data?.items ?? []).map((x) => x.industry))].sort(), [data]);
  const hits = useMemo(() => {
    if (!data) return [];
    return data.items
      .filter((x) => !inds.length || inds.includes(x.industry))
      .filter((x) => Object.entries(rules).every(([k, [lo, hi]]) => {
        const v = (x[k] as number) * 100;
        return v >= lo && v <= hi;
      }))
      .sort((a, b) => (b.mcap ?? -1) - (a.mcap ?? -1)); // delisted stocks have no market cap: last
  }, [data, rules, inds]);

  if (loading || !data) return <div className="grid h-96 place-items-center"><InfinityLoader label="加载因子截面" /></div>;

  return (
    <div>
      <PageHeader
        title="选股器"
        desc={`因子截面分位（0–100）· 截至 ${data.asof} · 结果可送入组合构建的股票池`}
        actions={
          <GlowButton onClick={() => { setPortfolio({ pool, usePool: true }); nav("/portfolio"); }} disabled={!pool.length}>
            <Check size={16} /> 用候选池构建组合（{pool.length}）
          </GlowButton>
        }
      />
      <DataBanner note={data.meta.survivorship_biased ? data.meta.note : undefined} />
      <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
        <div className="flex flex-col gap-4">
          <Panel
            title={`候选池（${pool.length}）`}
            subtitle="选股器、个股页「加入候选池」都会加到这里"
            actions={pool.length > 0 ? <button className="text-[11px] text-faint hover:text-ink" onClick={() => setPool([])}>清空</button> : undefined}
            bodyClass="p-2"
          >
            {pool.length === 0 ? (
              <p className="px-2 py-3 text-[11px] text-faint">还没有候选股：在右侧结果中点「+ 候选」，或在个股页点「加入候选池」。</p>
            ) : (
              <div className="flex max-h-48 flex-wrap gap-1.5 overflow-y-auto p-1">
                {pool.map((c) => (
                  <span key={c} className="flex items-center gap-1 rounded-full border border-line bg-panel-2 py-0.5 pl-2 pr-1 text-[11px]">
                    <button className="hover:text-accent" onClick={() => { select(c); nav(`/stock?code=${c}`); }}>
                      {secs?.byCode.get(c)?.name ?? c}
                    </button>
                    <button className="rounded-full p-0.5 text-faint hover:text-ink" onClick={() => setPool(pool.filter((x) => x !== c))} aria-label={`移出 ${c}`}>
                      <X size={10} />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </Panel>
          <Panel title="策略模板" subtitle="一键套用（后续接入 15 套技术策略 YAML）" bodyClass="p-3">
            <div className="grid gap-2">
              {TEMPLATES.map((t) => (
                <button key={t.name} onClick={() => setRules(t.rules)} className="rounded-xl border border-line px-3 py-2 text-left hover:border-accent">
                  <div className="text-xs font-semibold">{t.name}</div>
                  <div className="text-[11px] text-faint">{t.desc}</div>
                </button>
              ))}
            </div>
          </Panel>
          <Panel title="因子条件" actions={<button onClick={() => { setRules({}); setInds([]); }} className="text-faint hover:text-ink" title="重置"><RotateCcw size={14} /></button>} bodyClass="p-3">
            <div className="space-y-3">
              {FACTORS.map((f) => {
                const [lo, hi] = rules[f.key] ?? [0, 100];
                return (
                  <div key={f.key}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span title={f.hint}>{f.label}</span>
                      <span className="num text-muted">{lo}–{hi}</span>
                    </div>
                    <div className="flex gap-2">
                      <input type="range" min={0} max={100} value={lo} onChange={(e) => setRules({ ...rules, [f.key]: [Math.min(+e.target.value, hi), hi] })} className="flex-1 accent-[var(--accent)]" />
                      <input type="range" min={0} max={100} value={hi} onChange={(e) => setRules({ ...rules, [f.key]: [lo, Math.max(+e.target.value, lo)] })} className="flex-1 accent-[var(--accent)]" />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 text-xs text-muted">行业</div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {industries.map((i) => (
                <button key={i} onClick={() => setInds(inds.includes(i) ? inds.filter((x) => x !== i) : [...inds, i])} className={clsx("rounded-full border px-2 py-0.5 text-[11px]", inds.includes(i) ? "border-accent bg-accent/15 text-ink" : "border-line text-muted hover:text-ink")}>
                  {i}
                </button>
              ))}
            </div>
          </Panel>
        </div>
        <Panel title={`命中 ${hits.length} 只`} subtitle="按流通市值排序" actions={<RimButton onClick={() => setPool([...new Set([...pool, ...hits.map((h) => h.code)])])}>全部加入候选池</RimButton>} bodyClass="p-0">
          <div className="max-h-[680px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-panel text-left text-faint">
                <tr className="border-b border-line">
                  <th className="px-4 py-2 font-medium">代码</th>
                  <th className="px-2 py-2 font-medium">名称</th>
                  <th className="px-2 py-2 font-medium">行业</th>
                  <th className="px-2 py-2 text-right font-medium">市值(亿)</th>
                  <th className="px-2 py-2 text-right font-medium">近1月</th>
                  {FACTORS.slice(0, 4).map((f) => <th key={f.key} className="px-2 py-2 text-right font-medium">{f.label.split(" ")[0]}</th>)}
                  <th className="px-4 py-2" />
                </tr>
              </thead>
              <tbody>
                {hits.map((h) => (
                  <tr key={h.code} className="border-b border-line/60 hover:bg-panel-2/60">
                    <td className="num px-4 py-1.5 text-muted">{h.code}</td>
                    <td className="px-2 py-1.5">
                      <button className="hover:text-accent" onClick={() => { select(h.code); nav(`/stock?code=${h.code}`); }}>{h.name}</button>
                    </td>
                    <td className="px-2 py-1.5 text-muted">{h.industry}</td>
                    <td className="num px-2 py-1.5 text-right">{h.mcap != null ? h.mcap.toFixed(0) : "—"}</td>
                    <td className="px-2 py-1.5 text-right"><Delta value={h.ret_1m} digits={1} /></td>
                    {FACTORS.slice(0, 4).map((f) => (
                      <td key={f.key} className="px-2 py-1.5 text-right">
                        <div className="ml-auto flex w-16 items-center gap-1.5">
                          <div className="h-1 flex-1 overflow-hidden rounded-full bg-panel-2"><div className="h-full rounded-full bg-cil" style={{ width: `${(h[f.key] as number) * 100}%` }} /></div>
                          <span className="num w-6 text-right text-muted">{Math.round((h[f.key] as number) * 100)}</span>
                        </div>
                      </td>
                    ))}
                    <td className="px-4 py-1.5 text-right">
                      <button onClick={() => setPool(pool.includes(h.code) ? pool.filter((c) => c !== h.code) : [...pool, h.code])} className={clsx("rounded-md px-2 py-0.5 text-[11px]", pool.includes(h.code) ? "bg-accent text-white" : "border border-line text-muted hover:text-ink")}>
                        {pool.includes(h.code) ? "已加入" : "+ 候选"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}
