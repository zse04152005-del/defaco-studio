import clsx from "clsx";
import { useMemo, useState } from "react";
import { Download, Trash2 } from "lucide-react";
import type { EChartsOption } from "echarts";
import { useData, type Portfolio, type Security } from "../lib/data";
import { modelOptions, paramsFrom, useDefacoAvailable, useLivePortfolio, usePortfolioSettings } from "../lib/portfolio";
import { useQuotes, useRT, type Quote } from "../lib/realtime";
import { useLists } from "../lib/securities";
import { cssVar } from "../lib/store";
import { pct } from "../lib/format";
import { EChart } from "../components/charts/EChart";
import { DataBanner, InfinityLoader, PageHeader, Panel, RimButton, Segmented, Stat, Switch, UploadButton } from "../components/ui";

const COMMISSION_BPS = 2.5; // per side
const MIN_COMMISSION = 5; // yuan per order
const STAMP_BPS = 5; // sell side, since 2023-08-28

type Freeze = "limit_up" | "limit_down" | "suspended" | null;
const FREEZE_CN: Record<Exclude<Freeze, null>, string> = { limit_up: "涨停", limit_down: "跌停", suspended: "停牌" };

/** Trading restriction from a live quote. Suspension needs an opened session (pre-market every
 * stock has no open price and no volume). */
function freezeOf(q: Quote | undefined, sessionOpened: boolean): Freeze {
  if (!q) return null;
  if (sessionOpened && q.no_trade) return "suspended";
  if (q.at_limit_up) return "limit_up";
  if (q.at_limit_down) return "limit_down";
  return null;
}

/** A-share lot rules: main board 100-share lots; STAR board min 200 then 1-share steps. */
function roundLot(shares: number, star: boolean) {
  if (shares <= 0) return 0;
  if (star) return shares < 200 ? 0 : Math.floor(shares);
  return Math.floor(shares / 100) * 100;
}

export default function Rebalance() {
  const { data: port, loading } = useData<Portfolio>("portfolio.json");
  const { data: secs } = useData<{ items: Security[] }>("securities.json");
  const settings = usePortfolioSettings();
  const defacoOn = useDefacoAvailable();          // same model and caps as the 组合构建 page
  const { model } = settings;
  const setModel = (v: string) => settings.set({ model: v });
  const live = useLivePortfolio(paramsFrom(settings, model));
  const targetW = live.data?.weights ?? port?.models[model]?.weights;
  const targetLabel = live.data?.label ?? port?.models[model]?.label ?? model;
  // holdings and cash persist in the browser and are shared with the stock page's 持仓 tab
  const { holdings: pos, setHoldings: setPos, cash, setCash } = useLists();
  const [newCode, setNewCode] = useState("");

  const [useLive, setUseLive] = useState(true);
  const [useFreeze, setUseFreeze] = useState(true);
  const session = useRT((s) => s.session);
  // the open auction has happened: trading, lunch break, closing auction or after the close (not pre-market)
  const sessionOpened = !!session && session.code !== "pre" && !(session.code === "auction" && new Date().getHours() < 10);
  const liveCodes = useMemo(() => {
    const t = targetW?.map((w) => w.code) ?? [];
    return [...new Set([...t, ...pos.map((p) => p.code)])].sort();
  }, [targetW, pos]);
  const quotes = useQuotes(liveCodes);
  const nLive = liveCodes.filter((c) => quotes[c]?.price).length;
  const sec = useMemo(() => {
    const m = new Map((secs?.items ?? []).map((s) => [s.code, { ...s }]));
    if (useLive) for (const [c, q] of Object.entries(quotes)) { const s = m.get(c); if (s && q.price) s.close = q.price; }
    return m;
  }, [secs, quotes, useLive]);

  const plan = useMemo(() => {
    if (!targetW || !secs) return null;
    const target = targetW;
    const cur = pos.filter((p) => sec.has(p.code)).map((p) => ({ ...p, price: sec.get(p.code)!.close }));
    const nav = cur.reduce((s, p) => s + p.shares * p.price, 0) + cash;
    const codes = [...new Set([...cur.map((p) => p.code), ...target.map((t) => t.code)])];
    const rows = codes.map((code) => {
      const s = sec.get(code)!;
      const held = cur.find((p) => p.code === code)?.shares ?? 0;
      const w = target.find((t) => t.code === code)?.w ?? 0;
      const ideal = roundLot((w * nav) / s.close, s.board === "STAR") - held;
      const fz = useLive && useFreeze ? freezeOf(quotes[code], sessionOpened) : null;
      // frozen: no buying at limit-up, no selling at limit-down, nothing while suspended
      const blocked = fz === "suspended" || (fz === "limit_up" && ideal > 0) || (fz === "limit_down" && ideal < 0);
      const trade = blocked ? 0 : ideal;
      return { code, name: s.name, industry: s.industry, price: s.close, held, wCur: (held * s.close) / nav, w, tgtShares: held + trade, trade,
               value: trade * s.close, freeze: fz, blocked, ideal, star: s.board === "STAR" };
    });
    // frozen sells raise no cash: scale the remaining buys down to what the account can pay for
    const fee = (v: number) => Math.max(MIN_COMMISSION, (Math.abs(v) * COMMISSION_BPS) / 1e4);
    const sellCash = rows.filter((r) => r.trade < 0).reduce((s, r) => s - r.value - fee(r.value) - (-r.value * STAMP_BPS) / 1e4, 0);
    const wantBuy = rows.filter((r) => r.trade > 0).reduce((s, r) => s + r.value + fee(r.value), 0);
    const scale = wantBuy > 0 ? Math.min(1, (cash + sellCash) / wantBuy) : 1;
    if (scale < 1)
      for (const r of rows.filter((x) => x.trade > 0)) {
        r.trade = roundLot(r.trade * scale, r.star);
        r.tgtShares = r.held + r.trade;
        r.value = r.trade * r.price;
      }
    const orders = rows.filter((r) => r.trade !== 0);
    const buys = orders.filter((r) => r.trade > 0).reduce((s, r) => s + r.value, 0);
    const sells = -orders.filter((r) => r.trade < 0).reduce((s, r) => s + r.value, 0);
    const commission = orders.reduce((s, r) => s + Math.max(MIN_COMMISSION, (Math.abs(r.value) * COMMISSION_BPS) / 1e4), 0);
    const stamp = (sells * STAMP_BPS) / 1e4;
    const invested = rows.reduce((s, r) => s + r.tgtShares * r.price, 0);
    const frozen = rows.filter((r) => r.blocked);
    return { rows: rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || Number(b.blocked) - Number(a.blocked)), nav, buys, sells, commission, stamp,
             cashLeft: nav - invested - commission - stamp, turnover: (buys + sells) / nav, frozen, scale };
  }, [port, secs, pos, cash, model, sec, quotes, useLive, useFreeze, sessionOpened]);

  if (loading || !port || !plan) return <div className="grid h-96 place-items-center"><InfinityLoader label="生成交易单" /></div>;

  // industry flow sankey: sells by industry -> buys by industry
  const sellBy: Record<string, number> = {}, buyBy: Record<string, number> = {};
  plan.rows.forEach((r) => {
    if (r.value < 0) sellBy[r.industry] = (sellBy[r.industry] ?? 0) - r.value;
    if (r.value > 0) buyBy[r.industry] = (buyBy[r.industry] ?? 0) + r.value;
  });
  // keep the 8 largest industries on each side, fold the rest into "其他"
  const fold = (m: Record<string, number>) => {
    const top = Object.entries(m).sort((x, y) => y[1] - x[1]);
    const out: Record<string, number> = Object.fromEntries(top.slice(0, 8));
    const rest = top.slice(8).reduce((s, [, v]) => s + v, 0);
    if (rest > 0) out["其他"] = rest;
    return out;
  };
  Object.assign(buyBy, fold(buyBy));
  Object.keys(buyBy).forEach((k) => { if (!(k in fold(buyBy))) delete buyBy[k]; });
  const totalBuy = Object.values(buyBy).reduce((a, b) => a + b, 0) || 1;
  const sources = { ...sellBy, 现金: Math.max(0, plan.buys - plan.sells) };
  const links = Object.entries(sources).flatMap(([s, sv]) =>
    Object.entries(buyBy).map(([b, bv]) => ({ source: `卖·${s}`, target: `买·${b}`, value: (sv * bv) / totalBuy })),
  ).filter((l) => l.value > 1);
  const sankey: EChartsOption = {
    tooltip: { trigger: "item", valueFormatter: (v) => `${((v as number) / 1e4).toFixed(1)} 万元` },
    series: [
      {
        type: "sankey",
        left: 8,
        right: 90,
        nodeGap: 6,
        nodeWidth: 10,
        data: [...new Set(links.flatMap((l) => [l.source, l.target]))].map((n) => ({
          name: n,
          itemStyle: { color: n.startsWith("卖") ? cssVar("--down") : cssVar("--up") },
        })),
        links,
        label: { color: cssVar("--muted"), fontSize: 10 },
        lineStyle: { color: "gradient", opacity: 0.35, curveness: 0.5 },
      },
    ],
  };

  const exportCsv = () => {
    const head = "代码,名称,方向,股数,参考价,金额\n";
    const body = plan.rows.filter((r) => r.trade !== 0).map((r) => `${r.code},${r.name},${r.trade > 0 ? "买入" : "卖出"},${Math.abs(r.trade)},${r.price},${Math.abs(r.value).toFixed(2)}`).join("\n");
    const url = URL.createObjectURL(new Blob(["﻿" + head + body], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `交易单_${model}_${port.asof}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <PageHeader
        title="持仓与调仓"
        desc={`参考价：${useLive && nLive ? `实时价（${nLive}/${liveCodes.length} 只有报价）` : `${port.asof} 收盘价`} · 主板 100 股/手，科创板 200 股起 · 佣金 ${COMMISSION_BPS}bp（最低 ${MIN_COMMISSION} 元）+ 卖出印花税 ${STAMP_BPS}bp`}
        actions={
          <>
            <Switch checked={useLive} onChange={setUseLive} label="用实时价" />
            <Switch checked={useFreeze && useLive} onChange={setUseFreeze} label="涨跌停/停牌冻结" />
            <Segmented name="rb-model" value={model} onChange={setModel} options={modelOptions(port?.models, defacoOn)} />
            <RimButton onClick={exportCsv}><Download size={14} /> 导出交易单</RimButton>
          </>
        }
      />
      <DataBanner note="仅生成交易单，不连接任何券商下单接口。" />
      {useLive && useFreeze && (plan.frozen.length > 0 || plan.scale < 1) && (
        <div className="mb-4 rounded-xl border border-warn/40 bg-warn/10 px-4 py-2.5 text-[12px]">
          <b>交易限制已生效</b>（依据 {session?.label ?? "最新"} 实时行情）：
          {plan.frozen.map((r) => `${r.name}${FREEZE_CN[r.freeze as Exclude<Freeze, null>]}，原计划${r.ideal > 0 ? "买入" : "卖出"} ${Math.abs(r.ideal)} 股冻结`).join("；")}
          {plan.scale < 1 && `${plan.frozen.length ? "；" : ""}卖出受限导致资金不足，其余买单按 ${(plan.scale * 100).toFixed(0)}% 等比例缩减并重新取整手`}
          。
        </div>
      )}
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="账户总资产" value={`${(plan.nav / 1e4).toFixed(2)} 万`} />
        <Stat label="买入金额" value={<span className="text-up">{(plan.buys / 1e4).toFixed(2)} 万</span>} />
        <Stat label="卖出金额" value={<span className="text-down">{(plan.sells / 1e4).toFixed(2)} 万</span>} />
        <Stat label="换手率" value={pct(plan.turnover, 1)} />
        <Stat label="交易成本" value={`${(plan.commission + plan.stamp).toFixed(0)} 元`} hint={`佣金 ${plan.commission.toFixed(0)} · 印花税 ${plan.stamp.toFixed(0)}`} />
        <Stat label="调仓后现金" value={`${(plan.cashLeft / 1e4).toFixed(2)} 万`} hint="整手取整后的余额" />
      </div>
      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="当前持仓" actions={<UploadButton>导入 CSV</UploadButton>} bodyClass="p-3">
          <label className="mb-3 flex items-center gap-2 text-xs text-muted">
            可用现金
            <input type="number" value={cash} step={10000} onChange={(e) => setCash(Math.max(0, +e.target.value))} className="num h-8 flex-1 rounded-lg border border-line bg-bg-2 px-2 text-ink outline-none focus:border-accent" />
          </label>
          <div className="space-y-1.5">
            {pos.map((p, i) => {
              const s = sec.get(p.code);
              return (
                <div key={p.code} className="flex items-center gap-2 rounded-lg border border-line px-2.5 py-1.5 text-xs">
                  <div className="min-w-0 flex-1">
                    <div className="truncate">{s?.name ?? p.code}</div>
                    <div className="num text-[10px] text-faint">{p.code} · {s ? s.close.toFixed(2) : "无价格"}</div>
                  </div>
                  <input type="number" step={100} value={p.shares} onChange={(e) => setPos(pos.map((q, j) => (j === i ? { ...q, shares: Math.max(0, +e.target.value) } : q)))} className="num h-7 w-20 rounded-md border border-line bg-bg-2 px-1.5 text-right outline-none focus:border-accent" />
                  <button onClick={() => setPos(pos.filter((_, j) => j !== i))} className="text-faint hover:text-down" aria-label="删除"><Trash2 size={14} /></button>
                </div>
              );
            })}
          </div>
          <div className="mt-2 flex gap-2">
            <input value={newCode} onChange={(e) => setNewCode(e.target.value.toUpperCase())} placeholder="代码，如 601888.SH" className="h-8 flex-1 rounded-lg border border-line bg-bg-2 px-2 text-xs outline-none focus:border-accent" />
            <button
              onClick={() => { if (sec.has(newCode) && !pos.find((p) => p.code === newCode)) { setPos([...pos, { code: newCode, shares: 100 }]); setNewCode(""); } }}
              className="rounded-lg bg-panel-2 px-3 text-xs hover:text-accent"
            >
              添加
            </button>
          </div>
        </Panel>
        <Panel title="资金流向" subtitle="卖出行业 → 买入行业（按金额）" className="xl:col-span-2" bodyClass="p-2">
          <EChart option={sankey} height={330} />
        </Panel>
      </div>
      <Panel className="mt-4" title="建议交易单" subtitle={`目标：${targetLabel} · 最小方差（参数与组合构建页一致）`} bodyClass="p-0">
        <div className="max-h-[440px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-panel text-left text-faint">
              <tr className="border-b border-line">
                {["代码", "名称", "行业", "参考价", "当前股数", "当前权重", "目标权重", "目标股数", "交易", "金额"].map((h, i) => (
                  <th key={h} className={clsx("py-2 font-medium", i === 0 ? "px-4" : "px-2", i >= 3 && "text-right")}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {plan.rows.map((r) => (
                <tr key={r.code} className="border-b border-line/60 hover:bg-panel-2/60">
                  <td className="num px-4 py-1.5 text-muted">{r.code}</td>
                  <td className="px-2 py-1.5">
                    {r.name}
                    {r.freeze && (
                      <span className={clsx("ml-1.5 rounded px-1 text-[10px]", r.blocked ? "bg-warn/20 text-warn" : "border border-line text-faint")}>
                        {FREEZE_CN[r.freeze]}
                      </span>
                    )}
                  </td>
                  <td className="px-2 py-1.5 text-muted">{r.industry}</td>
                  <td className="num px-2 py-1.5 text-right">{r.price.toFixed(2)}</td>
                  <td className="num px-2 py-1.5 text-right text-muted">{r.held}</td>
                  <td className="num px-2 py-1.5 text-right text-muted">{pct(r.wCur, 1)}</td>
                  <td className="num px-2 py-1.5 text-right">{pct(r.w, 1)}</td>
                  <td className="num px-2 py-1.5 text-right">{r.tgtShares}</td>
                  <td className={clsx("num px-2 py-1.5 text-right font-semibold", r.trade > 0 ? "text-up" : r.trade < 0 ? "text-down" : "text-faint")}>
                    {r.blocked ? (
                      <span className="font-normal text-warn" title={`原计划${r.ideal > 0 ? "买入" : "卖出"} ${Math.abs(r.ideal)} 股`}>冻结</span>
                    ) : r.trade === 0 ? "—" : `${r.trade > 0 ? "买入" : "卖出"} ${Math.abs(r.trade)}`}
                  </td>
                  <td className="num px-4 py-1.5 text-right">{r.value === 0 ? "—" : `${(Math.abs(r.value) / 1e4).toFixed(2)} 万`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
