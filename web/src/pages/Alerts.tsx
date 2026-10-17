import clsx from "clsx";
import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { BellRing, CircleAlert, Clock, MessageSquare, Play, Send, Trash2 } from "lucide-react";
import { monitorApi, type AlertEvent, type MonitorConfig } from "../lib/monitor";
import { Badge, PageHeader, Panel, RimButton, Switch } from "../components/ui";
import { useDefacoAvailable } from "../lib/portfolio";

// editable thresholds per rule: [param, label, kind]; "pct" params are fractions shown as %
const PARAMS: Record<string, [string, string, "pct" | "bool" | "model"][]> = {
  vol: [["threshold", "预期年化波动上限", "pct"], ["model", "风险模型", "model"]],
  drift: [["stock", "单股偏离", "pct"], ["industry", "行业偏离", "pct"]],
  regime: [["threshold", "沪深300 20 日年化波动阈值", "pct"]],
  limit: [["include_watch", "同时监控自选股", "bool"]],
  data: [],
};
const RULE_DESC: Record<string, string> = {
  vol: "按实时价计算持仓权重，用风险模型估计年化波动；超过阈值时推送，并列出风险贡献最大的股票",
  drift: "持仓与「组合构建」当前参数下的目标组合比较，单股偏离超过阈值时推送",
  regime: "高 / 低波动区制发生切换时推送（只在状态变化时）",
  limit: "持仓（及自选）出现涨停、跌停或停牌时推送；调仓页同步冻结对应交易",
  data: "行情数据源请求失败时推送",
};
const MODELS = { defaco: "DeFaCo（论文 M4）", lw_nonlinear: "非线性收缩 NLS", lw_linear: "线性收缩 LW", pca_factor: "PCA 因子模型", sample: "样本协方差", ewma: "EWMA" };
const LEVEL = { critical: ["严重", "var(--down)"], warn: ["预警", "var(--warn)"], info: ["提示", "var(--accent)"] } as const;
const HINT: Record<string, string> = {
  wecom: "群聊 → 添加群机器人 → 复制 Webhook 地址（https://qyapi.weixin.qq.com/…）",
  feishu: "群设置 → 群机器人 → 自定义机器人；如开启“签名校验”，把密钥填在下面",
  dingtalk: "群设置 → 智能群助手 → 自定义机器人；安全设置选“加签”时填写密钥",
};

const fmt = (t: string | null) => (t ? t.replace("T", " ").slice(0, 16) : "—");

export default function Alerts() {
  const defacoOn = useDefacoAvailable();
  const [cfg, setCfg] = useState<MonitorConfig | null>(null);
  const [events, setEvents] = useState<AlertEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState("");
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [secrets, setSecrets] = useState<Record<string, string>>({});

  const load = useCallback(async () => {
    try {
      const [c, e] = await Promise.all([monitorApi.config(), monitorApi.events(60)]);
      setCfg(c);
      setEvents(e.events);
      setErr(null);
    } catch (e) {
      setErr(`监控服务不可用：${(e as Error).message}（请确认 API 已启动：scripts/dev.sh）`);
    }
  }, []);
  useEffect(() => {
    load();
    const id = window.setInterval(load, 30000);
    return () => window.clearInterval(id);
  }, [load]);

  const update = async (patch: unknown, note?: string) => {
    try {
      setCfg(await monitorApi.update(patch));
      if (note) setMsg(note);
    } catch (e) {
      setMsg(`保存失败：${(e as Error).message}`);
    }
  };
  const act = async (key: string, f: () => Promise<void>) => {
    setBusy(key);
    try {
      await f();
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  if (!cfg)
    return (
      <div>
        <PageHeader title="监控与预警" desc="定时任务 + 规则引擎 + 企业微信 / 飞书 / 钉钉推送" />
        <div className="rounded-xl border border-line p-6 text-sm text-muted">{err ?? "加载中…"}</div>
      </div>
    );

  const anyChannel = Object.values(cfg.channels).some((c) => c.enabled && c.configured);
  return (
    <div>
      <PageHeader
        title="监控与预警"
        desc={`定时任务 + 规则引擎，触发后推送到企业微信 / 飞书 / 钉钉 · 市场状态：${cfg.session.label} · 上次检查 ${fmt(cfg.last_run)}`}
        actions={
          <>
            <RimButton onClick={() => act("run", async () => {
              const r = await monitorApi.run({ push: anyChannel });
              setMsg(r.events.length ? `检查完成：新增 ${r.events.length} 条事件${anyChannel ? "，已推送" : "（未配置推送渠道，只记录）"}` : "检查完成：没有新的预警（同一事件每天只提醒一次）");
              await load();
            })}>
              <Play size={14} /> {busy === "run" ? "检查中…" : "立即检查"}
            </RimButton>
            <RimButton disabled={!anyChannel} title={anyChannel ? undefined : "先在右侧启用并保存一个推送渠道"} onClick={() => act("test", async () => {
              const r = await monitorApi.test();
              setMsg("测试推送：" + Object.entries(r.results).map(([k, v]) => `${cfg.channels[k]?.name ?? k} ${v === "ok" ? "成功" : `失败（${v}）`}`).join("；"));
            })}>
              <Send size={14} /> {busy === "test" ? "发送中…" : "发送测试推送"}
            </RimButton>
          </>
        }
      />
      {msg && (
        <div className="mb-4 flex items-start gap-2 rounded-xl border border-line bg-panel-2 px-4 py-2.5 text-[12px]">
          <CircleAlert size={14} className="mt-0.5 shrink-0 text-accent" />
          <span className="flex-1">{msg}</span>
          <button className="text-faint hover:text-ink" onClick={() => setMsg(null)}>关闭</button>
        </div>
      )}
      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="预警规则" subtitle="阈值修改后立即生效" className="xl:col-span-2" bodyClass="p-2">
          {Object.entries(cfg.rules).map(([id, r]) => (
            <div key={id} className="flex items-start gap-4 rounded-xl px-3 py-3 hover:bg-panel-2/60">
              <BellRing size={18} className={clsx("mt-0.5 shrink-0", r.enabled ? "text-accent" : "text-faint")} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 text-sm">
                  {r.name} <Badge>{r.when}</Badge>
                </div>
                <div className="text-xs text-muted">{RULE_DESC[id]}</div>
                {PARAMS[id].length > 0 && (
                  <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-muted">
                    {PARAMS[id].map(([p, label, kind]) => (
                      <label key={p} className="flex items-center gap-1.5">
                        {label}
                        {kind === "pct" ? (
                          <span className="flex items-center gap-1">
                            <input
                              type="number"
                              step={0.5}
                              min={0}
                              defaultValue={+(Number(r.params[p]) * 100).toFixed(2)}
                              onBlur={(e) => update({ rules: { [id]: { params: { [p]: +e.target.value / 100 } } } }, `已保存：${r.name} · ${label} ${e.target.value}%`)}
                              className="num h-7 w-16 rounded-md border border-line bg-bg-2 px-1.5 text-right text-ink outline-none focus:border-accent"
                            />
                            %
                          </span>
                        ) : kind === "model" ? (
                          <select
                            value={String(r.params[p])}
                            onChange={(e) => update({ rules: { [id]: { params: { [p]: e.target.value } } } })}
                            className="h-7 rounded-md border border-line bg-bg-2 px-1.5 text-ink outline-none"
                          >
                            {Object.entries(MODELS).filter(([k]) => defacoOn || k !== "defaco").map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                          </select>
                        ) : (
                          <Switch checked={!!r.params[p]} onChange={(v) => update({ rules: { [id]: { params: { [p]: v } } } })} />
                        )}
                      </label>
                    ))}
                  </div>
                )}
              </div>
              <Switch checked={r.enabled} onChange={(v) => update({ rules: { [id]: { enabled: v } } })} />
            </div>
          ))}
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="推送渠道" subtitle="Webhook 只保存在本机 state/monitor.json（不提交仓库）" bodyClass="p-3 space-y-3">
            {Object.entries(cfg.channels).map(([k, c]) => (
              <div key={k} className="rounded-xl border border-line p-3">
                <div className="mb-1.5 flex items-center gap-2">
                  <MessageSquare size={15} className="text-muted" />
                  <span className="flex-1 text-sm">{c.name}</span>
                  {c.configured && <Badge color="var(--m-sic)">{c.from_env ? ".env 已配置" : "已配置"}</Badge>}
                  <Switch checked={c.enabled} onChange={(v) => update({ channels: { [k]: { enabled: v } } })} />
                </div>
                <p className="mb-1.5 text-[10.5px] text-faint">{HINT[k]}</p>
                <div className="flex gap-1.5">
                  <input
                    type="password"
                    autoComplete="off"
                    value={urls[k] ?? ""}
                    placeholder={c.configured ? `已保存：${c.url_masked}` : "粘贴 Webhook 地址"}
                    onChange={(e) => setUrls({ ...urls, [k]: e.target.value })}
                    disabled={c.from_env}
                    className="h-8 min-w-0 flex-1 rounded-lg border border-line bg-bg-2 px-2 text-[11px] outline-none focus:border-accent disabled:opacity-50"
                  />
                  {k !== "wecom" && (
                    <input
                      type="password"
                      autoComplete="off"
                      value={secrets[k] ?? ""}
                      placeholder={c.has_secret ? "密钥已保存" : "签名密钥（可选）"}
                      onChange={(e) => setSecrets({ ...secrets, [k]: e.target.value })}
                      disabled={c.from_env}
                      className="h-8 w-28 rounded-lg border border-line bg-bg-2 px-2 text-[11px] outline-none focus:border-accent disabled:opacity-50"
                    />
                  )}
                </div>
                <div className="mt-2 flex gap-2 text-[11px]">
                  <button
                    className="rounded-md border border-line px-2 py-1 hover:border-accent disabled:opacity-40"
                    disabled={!urls[k] && !secrets[k]}
                    onClick={async () => {
                      await update({ channels: { [k]: { url: urls[k] ?? "", secret: secrets[k] ?? "", enabled: true } } }, `${c.name} 已保存并启用`);
                      setUrls({ ...urls, [k]: "" });
                      setSecrets({ ...secrets, [k]: "" });
                    }}
                  >
                    保存
                  </button>
                  <button
                    className="rounded-md border border-line px-2 py-1 hover:border-accent disabled:opacity-40"
                    disabled={!c.configured || busy === `test-${k}`}
                    onClick={() => act(`test-${k}`, async () => {
                      const r = await monitorApi.test(k);
                      setMsg(`${c.name} 测试推送：${r.results[k] === "ok" ? "成功，请到群里查看" : `失败（${r.results[k]}）`}`);
                    })}
                  >
                    {busy === `test-${k}` ? "发送中…" : "测试"}
                  </button>
                  {c.configured && !c.from_env && (
                    <button className="ml-auto text-faint hover:text-down" onClick={() => update({ channels: { [k]: { clear: true, enabled: false } } }, `${c.name} 已清除`)}>
                      清除
                    </button>
                  )}
                </div>
              </div>
            ))}
          </Panel>

          <Panel title="定时任务" bodyClass="p-3 space-y-2.5 text-[12px]">
            <div className="flex items-center gap-2">
              <Clock size={14} className="text-muted" />
              <span className="flex-1">启用定时检查（工作日）</span>
              <Switch checked={cfg.schedule.enabled} onChange={(v) => update({ schedule: { enabled: v } })} />
            </div>
            <label className="flex items-center gap-2 text-muted">
              盘中每
              <input
                type="number"
                min={1}
                max={60}
                defaultValue={cfg.schedule.intraday_minutes}
                onBlur={(e) => update({ schedule: { intraday_minutes: +e.target.value } }, "已保存盘中检查间隔")}
                className="num h-7 w-14 rounded-md border border-line bg-bg-2 px-1.5 text-right text-ink outline-none focus:border-accent"
              />
              分钟检查涨跌停 / 区制 / 数据源
            </label>
            <label className="flex items-center gap-2 text-muted">
              收盘后
              <input
                type="time"
                defaultValue={cfg.schedule.eod_time}
                onBlur={(e) => e.target.value && update({ schedule: { eod_time: e.target.value } }, "已保存收盘检查时间")}
                className="num h-7 rounded-md border border-line bg-bg-2 px-1.5 text-ink outline-none focus:border-accent"
              />
              运行全部规则
            </label>
            <p className="text-[10.5px] text-faint">
              今日收盘检查：{cfg.eod_done === new Date().toLocaleDateString("sv-SE") ? "已完成" : "未运行"} · 持仓 / 自选同步于 {fmt(cfg.client_synced)}。定时任务运行在本地 API 服务中，需保持服务开启。
            </p>
          </Panel>
        </div>
      </div>

      <Panel
        className="mt-4"
        title={`最近事件（${events.length}）`}
        subtitle="同一事件每天只提醒一次；区制只在切换时提醒"
        actions={events.length > 0 ? <button className="flex items-center gap-1 text-[11px] text-faint hover:text-ink" onClick={async () => { await monitorApi.clear(); load(); }}><Trash2 size={12} /> 清空</button> : undefined}
      >
        {events.length === 0 ? (
          <p className="py-6 text-center text-xs text-faint">还没有事件。点击右上角「立即检查」运行一次。</p>
        ) : (
          <div className="space-y-2">
            {events.map((e) => {
              const [lv, color] = LEVEL[e.level] ?? LEVEL.info;
              const pushed = Object.entries(e.pushed ?? {});
              return (
                <div key={e.id} className="rounded-lg border border-line px-3 py-2 text-xs">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="num text-faint">{fmt(e.at)}</span>
                    <Badge color={color}>{lv}</Badge>
                    <span className="font-medium text-ink">{e.title}</span>
                    <span className="ml-auto text-[10.5px] text-faint">
                      {pushed.length
                        ? pushed.map(([k, v]) => `${cfg.channels[k]?.name ?? k}${v === "ok" ? " ✓" : ` ✗ ${v}`}`).join(" · ")
                        : "未推送（只记录）"}
                    </span>
                  </div>
                  <div className="md mt-1 text-muted"><ReactMarkdown>{e.body}</ReactMarkdown></div>
                </div>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
