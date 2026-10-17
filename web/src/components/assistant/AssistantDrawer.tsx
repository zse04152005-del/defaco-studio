import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BadgeCheck,
  Bot,
  Brain,
  ChevronDown,
  CircleAlert,
  Copy,
  Loader2,
  History,
  Maximize2,
  MessageSquarePlus,
  Minimize2,
  Search,
  ScrollText,
  Send,
  ShieldAlert,
  Square,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import type { EChartsOption } from "echarts";
import { useAssistant, suggestions, type ChartMsg, type Msg, type Step } from "../../lib/assistant";
import { cssVar, useUI } from "../../lib/store";
import { EChart } from "../charts/EChart";
import { Dots } from "../ui";

const ARG_FMT = (a: Record<string, unknown>) =>
  Object.values(a)
    .map((v) => (Array.isArray(v) ? v.join("、") : typeof v === "object" && v ? Object.keys(v).join("、") : String(v)))
    .filter(Boolean)
    .join(" · ");

function Collapse({ icon, title, children, open: init = false }: { icon: React.ReactNode; title: React.ReactNode; children: React.ReactNode; open?: boolean }) {
  const [open, setOpen] = useState(init);
  return (
    <div className="rounded-xl border border-line bg-panel/60">
      <button className="flex w-full items-center gap-2 px-3 py-2 text-[11px] text-muted hover:text-ink" onClick={() => setOpen(!open)}>
        {icon}
        <span className="flex-1 text-left">{title}</span>
        <ChevronDown size={13} className={clsx("transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="border-t border-line px-3 py-2">{children}</div>}
    </div>
  );
}

/** Theme a server-built ECharts option: palette, "pct" formatters, candle / heatmap / radar colours. */
function themed(option: Record<string, unknown>): EChartsOption {
  const o = JSON.parse(JSON.stringify(option)) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  const pct = (v: unknown) => (typeof v === "number" ? `${(v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0)}%` : String(v ?? "—"));
  const walk = (x: any) => { // eslint-disable-line @typescript-eslint/no-explicit-any
    if (!x || typeof x !== "object") return;
    for (const k of Object.keys(x)) {
      if ((k === "formatter" || k === "valueFormatter") && x[k] === "pct") x[k] = pct;
      else walk(x[k]);
    }
  };
  walk(o);
  const text = cssVar("--text"), muted = cssVar("--muted"), grid = cssVar("--grid");
  o.color = ["--accent", "--m-sic", "--warn", "--m-dfl", "--accent-2", "--m-cil", "--m-fce", "--m-reg"].map(cssVar);
  if (o.title) o.title = { ...o.title, textStyle: { ...(o.title.textStyle ?? {}), color: text } };
  for (const ser of o.series ?? []) {
    if (ser.type === "candlestick") ser.itemStyle = { color: cssVar("--up"), color0: cssVar("--down"), borderColor: cssVar("--up"), borderColor0: cssVar("--down") };
    if (ser.type === "pie") ser.label = { ...(ser.label ?? {}), color: muted };
    if (ser.type === "heatmap") ser.label = { ...(ser.label ?? {}), color: text };
  }
  if (o.radar) o.radar = { ...o.radar, axisName: { color: muted, fontSize: 10 }, splitLine: { lineStyle: { color: grid } },
    splitArea: { areaStyle: { color: ["transparent"] } }, axisLine: { lineStyle: { color: grid } } };
  if (o.visualMap) o.visualMap = { ...o.visualMap, inRange: { color: [cssVar("--panel-2"), cssVar("--accent")] } };
  return o as EChartsOption;
}

function ChartCard({ c }: { c: ChartMsg }) {
  const theme = useUI((s) => s.theme);
  const option = useMemo(() => themed(c.option), [c.option, theme]); // eslint-disable-line react-hooks/exhaustive-deps
  const unchecked = c.check && c.check.total > c.check.verified;
  return (
    <figure className="rounded-xl border border-line bg-panel/60 p-2">
      <EChart option={option} height={c.kind === "correlation" ? 320 : 260} />
      <figcaption className="flex items-center gap-1.5 px-1 pt-1 text-[10.5px] text-faint">
        <span className="rounded bg-panel-2 px-1.5 text-ink">图 {c.no}</span>
        {c.kind === "bar" || c.kind === "line" ? (
          <span className={unchecked ? "text-[#f0a020]" : ""}>
            {unchecked ? `图中有 ${c.check!.total - c.check!.verified} 个数值未能在工具结果中核验` : "自定义图表 · 数值已核验"}
          </span>
        ) : (
          <span>数据由引擎直接生成</span>
        )}
      </figcaption>
    </figure>
  );
}

function StepRow({ s }: { s: Step }) {
  return (
    <div className="flex items-center gap-2 py-0.5 text-[11px]">
      {s.status === "running" ? (
        <Loader2 size={12} className="animate-spin text-accent" />
      ) : s.status === "ok" ? (
        <BadgeCheck size={12} className="text-[#22b07d]" />
      ) : (
        <CircleAlert size={12} className="text-[#f0a020]" />
      )}
      <span className="font-medium text-ink">{s.label}</span>
      <span className="truncate text-faint">{ARG_FMT(s.args)}</span>
      {s.error && <span className="truncate text-[#f0a020]">{s.error}</span>}
    </div>
  );
}

function AiMessage({ m }: { m: Msg }) {
  const steps = m.steps ?? [];
  const running = steps.filter((s) => s.status === "running").length;
  const f = m.final;
  const ratio = f && f.check.total ? f.check.verified / f.check.total : 1;
  return (
    <div className="space-y-2">
      {m.stages && m.stages.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
          {m.stages.map((st, i) => (
            <span key={st.key} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-faint">→</span>}
              <span className={clsx("flex items-center gap-1 rounded-full border px-2 py-0.5", st.status === "done" ? "border-[#22b07d]/50 text-[#22b07d]" : "border-accent text-accent")}>
                {st.status === "done" ? <BadgeCheck size={11} /> : <Loader2 size={11} className="animate-spin" />}
                {st.label}
              </span>
            </span>
          ))}
        </div>
      )}
      {(steps.length > 0 || (m.notes?.length ?? 0) > 0) && (
        <Collapse
          icon={running ? <Loader2 size={13} className="animate-spin text-accent" /> : <Wrench size={13} />}
          title={running ? `正在查询 ${running} 项数据…` : `执行过程 · 调用 ${steps.length} 次工具`}
          open={!!m.streaming && !m.text}
        >
          {m.notes?.map((n, i) => (
            <p key={i} className="mb-1 text-[11px] text-muted">{n}</p>
          ))}
          {steps.map((s) => (
            <StepRow key={s.id} s={s} />
          ))}
        </Collapse>
      )}
      {m.reasoning && (
        <Collapse icon={<Brain size={13} />} title="思考过程">
          <p className="max-h-48 overflow-y-auto whitespace-pre-wrap text-[11px] leading-relaxed text-faint">{m.reasoning}</p>
        </Collapse>
      )}
      {m.debate && (
        <Collapse icon={<ScrollText size={13} />} title="多头 / 空头分析师原稿">
          <div className="grid gap-2">
            <div>
              <div className="mb-1 text-[11px] font-semibold text-up">多头分析师</div>
              <div className="md text-[12px]"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.debate.bull}</ReactMarkdown></div>
            </div>
            <div>
              <div className="mb-1 text-[11px] font-semibold text-down">空头分析师</div>
              <div className="md text-[12px]"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.debate.bear}</ReactMarkdown></div>
            </div>
          </div>
        </Collapse>
      )}
      {m.text ? (
        <div className="md rounded-2xl bg-panel-2 px-3.5 py-2.5 text-[13px] leading-relaxed">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
          {m.streaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse rounded-sm bg-accent align-middle" />}
        </div>
      ) : m.streaming ? (
        <div className="flex items-center gap-3 rounded-2xl bg-panel-2 px-3.5 py-3 text-[12px] text-muted">
          <Dots /> {m.status}
        </div>
      ) : null}
      {m.charts?.map((c) => <ChartCard key={c.id} c={c} />)}
      {m.error && (
        <div className="flex items-start gap-2 rounded-xl border border-[#f0a020]/40 bg-[#f0a020]/10 px-3 py-2 text-[12px] text-ink">
          <CircleAlert size={14} className="mt-0.5 shrink-0 text-[#f0a020]" /> {m.error}
        </div>
      )}
      {f && (
        <div className="space-y-1.5 px-1">
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
            <span
              title={f.check.unverified.length ? `未能在工具结果中找到：${f.check.unverified.join("、")}（可能是推算值）` : "回答中的数字均可在工具结果中找到"}
              className={clsx("flex items-center gap-1 rounded-full px-2 py-0.5", ratio >= 0.95 ? "bg-[#22b07d]/15 text-[#22b07d]" : "bg-[#f0a020]/15 text-[#f0a020]")}
            >
              <BadgeCheck size={11} /> 数字核验 {f.check.verified}/{f.check.total}
            </span>
            {f.sources.map((s) => (
              <span key={s} className="rounded-full border border-line px-2 py-0.5 text-muted">{s}</span>
            ))}
            <button
              className="ml-auto rounded p-1 text-faint hover:text-ink"
              title="复制回答"
              onClick={() => navigator.clipboard?.writeText(m.text).catch(() => undefined)}
            >
              <Copy size={12} />
            </button>
          </div>
          {f.check.unverified.length > 0 && (
            <div className="text-[10.5px] text-faint">未直接核验的数字：{f.check.unverified.join("、")}（可能为推算或常识阈值）</div>
          )}
          {f.compliance.length > 0 && (
            <div className="flex items-center gap-1.5 text-[11px] text-[#f0a020]">
              <ShieldAlert size={12} /> 合规提醒：{f.compliance.join("；")}。请勿将其视为交易指令。
            </div>
          )}
          <div className="text-[10.5px] text-faint">
            {f.disclaimer} · tokens {f.usage.prompt_tokens + f.usage.completion_tokens}
          </div>
        </div>
      )}
    </div>
  );
}

function when(t: number) {
  const d = new Date(t);
  const today = new Date();
  const days = Math.floor((new Date(today.toDateString()).getTime() - new Date(d.toDateString()).getTime()) / 864e5);
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  return days === 0 ? `今天 ${hm}` : days === 1 ? `昨天 ${hm}` : `${d.getMonth() + 1}/${d.getDate()} ${hm}`;
}

function HistoryPanel({ onPick }: { onPick: () => void }) {
  const { convs, current, busy, switchConv, deleteConv } = useAssistant();
  const [f, setF] = useState("");
  const list = convs
    .filter((c) => c.msgs.some((m) => m.role === "user"))
    .filter((c) => !f.trim() || c.title.includes(f.trim()) || c.msgs.some((m) => m.role === "user" && m.text.includes(f.trim())));
  return (
    <div className="flex-1 overflow-y-auto p-3">
      <div className="mb-2 flex items-center gap-2 rounded-xl border border-line bg-panel-2 px-2.5 py-1.5 focus-within:border-accent">
        <Search size={13} className="text-faint" />
        <input value={f} onChange={(e) => setF(e.target.value)} placeholder="搜索历史问题" className="flex-1 bg-transparent text-[12px] outline-none" />
      </div>
      {list.length === 0 && <p className="py-8 text-center text-[12px] text-faint">{f ? "没有匹配的对话" : "还没有历史对话"}</p>}
      <div className="space-y-1">
        {list.map((c) => {
          const nq = c.msgs.filter((m) => m.role === "user").length;
          const nc = c.msgs.reduce((k, m) => k + (m.charts?.length ?? 0), 0);
          return (
            <div key={c.id} className={clsx("group flex items-center gap-2 rounded-xl px-3 py-2", c.id === current ? "bg-panel-2" : "hover:bg-panel-2/60")}>
              <button className="min-w-0 flex-1 text-left disabled:opacity-50" disabled={busy} onClick={() => { switchConv(c.id); onPick(); }}>
                <div className="truncate text-[12.5px] text-ink">{c.title}</div>
                <div className="text-[10.5px] text-faint">
                  {when(c.updated)} · {nq} 个问题{nc ? ` · ${nc} 张图` : ""}
                </div>
              </button>
              <button
                className="hidden rounded p-1 text-faint hover:text-ink disabled:opacity-40 group-hover:block"
                disabled={busy}
                onClick={() => deleteConv(c.id)}
                title="删除对话"
                aria-label={`删除对话 ${c.title}`}
              >
                <Trash2 size={13} />
              </button>
            </div>
          );
        })}
      </div>
      <p className="mt-3 px-1 text-[10.5px] text-faint">历史只保存在本浏览器（最多 40 个对话）。</p>
    </div>
  );
}

export function AssistantDrawer() {
  const { aiOpen, toggleAi } = useUI();
  const { msgs, busy, status, ctx, send, report, stop, newConv, loadStatus } = useAssistant();
  const [q, setQ] = useState("");
  const [wide, setWide] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (aiOpen) loadStatus();
  }, [aiOpen, loadStatus]);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [msgs, aiOpen]);

  const submit = (text: string) => {
    if (!text.trim() || busy) return;
    setQ("");
    send(text);
  };
  const tips = suggestions(ctx);
  const notReady = status && !status.ready;

  return (
    <div
      className={clsx(
        "fixed inset-y-0 right-0 z-40 flex max-w-[96vw] flex-col border-l border-line bg-panel shadow-2xl transition-[transform,width] duration-300",
        wide ? "w-[680px]" : "w-[420px]",
        aiOpen ? "translate-x-0" : "translate-x-full",
      )}
      aria-hidden={!aiOpen}
    >
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <Bot size={18} className="text-accent-2" />
        <div className="flex-1">
          <div className="text-sm font-semibold">AI 投研助手</div>
          <div className="flex items-center gap-1.5 text-[10.5px] text-faint">
            <i className={clsx("h-1.5 w-1.5 rounded-full", status?.ready ? "bg-[#22b07d]" : "bg-[#f0a020]")} />
            <span title={status?.ready ? `问答 ${status.model} · 研报主编 ${status.model_pro}` : undefined}>{status ? (status.ready ? `${status.provider ?? "大模型"} · ${status.model}` : "未配置大模型") : "连接中…"}</span>
          </div>
        </div>
        <button
          className={clsx("rounded-lg p-1.5 hover:bg-panel-2", showHistory ? "bg-panel-2 text-ink" : "text-muted")}
          onClick={() => setShowHistory(!showHistory)}
          title="历史对话"
          aria-label="历史对话"
          aria-pressed={showHistory}
        >
          <History size={15} />
        </button>
        <button
          className="rounded-lg p-1.5 text-muted hover:bg-panel-2 disabled:opacity-40"
          onClick={() => { newConv(); setShowHistory(false); }}
          disabled={busy}
          title="新对话"
          aria-label="新对话"
        >
          <MessageSquarePlus size={15} />
        </button>
        <button className="rounded-lg p-1.5 text-muted hover:bg-panel-2" onClick={() => setWide(!wide)} title={wide ? "收窄" : "展开"} aria-label="切换宽度">
          {wide ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
        </button>
        <button className="rounded-lg p-1.5 text-muted hover:bg-panel-2" onClick={() => toggleAi(false)} aria-label="关闭">
          <X size={16} />
        </button>
      </div>

      {showHistory && <HistoryPanel onPick={() => setShowHistory(false)} />}
      <div className={clsx("flex-1 space-y-4 overflow-y-auto p-4", showHistory && "hidden")}>
        {notReady && (
          <div className="rounded-xl border border-[#f0a020]/40 bg-[#f0a020]/10 p-3 text-[12px]">
            {status?.problem ?? "未配置大模型。"} 在仓库根目录的 <code>.env</code> 中设置 <code>LLM_PROVIDER</code>、<code>LLM_API_KEY</code>（可选 <code>LLM_MODEL</code>）后重启 API 服务；支持 DeepSeek、OpenAI、通义千问、Kimi、智谱 GLM、SiliconFlow、OpenRouter、本地 Ollama 及其他 OpenAI 兼容接口。
          </div>
        )}
        {msgs.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="ml-auto w-fit max-w-[88%] whitespace-pre-wrap rounded-2xl bg-accent px-3.5 py-2 text-[13px] text-white">
              {m.text}
            </div>
          ) : (
            <AiMessage key={m.id} m={m} />
          ),
        )}
        <div ref={end} />
      </div>

      <div className="flex flex-wrap gap-1.5 px-4 pb-2">
        {tips.map((t) => (
          <button
            key={t.label}
            disabled={busy || !!notReady}
            onClick={() => (t.report && ctx.code ? report(ctx.code, ctx.name) : submit(t.label))}
            className={clsx(
              "rounded-full border px-2.5 py-1 text-[11px] disabled:opacity-40",
              t.report ? "border-accent-2/60 text-accent-2 hover:bg-accent-2/10" : "border-line text-muted hover:border-accent hover:text-ink",
            )}
          >
            {t.report && "✦ "}
            {t.label}
          </button>
        ))}
      </div>
      <div className="border-t border-line p-3">
        <div className="flex items-end gap-2 rounded-2xl border border-line bg-panel-2 px-3 py-2 focus-within:border-accent">
          <textarea
            rows={1}
            value={q}
            placeholder={busy ? "正在回答…" : "向助手提问（Enter 发送，Shift+Enter 换行）"}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit(q);
              }
            }}
            className="max-h-32 min-h-[22px] flex-1 resize-none bg-transparent text-[13px] outline-none placeholder:text-faint"
          />
          {busy ? (
            <button onClick={stop} className="rounded-lg bg-panel p-1.5 text-ink hover:bg-line" title="停止" aria-label="停止">
              <Square size={14} />
            </button>
          ) : (
            <button onClick={() => submit(q)} disabled={!q.trim()} className="rounded-lg bg-accent p-1.5 text-white disabled:opacity-40" title="发送" aria-label="发送">
              <Send size={14} />
            </button>
          )}
        </div>
        <p className="mt-1.5 text-[10px] text-faint">
          发送给{status?.provider ?? "大模型服务商"}的只有问题、页面上下文、自选 / 候选池 / 持仓的股票名称和引擎汇总数据；持仓股数与金额只留在本地服务。仅供研究与教学。
        </p>
      </div>
    </div>
  );
}
