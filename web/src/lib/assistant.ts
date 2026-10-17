// AI assistant client: POST + Server-Sent Events from /api/assistant/{chat,report}.
// The conversation is kept in a zustand store (and in localStorage as a per-browser convenience).
import { create } from "zustand";
import { usePortfolioSettings } from "./portfolio";
import { useLists } from "./securities";

export interface Step {
  id: string;
  name: string;
  label: string;
  args: Record<string, unknown>;
  status: "running" | "ok" | "error";
  error?: string | null;
}

export interface Stage {
  key: string;
  label: string;
  status: "start" | "done";
}

export interface Final {
  sources: string[];
  check: { total: number; verified: number; unverified: string[] };
  compliance: string[];
  disclaimer: string;
  usage: { prompt_tokens: number; completion_tokens: number };
}

export interface Msg {
  id: string;
  role: "user" | "ai";
  text: string;
  mode?: "chat" | "report";
  notes?: string[]; // planning text from tool rounds
  steps?: Step[];
  stages?: Stage[];
  debate?: { bull: string; bear: string };
  reasoning?: string;
  status?: string;
  final?: Final;
  error?: string;
  streaming?: boolean;
  charts?: ChartMsg[];
  t: number;
}

export interface ChartMsg {
  id: string;
  no: number;
  title: string;
  kind: string;
  option: Record<string, unknown>;
  check?: { total: number; verified: number; unverified: string[] } | null;
}

export interface Conversation {
  id: string;
  title: string;
  created: number;
  updated: number;
  msgs: Msg[];
}

export interface AssistantContext {
  page?: string;
  code?: string;
  name?: string;
  model?: string;
  // live 组合构建 settings, so answers can say when they differ from the precomputed portfolio
  portfolio?: { max_weight: number; industry_cap: number | null; universe_size: number; live: boolean };
}

interface Status {
  ready: boolean;
  enabled: boolean;
  has_key: boolean;
  model: string;
  model_pro: string;
  provider?: string;
  problem?: string | null;
}

interface AssistantState {
  convs: Conversation[]; // most recently updated first; the current one included
  current: string;
  msgs: Msg[]; // messages of the current conversation
  busy: boolean;
  status: Status | null;
  ctx: AssistantContext;
  setCtx: (c: Partial<AssistantContext>) => void;
  loadStatus: () => Promise<void>;
  send: (text: string) => Promise<void>;
  report: (code: string, name?: string) => Promise<void>;
  stop: () => void;
  newConv: () => void;
  switchConv: (id: string) => void;
  deleteConv: (id: string) => void;
  clear: () => void; // = start a new conversation
}

const KEY = "defaco.assistant.v2";
const OLD_KEY = "defaco.assistant.v1";
const MAX_CONVS = 40;
const welcome = (): Msg => ({
  id: "welcome",
  role: "ai",
  t: Date.now(),
  text:
    "你好，我是 DeFaCo 投研助手。我会先调用研究引擎和行情接口查询数据，再作答；回答里的每个数字都会自动核验出处。\n\n" +
    "- **问答**：市场、个股、组合配置原因、回测、因子选股、论文结论\n" +
    "- **我的股票**：分析自选股、候选池、持仓（风险、相关性、集中度）\n" +
    "- **画图**：走势对比、K 线、因子雷达、相关性热力图、行业分布、权重与风险贡献、回测净值\n" +
    "- **个股深度研报**：多头 / 空头分析师并行研判，风控主编整合\n\n" +
    "我不会给出目标价或买卖指令。左上角「历史」可以找回以前的对话。",
});
const uid = () => Math.random().toString(36).slice(2, 10);
const fresh = (): Conversation => ({ id: uid(), title: "新对话", created: Date.now(), updated: Date.now(), msgs: [welcome()] });

function loadConvs(): Conversation[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "null") as Conversation[] | null;
    if (v?.length) return v.map((c) => ({ ...c, msgs: c.msgs.map((m) => ({ ...m, streaming: false })) }));
    const old = JSON.parse(localStorage.getItem(OLD_KEY) || "null") as Msg[] | null; // migrate the single v1 thread
    if (old?.some((m) => m.role === "user")) {
      const first = old.find((m) => m.role === "user");
      return [{ id: uid(), title: titleOf(first?.text ?? "对话"), created: old[0].t, updated: old[old.length - 1].t, msgs: old }];
    }
  } catch {
    /* fall through */
  }
  return [fresh()];
}

function saveConvs(convs: Conversation[]) {
  const keep = convs.filter((c) => c.msgs.some((m) => m.role === "user")).slice(0, MAX_CONVS);
  try {
    localStorage.setItem(KEY, JSON.stringify(keep.map((c) => ({ ...c, msgs: c.msgs.slice(-60) }))));
  } catch {
    // quota: drop charts from older conversations, then retry once
    try {
      localStorage.setItem(KEY, JSON.stringify(keep.slice(0, 15).map((c, i) => ({ ...c, msgs: c.msgs.slice(-40).map((m) => (i > 2 ? { ...m, charts: undefined, reasoning: undefined } : m)) }))));
    } catch {
      /* storage unavailable: history lives for this tab only */
    }
  }
}

const titleOf = (q: string) => (q.length > 24 ? q.slice(0, 24) + "…" : q);

/** Everything the server needs to answer about "my" stocks. Holdings share counts go to the local
 * API only (it derives weights); the LLM never sees them. */
function requestContext(ctx: AssistantContext) {
  const lists = useLists.getState();
  const pf = usePortfolioSettings.getState();
  return {
    ...ctx,
    watch: lists.watch,
    pool: pf.pool,
    holdings: lists.holdings,
    portfolio_settings: { model: pf.model, max_weight: pf.cap / 100, industry_cap: pf.indCap / 100, use_pool: pf.usePool && pf.pool.length > 0 },
  };
}

let controller: AbortController | null = null;

export const useAssistant = create<AssistantState>((set, get) => {
  const initial = loadConvs();
  const patch = (id: string, f: (m: Msg) => Msg) => set((s) => ({ msgs: s.msgs.map((m) => (m.id === id ? f(m) : m)) }));

  /** Write the current messages back into the conversation list (title, order) and persist. */
  function commit() {
    const { convs, current, msgs } = get();
    const first = msgs.find((m) => m.role === "user");
    const now = Date.now();
    const conv: Conversation = {
      ...(convs.find((c) => c.id === current) ?? fresh()),
      id: current,
      msgs,
      updated: now,
      title: first ? titleOf(first.text) : "新对话",
    };
    const next = [conv, ...convs.filter((c) => c.id !== current)];
    set({ convs: next });
    saveConvs(next);
  }

  async function run(url: string, body: unknown, aiId: string) {
    controller?.abort();
    controller = new AbortController();
    set({ busy: true });
    commit();
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!r.ok || !r.body) throw new Error(`服务不可用（HTTP ${r.status}），请确认 API 已启动：scripts/dev.sh`);
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i: number;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, i).trim();
          buf = buf.slice(i + 2);
          if (line.startsWith("data:")) handle(aiId, JSON.parse(line.slice(5)));
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") patch(aiId, (m) => ({ ...m, error: (e as Error).message }));
      else patch(aiId, (m) => ({ ...m, status: "已停止" }));
    } finally {
      patch(aiId, (m) => ({ ...m, streaming: false, status: undefined }));
      set({ busy: false });
      commit();
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function handle(id: string, ev: any) {
    switch (ev.type) {
      case "status":
        return patch(id, (m) => ({ ...m, status: ev.text }));
      case "reasoning":
        return patch(id, (m) => ({ ...m, reasoning: (m.reasoning ?? "") + ev.text }));
      case "delta":
        return patch(id, (m) => ({ ...m, text: m.text + ev.text }));
      case "round_end": // the text so far was planning, not the answer
        return patch(id, (m) => ({ ...m, notes: m.text.trim() ? [...(m.notes ?? []), m.text.trim()] : m.notes, text: "" }));
      case "tool_start":
        return patch(id, (m) => ({
          ...m,
          steps: [...(m.steps ?? []), { id: ev.id, name: ev.name, label: ev.label, args: ev.args, status: "running" }],
        }));
      case "tool_end":
        return patch(id, (m) => ({
          ...m,
          steps: (m.steps ?? []).map((s) => (s.id === ev.id ? { ...s, status: ev.ok ? "ok" : "error", error: ev.error } : s)),
        }));
      case "chart":
        return patch(id, (m) => ({
          ...m,
          charts: [...(m.charts ?? []), { id: ev.id, no: ev.no, title: ev.title, kind: ev.kind, option: ev.option, check: ev.check }],
        }));
      case "stage":
        return patch(id, (m) => {
          const stages = [...(m.stages ?? [])];
          const k = stages.findIndex((s) => s.key === ev.key);
          const st = { key: ev.key, label: ev.label ?? stages[k]?.label ?? ev.key, status: ev.status };
          if (k >= 0) stages[k] = st;
          else stages.push(st);
          return { ...m, stages, debate: ev.bull ? { bull: ev.bull, bear: ev.bear } : m.debate };
        });
      case "final":
        return patch(id, (m) => ({ ...m, final: ev }));
      case "error":
        return patch(id, (m) => ({ ...m, error: ev.text }));
    }
  }

  return {
    convs: initial,
    current: initial[0].id,
    msgs: initial[0].msgs,
    busy: false,
    status: null,
    ctx: {},
    setCtx: (c) => set((s) => ({ ctx: { ...s.ctx, ...c } })),
    loadStatus: async () => {
      try {
        const r = await fetch("/api/assistant/status");
        set({ status: r.ok ? await r.json() : null });
      } catch {
        set({ status: null });
      }
    },
    send: async (text) => {
      const q = text.trim();
      if (!q || get().busy) return;
      const user: Msg = { id: uid(), role: "user", text: q, t: Date.now() };
      const ai: Msg = { id: uid(), role: "ai", text: "", mode: "chat", streaming: true, t: Date.now() };
      const history = [...get().msgs.filter((m) => m.id !== "welcome" && !m.error && m.text), user].map((m) => ({
        role: m.role === "ai" ? "assistant" : "user",
        content: m.text,
      }));
      set((s) => ({ msgs: [...s.msgs, user, ai] }));
      await run("/api/assistant/chat", { messages: history, context: requestContext(get().ctx) }, ai.id);
    },
    report: async (code, name) => {
      if (get().busy) return;
      const user: Msg = { id: uid(), role: "user", text: `生成 ${name ?? code} 的深度研报`, t: Date.now() };
      const ai: Msg = { id: uid(), role: "ai", text: "", mode: "report", streaming: true, t: Date.now() };
      set((s) => ({ msgs: [...s.msgs, user, ai] }));
      await run("/api/assistant/report", { code, context: requestContext(get().ctx) }, ai.id);
    },
    stop: () => controller?.abort(),
    newConv: () => {
      if (get().busy) return;
      if (!get().msgs.some((m) => m.role === "user")) return; // already on an empty conversation
      const c = fresh();
      set((s) => ({ convs: [c, ...s.convs], current: c.id, msgs: c.msgs }));
    },
    switchConv: (id) => {
      if (get().busy || id === get().current) return;
      const c = get().convs.find((x) => x.id === id);
      if (c) set({ current: id, msgs: c.msgs });
    },
    deleteConv: (id) => {
      if (get().busy) return;
      const rest = get().convs.filter((c) => c.id !== id);
      if (id === get().current) {
        const next = rest[0] ?? fresh();
        set({ convs: rest.length ? rest : [next], current: next.id, msgs: next.msgs });
      } else set({ convs: rest });
      saveConvs(get().convs);
    },
    clear: () => get().newConv(),
  };
});

/** Quick prompts that fit the page the user is on, plus ones about the user's own lists. */
export function suggestions(ctx: AssistantContext): { label: string; report?: boolean }[] {
  const name = ctx.name ?? ctx.code ?? "这只股票";
  const mine = [{ label: "分析一下我的自选股" }, { label: "画出自选股近一年的走势对比图" }];
  switch (ctx.page) {
    case "stock":
      return [{ label: `生成${name}深度研报`, report: true }, { label: `${name}的技术面和因子画像怎么样？` }, { label: `画出${name}和同行业龙头的走势对比` }, ...mine.slice(0, 1)];
    case "portfolio":
    case "risk":
      return [{ label: "当前组合最大的风险来源是什么？画出权重和风险贡献" }, { label: "为什么 PCA 因子模型相比 NLS 减配了食品饮料？" }, { label: "用候选池构建组合会怎样？" }];
    case "backtest":
      return [{ label: "画出各模型的回测净值，哪个最好？" }, { label: "PCA 组合的最大回撤发生在什么时候？" }];
    case "screener":
      return [{ label: "帮我找高 ROE、低波动、估值偏低的股票" }, { label: "分析一下我的候选池，画出相关性热力图" }];
    case "factors":
      return [{ label: "哪些因子在 A 股最有效？它们有什么共同点？" }, { label: "换手率因子 turn_1m 的效果稳定吗？画出它的分层收益" }];
    case "lab":
      return [{ label: "M1 实验的主要结论是什么？" }, { label: "为什么纯 IPCA 的换手会翻倍？" }, { label: "M0 复现和原论文有哪些不同？" }];
    case "rebalance":
      return [{ label: "我的持仓风险如何？集中度高吗？" }, { label: "画出我的持仓行业分布和风险贡献" }];
    default:
      return [{ label: "今天市场怎么样？现在是高波动还是低波动区制？" }, ...mine];
  }
}
