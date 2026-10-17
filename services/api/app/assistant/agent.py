"""Assistant workflows.

chat   — a single tool-using agent (plan -> call engine tools, in parallel, up to MAX_ROUNDS ->
         answer with sources), like OpenBB Copilot / FinRobot's data-grounded agents. Fast, and every
         number can be traced to a tool result.
report — a TradingAgents-style pipeline for one stock, kept small: deterministic data gathering
         (no LLM), bull and bear analysts in parallel, then a risk-control editor that writes the
         final report. Roughly 3 LLM calls instead of TradingAgents' 10+.

Both stream events (dicts) that the API sends to the browser as Server-Sent Events, and both end
with the same guardrails (number grounding, compliance) from guard.py.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any, AsyncIterator

from ..realtime import session_status
from .guard import DISCLAIMER, check_compliance, check_numbers
from .llm import ChatClient, LLMError
from .tools import SOURCE_LABEL, Toolbox, call_tool, openai_tools

MAX_ROUNDS = 5
MAX_TOOL_CHARS = 6000
PAGE_CN = {"overview": "市场总览", "stock": "个股 K 线", "screener": "选股器", "portfolio": "组合构建", "risk": "风险解析",
           "rebalance": "持仓调仓", "backtest": "回测绩效", "alerts": "监控预警", "lab": "论文实验室", "factors": "因子研究"}

RULES = """你是 DeFaCo Studio 的 A 股投研助手，服务对象是研究者和学生。你的工作方式：
1. **数字只能来自工具结果或用户输入**。先判断需要哪些数据，再调用工具（可在同一轮并行调用多个）；工具没有的数据就明确说“暂无数据”，绝不编造。
2. 区分两类数据并在回答中写明：研究数据（风险模型、组合、回测、因子，截至 {asof}）与实时行情（今日，免费接口可能延迟）。
3. 单位换算要正确：工具里的收益、权重、波动、pct 多为小数（0.0831 = 8.31%）；成交额单位为元。
4. 回答结构：第一句给出直接结论；然后分点给依据，关键数字后用【来源】标注，来源名取自：{sources}；必要时用 Markdown 表格；最后一行列出“数据来源”。
5. 可以解释模型为什么这样配置、风险来自哪里、指标意味着什么；**不得**给出目标价、具体买卖价位或仓位指令，不承诺收益。用户索要买卖建议时，改为给出需要关注的风险与数据。
6. 用户说“这只股票/当前组合/这个模型”时，优先使用页面上下文。涉及本项目论文或方法（DeFaCo、M0、M1、IPCA 等）时调用 get_research_findings。
7. 用户问到“我的自选 / 候选池 / 持仓 / 我的组合”时，用 analyze_stocks（source=watchlist/pool/holdings）、analyze_holdings、get_live_portfolio，不要只看单只股票。
8. **画图**：用户要求画图，或比较多只股票的走势、相关性、行业分布、权重时，调用 make_chart（一次回答最多 3 张）。图会直接显示给用户，回答里写“见图 1”并点出图中最重要的 1–2 个结论；不要用文字重复画出整张图。自定义 bar/line 的数值必须来自工具结果。
9. 用简洁的中文，不要复述工具的原始 JSON，不要输出与问题无关的长篇背景。"""


def _context_text(ctx: dict | None) -> str:
    if not ctx:
        return "（无页面上下文）"
    parts = [f"用户当前页面：{PAGE_CN.get(ctx.get('page', ''), ctx.get('page', '未知'))}"]
    if ctx.get("code"):
        parts.append(f"正在查看的股票：{ctx['code']}{('（' + ctx['name'] + '）') if ctx.get('name') else ''}")
    if ctx.get("model"):
        parts.append(f"选中的风险模型：{ctx['model']}")
    if isinstance(ctx.get("portfolio"), dict):
        pf = ctx["portfolio"]
        cap = f"{pf['industry_cap']:.0%}" if pf.get("industry_cap") else "无"
        parts.append(f"页面上的组合参数：单股上限 {pf.get('max_weight', 0.1):.0%}，行业上限 {cap}，股票池 {pf.get('universe_size', '?')} 只，"
                     f"{'实时求解' if pf.get('live') else '离线预计算'}（组合工具返回的是默认参数：单股上限 10%、无行业上限、沪深300 前100）")
    # the user's lists: names / codes only; holdings share counts stay on the local server
    for key, label in (("watch", "自选股"), ("pool", "候选池"), ("holdings", "持仓股票")):
        items = ctx.get(key) or []
        if items:
            codes = [h["code"] if isinstance(h, dict) else h for h in items]
            parts.append(f"{label} {len(codes)} 只：" + "、".join(_label(c) for c in codes[:20]) + ("…" if len(codes) > 20 else ""))
    return "；".join(parts)


def _label(code: str) -> str:
    from .analysis import name_of
    n = name_of(code)
    return f"{n}({code})" if n != code else code


def system_prompt(ctx: dict | None) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rules = RULES.format(asof="2025-12-31", sources="、".join(f"【{v}】" for v in SOURCE_LABEL.values()))
    return f"{rules}\n\n当前时间：{now}（北京时间），市场状态：{session_status()['label']}。\n{_context_text(ctx)}"


def _clip(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= MAX_TOOL_CHARS else s[:MAX_TOOL_CHARS] + '…（已截断）"'


def _finish(answer: str, evidence: list[Any], used: list[str], usage: dict) -> dict:
    return {"type": "final", "sources": sorted(set(used), key=used.index),
            "check": check_numbers(answer, evidence), "compliance": check_compliance(answer),
            "disclaimer": DISCLAIMER, "usage": usage}


async def run_chat(llm: ChatClient, box: Toolbox, history: list[dict], ctx: dict | None) -> AsyncIterator[dict]:
    msgs: list[dict] = [{"role": "system", "content": system_prompt(ctx)}]
    msgs += [{"role": m["role"], "content": str(m["content"])[:4000]} for m in history[-12:]
             if m.get("role") in ("user", "assistant") and m.get("content")]
    evidence: list[Any] = [history[-1]["content"]] if history else []
    used: list[str] = []
    n_charts = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    tools = openai_tools()
    for rnd in range(MAX_ROUNDS + 1):
        yield {"type": "status", "text": "分析问题、规划查询…" if rnd == 0 else "整合数据、撰写回答…"}
        turn = None
        async for kind, val in llm.stream(msgs, tools if rnd < MAX_ROUNDS else None):
            if kind == "content":
                yield {"type": "delta", "text": val}
            elif kind == "reasoning":
                yield {"type": "reasoning", "text": val}
            else:
                turn = val
        for k in usage:
            usage[k] += (turn.usage or {}).get(k, 0)
        if not turn.tool_calls:
            if turn.finish_reason == "length":
                yield {"type": "delta", "text": "\n\n…（回答达到长度上限被截断，可以让我“继续”）"}
            yield _finish(turn.content, evidence, used, usage)
            return
        # this round was planning: the text streamed so far is a step note, not the answer
        yield {"type": "round_end", "text": turn.content}
        msgs.append(turn.as_message())
        calls = turn.tool_calls
        for c in calls:
            try:
                c["_args"] = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                c["_args"] = {}
            name = c["function"]["name"]
            yield {"type": "tool_start", "id": c["id"], "name": name, "label": SOURCE_LABEL.get(name, name), "args": c["_args"]}
        results = await asyncio.gather(*(call_tool(box, c["function"]["name"], c["_args"]) for c in calls))
        for c, res in zip(calls, results):
            name = c["function"]["name"]
            ok = "error" not in res
            option = res.pop("_chart", None)
            if option is not None:
                n_charts += 1
                res["chart_no"] = n_charts
                chk = None
                if res.get("kind") in ("bar", "line"):   # values typed by the model: check them like answer numbers
                    vals = " ".join(str(v) for s in option.get("series", []) for v in s.get("data", []) if v is not None)
                    chk = check_numbers(vals, evidence)
                yield {"type": "chart", "id": c["id"], "no": n_charts, "title": res.get("title", ""), "kind": res.get("kind"),
                       "option": option, "check": chk}
            if ok and not (name == "make_chart" and res.get("kind") in ("bar", "line")):
                evidence.append(res)   # built-in chart summaries are engine data; custom chart values are not
            if ok:
                used.append(SOURCE_LABEL.get(name, name))
            yield {"type": "tool_end", "id": c["id"], "name": name, "ok": ok, "error": res.get("error")}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": _clip(res)})


# ----------------------------------------------------------------------------- stock report
BULL = "你是多头分析师。仅依据给定数据，列出 3–5 条支持该股票的论据（技术面、因子画像、行业、在低风险组合中的角色等），每条带上所依据的数字。不要编造数据，不要给价格目标。"
BEAR = "你是空头分析师。仅依据给定数据，列出 3–5 条风险与不利因素（估值、波动、回撤、动量、流动性、集中度等），每条带上所依据的数字。不要编造数据，不要给价格目标。"
EDITOR = """你是风控主编。把数据与多空双方观点整合成一份简明的中文个股研究简报（Markdown），结构固定：
## 一句话结论（中性措辞：偏积极 / 中性 / 偏谨慎，并说明主要理由）
## 行情与技术面（区分实时行情与日线指标）
## 因子画像（解释分位数含义，突出最极端的 3–4 个因子）
## 多空论据（两栏表格，各 3 条）
## 风险提示
## 在 DeFaCo 低风险组合中的角色（引用各模型组合中的权重与风险贡献；不在组合中则说明原因）
规则：所有数字必须来自给定数据；关键数字后标注【来源】；不得给目标价、买卖价位或仓位指令。"""


async def run_report(llm: ChatClient, box: Toolbox, code: str, ctx: dict | None) -> AsyncIterator[dict]:
    yield {"type": "stage", "key": "data", "status": "start", "label": "收集数据"}
    plan = [("get_stock_analysis", {"code": code}), ("get_realtime_quotes", {"codes": [code]}), ("get_market_overview", {})]
    for i, (n, a) in enumerate(plan):
        yield {"type": "tool_start", "id": f"r{i}", "name": n, "label": SOURCE_LABEL[n], "args": a}
    results = await asyncio.gather(*(call_tool(box, n, a) for n, a in plan))
    data = {}
    for i, ((n, _), res) in enumerate(zip(plan, results)):
        yield {"type": "tool_end", "id": f"r{i}", "name": n, "ok": "error" not in res, "error": res.get("error")}
        data[SOURCE_LABEL[n]] = res
    if "error" in results[0]:
        yield {"type": "error", "text": results[0]["error"]}
        return
    yield {"type": "stage", "key": "data", "status": "done"}
    blob = _clip(data)
    name = results[0].get("name", code)
    yield {"type": "stage", "key": "debate", "status": "start", "label": "多空分析师并行研判"}
    bull, bear = await asyncio.gather(
        llm.complete([{"role": "system", "content": BULL}, {"role": "user", "content": f"股票：{name}\n数据：{blob}"}], max_tokens=1200),
        llm.complete([{"role": "system", "content": BEAR}, {"role": "user", "content": f"股票：{name}\n数据：{blob}"}], max_tokens=1200))
    yield {"type": "stage", "key": "debate", "status": "done", "bull": bull.content, "bear": bear.content}
    yield {"type": "stage", "key": "editor", "status": "start", "label": "风控主编整合"}
    msgs = [{"role": "system", "content": EDITOR + "\n\n" + system_prompt(ctx)},
            {"role": "user", "content": f"股票：{name}\n数据：{blob}\n\n多头观点：\n{bull.content}\n\n空头观点：\n{bear.content}"}]
    final = None
    async for kind, val in llm.stream(msgs, None, model=llm.cfg.model_pro, max_tokens=8000, effort="low"):
        if kind == "content":
            yield {"type": "delta", "text": val}
        elif kind == "turn":
            final = val
    usage = {k: sum((t.usage or {}).get(k, 0) for t in (bull, bear, final)) for k in ("prompt_tokens", "completion_tokens")}
    yield {"type": "stage", "key": "editor", "status": "done"}
    yield _finish(final.content, list(data.values()), list(data), usage)


async def guarded(gen: AsyncIterator[dict]) -> AsyncIterator[dict]:
    try:
        async for ev in gen:
            yield ev
    except LLMError as exc:
        yield {"type": "error", "text": str(exc)}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "text": f"助手出错：{type(exc).__name__}: {exc}"}
