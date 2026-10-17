"""Post-answer guardrails.

1. Grounding: every number in the answer should be traceable to a tool result (or the user's
   own message), allowing for rounding and the usual unit changes (fraction <-> %, 元 -> 万/亿).
   Numbers that cannot be matched are reported back to the UI as "unverified" (they may be
   legitimately derived, e.g. a difference of two engine numbers, so this is a flag, not a block).
2. Compliance: the assistant must not issue trading instructions or price targets.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

_NUM = re.compile(r"(?<![\w.])[-+−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])[-+−]?\d+(?:\.\d+)?%?")
_STRIP = [
    re.compile(r"(?:19|20)\d{2}(?:[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?|\.\d{1,2}\.\d{1,2})"),  # dates (not prices like 1315.87)
    re.compile(r"\d{1,2}:\d{2}(?::\d{2})?"),                          # times
    re.compile(r"\b\d{6}(?:\.(?:SH|SZ|BJ))?\b", re.I),               # security codes
    re.compile(r"\b[Mm]\d\b|\bH\d\b|\bC\d\b"),                          # M0..M4, H1.., C1..
    re.compile(r"(?:MA|EMA|RSI|ma|rsi)\s?\d+"),                        # indicator windows
    re.compile(r"\d+\s?(?:日|天|个月|月|年|周|只|个|家|项|条|步|bp|bps)(?![\d.])"),  # counts and horizons
]
_FORBIDDEN = [
    (re.compile(r"目标价|目标位"), "给出了目标价"),
    (re.compile(r"(?:建议|应当|应该|不妨)[^。\n]{0,15}?(?:买入|卖出|加仓|减仓|清仓|抄底|建仓)"), "包含交易指令"),
    (re.compile(r"\d+(?:\.\d+)?\s*元?(?:附近|以下|以上|左右)?(?:买入|卖出|加仓|减仓|止损|止盈)"), "包含带价位的交易指令"),
    (re.compile(r"必涨|稳赚|保本|翻倍股|无风险套利"), "含有收益承诺类措辞"),
]


def _collect(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect(v, out)
            for m in _NUM.findall(str(k)):
                out.append(_to_float(m)[0])
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            out.append(_to_float(m)[0])


def _to_float(tok: str) -> tuple[float, int, bool]:
    pct = tok.endswith("%")
    t = tok.rstrip("%").replace(",", "").replace("−", "-")
    dec = len(t.split(".")[1]) if "." in t else 0
    return float(t), dec, pct


def check_numbers(answer: str, evidence: Iterable[Any]) -> dict[str, Any]:
    text = answer
    for pat in _STRIP:
        text = pat.sub(" ", text)
    ev: list[float] = []
    for e in evidence:
        _collect(e, ev)
    cands = set()
    for v in ev:
        for s in (1, 100, 1e-4, 1e-8, 1e4, 0.01):
            cands.add(v * s)
        cands.add(abs(v) * 100)
        cands.add(abs(v))
    cands = sorted(cands)
    total, unverified = 0, []
    for tok in _NUM.findall(text):
        val, dec, pct = _to_float(tok)
        if not pct and dec == 0 and abs(val) <= 31:      # small integers: counts, ranks, list numbers
            continue
        total += 1
        tol = 0.5 * 10 ** (-dec) + 1e-9 + (0.005 * abs(val) if abs(val) >= 100 else 0)
        if not any(abs(abs(val) - abs(c)) <= tol for c in cands if abs(abs(c) - abs(val)) <= tol + 1):
            unverified.append(tok)
    return {"total": total, "verified": total - len(unverified), "unverified": unverified[:12]}


def check_compliance(answer: str) -> list[str]:
    return [why for pat, why in _FORBIDDEN if pat.search(answer)]


DISCLAIMER = "以上内容由模型基于研究引擎与公开行情数据生成，仅供研究与教学，不构成投资建议。"
