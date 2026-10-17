"""DeepSeek chat client (OpenAI-compatible /chat/completions) with streaming tool calls.

The key is read from the repo's .env (git-ignored); nothing secret lives in code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[4] / ".env")


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"))
    model: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    model_pro: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL_PRO", "deepseek-v4-pro"))
    enabled: bool = field(default_factory=lambda: os.environ.get("ASSISTANT_ENABLED", "1") != "0")

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.api_key)


class LLMError(RuntimeError):
    pass


@dataclass
class Turn:
    """One assistant turn: text, optional reasoning and the tool calls it requested."""
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
            if self.reasoning:   # thinking models expect their reasoning back inside a tool-use turn
                msg["reasoning_content"] = self.reasoning
        return msg


class DeepSeek:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))

    async def aclose(self):
        await self.http.aclose()

    async def stream(self, messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
                     temperature: float = 0.3, max_tokens: int = 6000,
                     effort: str | None = "low") -> AsyncIterator[tuple[str, Any]]:
        """Yields ("content", text) / ("reasoning", text) deltas, then ("turn", Turn).

        effort: "off" disables thinking, "low"/"high"/"max" set the reasoning effort (thinking
        tokens count towards max_tokens), None keeps the model default."""
        if not self.cfg.ready:
            raise LLMError("未配置 DeepSeek API Key（.env 中的 DEEPSEEK_API_KEY）或已在设置中关闭")
        body: dict[str, Any] = {"model": model or self.cfg.model, "messages": messages, "stream": True,
                                "temperature": temperature, "max_tokens": max_tokens,
                                "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        if effort == "off":
            body["thinking"] = {"type": "disabled"}
        elif effort:
            body["reasoning_effort"] = effort
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        turn = Turn()
        calls: dict[int, dict[str, Any]] = {}
        async with self.http.stream("POST", f"{self.cfg.base_url}/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                detail = (await r.aread()).decode("utf-8", "replace")[:300]
                raise LLMError(f"DeepSeek API {r.status_code}: {detail}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    turn.usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    turn.finish_reason = ch.get("finish_reason") or turn.finish_reason
                    d = ch.get("delta", {})
                    if d.get("reasoning_content"):
                        turn.reasoning += d["reasoning_content"]
                        yield "reasoning", d["reasoning_content"]
                    if d.get("content"):
                        turn.content += d["content"]
                        yield "content", d["content"]
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                                   "function": {"name": "", "arguments": ""}})
                        c["id"] = tc.get("id") or c["id"]
                        fn = tc.get("function") or {}
                        c["function"]["name"] += fn.get("name") or ""
                        c["function"]["arguments"] += fn.get("arguments") or ""
        turn.tool_calls = [calls[i] for i in sorted(calls)]
        yield "turn", turn

    async def complete(self, messages: list[dict], model: str | None = None, temperature: float = 0.3,
                       max_tokens: int = 2048, effort: str | None = "off") -> Turn:
        turn = Turn()
        async for kind, val in self.stream(messages, None, model, temperature, max_tokens, effort):
            if kind == "turn":
                turn = val
        return turn
