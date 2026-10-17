"""Chat client for any OpenAI-compatible ``/chat/completions`` API, with streaming tool calls.

Supported out of the box (``LLM_PROVIDER``): DeepSeek (default), OpenAI, Qwen (DashScope), Kimi (Moonshot),
GLM (Zhipu), SiliconFlow, OpenRouter, a local Ollama server, or ``custom`` with your own ``LLM_BASE_URL``.
Any other service that speaks the OpenAI chat-completions protocol and supports tool (function) calling
works as ``custom``.

Configuration comes from the repo's ``.env`` (git-ignored); nothing secret lives in code:

    LLM_PROVIDER=deepseek        # see PROVIDERS below
    LLM_API_KEY=...              # not needed for a local Ollama server
    LLM_BASE_URL=                # optional: overrides the provider's default endpoint (required for custom)
    LLM_MODEL=                   # model for Q&A and analysts (default: the provider's default, if any)
    LLM_MODEL_PRO=               # model for the report editor (default: LLM_MODEL)

The older ``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_MODEL`` / ``DEEPSEEK_MODEL_PRO`` variables are
still read as fallbacks. DeepSeek-specific request fields (reasoning effort, switching thinking off, echoing
the reasoning inside tool-use turns) are only sent to DeepSeek, so other providers receive standard requests.
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


@dataclass(frozen=True)
class Provider:
    label: str
    base_url: str
    model: str = ""            # default model ("" = must be set with LLM_MODEL)
    model_pro: str = ""
    needs_key: bool = True
    deepseek_params: bool = False   # reasoning_effort / thinking / reasoning_content round trip
    usage_in_stream: bool = True    # stream_options.include_usage


# Default endpoints of OpenAI-compatible APIs (check your provider's documentation if a request fails).
PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider("DeepSeek", "https://api.deepseek.com", "deepseek-flash", "deepseek-v4-pro", deepseek_params=True),
    "openai": Provider("OpenAI", "https://api.openai.com/v1"),
    "qwen": Provider("通义千问 Qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "moonshot": Provider("Kimi (Moonshot)", "https://api.moonshot.cn/v1", usage_in_stream=False),
    "zhipu": Provider("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4", usage_in_stream=False),
    "siliconflow": Provider("SiliconFlow", "https://api.siliconflow.cn/v1"),
    "openrouter": Provider("OpenRouter", "https://openrouter.ai/api/v1"),
    "ollama": Provider("Ollama（本地）", "http://127.0.0.1:11434/v1", needs_key=False, usage_in_stream=False),
    "custom": Provider("自定义", "", usage_in_stream=False),
}


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return default


def _provider_name() -> str:
    name = _env("LLM_PROVIDER").lower()
    if name:
        return name if name in PROVIDERS else "custom"
    return "deepseek"


@dataclass
class LLMConfig:
    provider: str = field(default_factory=_provider_name)
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "DEEPSEEK_API_KEY"))
    base_url: str = ""
    model: str = ""
    model_pro: str = ""
    enabled: bool = field(default_factory=lambda: os.environ.get("ASSISTANT_ENABLED", "1") != "0")

    def __post_init__(self):
        p = self.spec
        legacy = self.provider == "deepseek"      # DEEPSEEK_* variables only describe DeepSeek
        self.base_url = (self.base_url or _env("LLM_BASE_URL", *(["DEEPSEEK_BASE_URL"] if legacy else []), default=p.base_url)).rstrip("/")
        self.model = self.model or _env("LLM_MODEL", *(["DEEPSEEK_MODEL"] if legacy else []), default=p.model)
        self.model_pro = self.model_pro or _env("LLM_MODEL_PRO", *(["DEEPSEEK_MODEL_PRO"] if legacy else []), default=p.model_pro or self.model)

    @property
    def spec(self) -> Provider:
        return PROVIDERS.get(self.provider, PROVIDERS["custom"])

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def problem(self) -> str | None:
        """Why the assistant cannot run (None when it can)."""
        if not self.enabled:
            return "AI 助手已关闭（.env 中 ASSISTANT_ENABLED=0）"
        if self.spec.needs_key and not self.api_key:
            return f"未配置 {self.label} 的 API Key（.env 中的 LLM_API_KEY）"
        if not self.base_url:
            return "未配置接口地址（自定义服务需要在 .env 中设置 LLM_BASE_URL）"
        if not self.model:
            return f"未配置模型名（.env 中的 LLM_MODEL；{self.label} 没有默认模型）"
        return None

    @property
    def ready(self) -> bool:
        return self.problem is None


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
    echo_reasoning: bool = False      # DeepSeek's thinking models expect their reasoning back in a tool-use turn

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
            if self.reasoning and self.echo_reasoning:
                msg["reasoning_content"] = self.reasoning
        return msg


class ChatClient:
    def __init__(self, cfg: LLMConfig | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg or LLMConfig()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0), transport=transport)

    async def aclose(self):
        await self.http.aclose()

    def build_body(self, messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
                   temperature: float = 0.3, max_tokens: int = 6000, effort: str | None = "low") -> dict[str, Any]:
        """Request body. ``effort`` ("off" / "low" / "high" / "max") is a DeepSeek extension and is ignored
        for other providers, which use their model's default behaviour."""
        p = self.cfg.spec
        body: dict[str, Any] = {"model": model or self.cfg.model, "messages": messages, "stream": True,
                                "temperature": temperature, "max_tokens": max_tokens}
        if p.usage_in_stream:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools
        if p.deepseek_params:
            if effort == "off":
                body["thinking"] = {"type": "disabled"}
            elif effort:
                body["reasoning_effort"] = effort
        return body

    async def stream(self, messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
                     temperature: float = 0.3, max_tokens: int = 6000,
                     effort: str | None = "low") -> AsyncIterator[tuple[str, Any]]:
        """Yields ("content", text) / ("reasoning", text) deltas, then ("turn", Turn)."""
        if self.cfg.problem:
            raise LLMError(self.cfg.problem)
        body = self.build_body(messages, tools, model, temperature, max_tokens, effort)
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        turn = Turn(echo_reasoning=self.cfg.spec.deepseek_params)
        calls: dict[int, dict[str, Any]] = {}
        async with self.http.stream("POST", f"{self.cfg.base_url}/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                detail = (await r.aread()).decode("utf-8", "replace")[:300]
                raise LLMError(f"{self.cfg.label} API {r.status_code}: {detail}")
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
                    d = ch.get("delta") or {}
                    reasoning = d.get("reasoning_content") or d.get("reasoning")   # DeepSeek / Qwen, OpenRouter
                    if reasoning:
                        turn.reasoning += reasoning
                        yield "reasoning", reasoning
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


DeepSeek = ChatClient   # backwards-compatible name
