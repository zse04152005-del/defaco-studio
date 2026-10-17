"""LLM client: provider presets, backwards-compatible DeepSeek settings, standard requests for other
providers, and parsing of streamed tool calls (mock transport, no network)."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.assistant.llm import ChatClient, LLMConfig  # noqa: E402

ENV = ("LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_MODEL_PRO",
       "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_MODEL_PRO", "ASSISTANT_ENABLED")


@pytest.fixture
def env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_legacy_deepseek_settings_still_work(env):
    env.setenv("DEEPSEEK_API_KEY", "k")
    c = LLMConfig()
    assert c.provider == "deepseek" and c.ready and c.base_url == "https://api.deepseek.com" and c.model


def test_other_provider_needs_model_and_gets_standard_body(env):
    env.setenv("LLM_PROVIDER", "openai")
    env.setenv("LLM_API_KEY", "k")
    env.setenv("DEEPSEEK_MODEL", "deepseek-flash")          # DeepSeek variables must not leak into other providers
    c = LLMConfig()
    assert not c.ready and "LLM_MODEL" in c.problem
    env.setenv("LLM_MODEL", "some-model")
    c = LLMConfig()
    assert c.ready and c.base_url == "https://api.openai.com/v1" and c.model_pro == "some-model"
    body = ChatClient(c).build_body([{"role": "user", "content": "hi"}], effort="off")
    assert "thinking" not in body and "reasoning_effort" not in body and body["model"] == "some-model"


def test_deepseek_extensions_only_for_deepseek(env):
    env.setenv("LLM_API_KEY", "k")
    body = ChatClient(LLMConfig()).build_body([], effort="off")
    assert body["thinking"] == {"type": "disabled"}
    assert ChatClient(LLMConfig()).build_body([], effort="low")["reasoning_effort"] == "low"


def test_local_ollama_needs_no_key_and_custom_needs_url(env):
    env.setenv("LLM_PROVIDER", "ollama")
    env.setenv("LLM_MODEL", "qwen3")
    assert LLMConfig().ready
    env.setenv("LLM_PROVIDER", "my-gateway")                   # unknown names are treated as custom
    env.setenv("LLM_API_KEY", "k")
    c = LLMConfig()
    assert c.provider == "custom" and not c.ready and "LLM_BASE_URL" in c.problem


def test_streamed_tool_calls_are_parsed(env):
    env.setenv("LLM_PROVIDER", "custom")
    env.setenv("LLM_BASE_URL", "https://llm.example/v1")
    env.setenv("LLM_API_KEY", "k")
    env.setenv("LLM_MODEL", "m")
    chunks = [{"choices": [{"delta": {"content": "看"}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_", "arguments": "{\"co"}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "portfolio", "arguments": "de\": 1}"}}]}, "finish_reason": "tool_calls"}]}]
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers.get("authorization"), json.loads(request.content)
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    async def run():
        client = ChatClient(transport=httpx.MockTransport(handler))
        out = [x async for x in client.stream([{"role": "user", "content": "q"}], tools=[{"type": "function"}])]
        await client.aclose()
        return out

    out = asyncio.run(run())
    turn = out[-1][1]
    assert seen["url"] == "https://llm.example/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert "stream_options" not in seen["body"] and "thinking" not in seen["body"]
    assert turn.content == "看" and turn.finish_reason == "tool_calls"
    assert turn.tool_calls[0]["function"] == {"name": "get_portfolio", "arguments": "{\"code\": 1}"}
    assert "reasoning_content" not in turn.as_message()
