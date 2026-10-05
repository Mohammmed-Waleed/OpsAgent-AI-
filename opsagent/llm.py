"""Thin wrapper over the Anthropic Messages API returning plain dicts.

Plain dicts (not SDK objects) are what the store persists and what tests fake.  Thinking blocks
carry a signature and must be replayed unchanged, so blocks are normalised without altering
their payloads.  History is only ever appended to - that keeps replayed thinking valid and the
prompt cache warm across turns and across resumes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    request_id: str | None = None
    stop_detail: str | None = None


class LLM(Protocol):
    model: str

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


def _block(b: Any) -> dict[str, Any]:
    d = b.model_dump(exclude_none=True)
    t = d.get("type")
    if t == "text":
        return {"type": "text", "text": d["text"]}
    if t == "tool_use":
        return {"type": "tool_use", "id": d["id"], "name": d["name"], "input": d["input"]}
    if t == "thinking":
        return {"type": "thinking", "thinking": d.get("thinking", ""), "signature": d["signature"]}
    return d


class AnthropicLLM:
    def __init__(self, model: str | None = None, effort: str | None = None, max_tokens: int = 16000):
        self.model = model or os.environ.get("OPSAGENT_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("OPSAGENT_EFFORT", "medium")
        self.max_tokens = max_tokens
        self.client = anthropic.AsyncAnthropic()  # SDK retries 429/5xx/connection errors itself

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # cache the growing prefix turn over turn
        )
        u = resp.usage
        detail = getattr(resp, "stop_details", None)
        return LLMResponse(
            content=[_block(b) for b in resp.content],
            stop_reason=resp.stop_reason or "end_turn",
            usage={"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                   "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                   "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0},
            model=resp.model,
            request_id=getattr(resp, "_request_id", None),
            stop_detail=getattr(detail, "category", None) if detail else None,
        )
