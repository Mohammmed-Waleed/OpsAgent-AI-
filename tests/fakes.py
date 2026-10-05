from __future__ import annotations

from typing import Any, Callable

from opsagent.llm import LLMResponse

_n = 0


def tool_use(name: str, **inp: Any) -> dict:
    global _n
    _n += 1
    return {"type": "tool_use", "id": f"toolu_{_n:04d}", "name": name, "input": inp}


def say(text: str) -> dict:
    return {"type": "text", "text": text}


class ScriptedLLM:
    """Replays a fixed script of assistant turns and records what it was shown."""

    model = "fake-model"

    def __init__(self, *turns: list[dict] | Callable[[list[dict]], list[dict]]):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    async def complete(self, system, messages, tools) -> LLMResponse:
        self.seen.append(messages)
        turn = self.turns.pop(0)
        content = turn(messages) if callable(turn) else turn
        has_tool = any(b["type"] == "tool_use" for b in content)
        return LLMResponse(content=content, stop_reason="tool_use" if has_tool else "end_turn",
                           usage={"input_tokens": 100, "output_tokens": 20}, model=self.model)

    @property
    def calls(self) -> int:
        return len(self.seen)

    def results_at(self, call_index: int) -> list[dict]:
        """tool_result blocks in the last user message shown to the model on its Nth call."""
        return [b for b in self.seen[call_index][-1]["content"] if b["type"] == "tool_result"]
