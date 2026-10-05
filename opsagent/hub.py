"""MCP client side: spawn the ops MCP server over stdio and expose its tools to Claude."""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.types import TextContent

MAX_RESULT_CHARS = 8000


class ToolHub:
    def __init__(self, command: str | None = None, args: list[str] | None = None, env: dict[str, str] | None = None):
        self.params = StdioServerParameters(command=command or sys.executable,
                                            args=args if args is not None else ["-m", "opsagent.mcp_server"],
                                            env={**os.environ, **(env or {})})
        self._stack = AsyncExitStack()
        self.client: Client | None = None
        self.annotations: dict[str, Any] = {}
        self.tools: list[dict[str, Any]] = []
        self.runbook = ""

    async def __aenter__(self) -> "ToolHub":
        self.client = await self._stack.enter_async_context(Client(self.params))
        listed = await self.client.list_tools()
        # Sorted: a deterministic tool list keeps the prompt-cache prefix stable across turns and resumes.
        for t in sorted(listed.tools, key=lambda t: t.name):
            self.annotations[t.name] = t.annotations
            self.tools.append({"name": t.name, "description": t.description or "", "input_schema": t.input_schema})
        try:
            res = await self.client.read_resource("ops://runbook")
            self.runbook = "".join(getattr(c, "text", "") for c in res.contents)
        except Exception:
            self.runbook = ""  # the runbook is a nicety, not a dependency
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        assert self.client is not None
        res = await self.client.call_tool(name, args)
        text = "\n".join(c.text for c in res.content if isinstance(c, TextContent)) or "(no output)"
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n…[truncated {len(text) - MAX_RESULT_CHARS} chars]"
        return text, bool(res.is_error)
