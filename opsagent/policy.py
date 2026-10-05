"""Risk policy: which tool calls need a human.

Tiers come from the MCP server's own annotations (``readOnlyHint`` / ``destructiveHint``),
defaulting the way the MCP spec does for an unannotated tool: destructive.  The agent adds
context the server cannot know: arguments can escalate a call (restarting a stateful
service is treated as destructive) and operators can hard-block tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

READ, MUTATE, DESTRUCTIVE, BLOCKED = "read", "mutate", "destructive", "blocked"
STATEFUL = {"redis", "postgres"}


@dataclass
class Policy:
    blocked_tools: set[str] = field(default_factory=lambda: set(filter(None, os.environ.get("OPSAGENT_BLOCK", "").split(","))))
    stateful_services: set[str] = field(default_factory=lambda: set(STATEFUL))

    def classify(self, tool: str, annotations: Any, args: dict[str, Any]) -> str:
        if tool in self.blocked_tools:
            return BLOCKED
        read_only = getattr(annotations, "read_only_hint", None)
        destructive = getattr(annotations, "destructive_hint", None)
        if read_only:
            return READ
        tier = MUTATE if destructive is False else DESTRUCTIVE  # spec default: unannotated == destructive
        if tier == MUTATE and args.get("service") in self.stateful_services:
            return DESTRUCTIVE
        return tier

    @staticmethod
    def needs_approval(tier: str) -> bool:
        return tier != READ
