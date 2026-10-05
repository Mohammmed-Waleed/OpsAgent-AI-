"""Human-in-the-loop approval for risky tool calls.

Three ways to get a decision:
  ask    interactive prompt in this terminal (the run waits)
  queue  park the run durably; decide later with `opsagent approve`, then `opsagent resume`
  auto / deny  non-interactive policies (auto is for sandboxes and tests only)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

MODES = ("ask", "queue", "auto", "deny")


@dataclass
class ApprovalRequest:
    run_id: str
    call_id: str
    tool: str
    args: dict
    tier: str
    reason: str | None


@dataclass
class Decision:
    approve: bool
    by: str
    note: str = ""


class Approver(Protocol):
    async def ask(self, req: ApprovalRequest) -> Decision: ...


class CliApprover:
    """Prompts on the terminal.  Anything other than y/yes is a denial; the text is fed back to the model."""

    async def ask(self, req: ApprovalRequest) -> Decision:
        banner = "\033[1;31mDESTRUCTIVE\033[0m" if req.tier == "destructive" else "\033[1;33mCHANGE\033[0m"
        print(f"\n  ┌─ approval needed [{banner}]")
        print(f"  │ tool:   {req.tool}")
        print(f"  │ args:   {json.dumps(req.args)}")
        if req.reason:
            print(f"  │ why:    {req.reason.strip()[:300]}")
        print("  └─ approve? [y = yes / anything else = deny, text becomes the reason]")
        try:
            answer = (await asyncio.to_thread(input, "  > ")).strip()
        except EOFError:
            return Decision(False, "policy:no-terminal", "no interactive terminal; use --approve queue")
        if answer.lower() in ("y", "yes"):
            return Decision(True, "human")
        note = "" if answer.lower() in ("", "n", "no") else answer
        return Decision(False, "human", note)
