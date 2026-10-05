"""Command line: run / resume / approve / runs / trace / tools / world."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from . import world as W
from .agent import Agent
from .approval import MODES
from .faults import Faults, SimulatedCrash
from .hub import ToolHub
from .policy import Policy
from .store import Store
from .tracing import render

DEFAULT_DB = "data/opsagent.db"
DEMO_TASK = "Customers report checkout is failing with 5xx errors, and the worker's exports look stuck. Find out why and fix what you safely can."


def ensure_utf8() -> None:
    """Windows consoles default to a legacy codepage; the tree/box characters need UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _load_dotenv() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _store(args: argparse.Namespace) -> Store:
    return Store(args.db)


def _check_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    print("ANTHROPIC_API_KEY is not set. Put it in .env (see .env.example) or run `ant auth login`.", file=sys.stderr)
    return False


async def _go(args: argparse.Namespace, resume_id: str | None) -> int:
    from .llm import AnthropicLLM
    store = _store(args)
    run_id = resume_id
    try:
        async with ToolHub() as hub:
            agent = Agent(store, hub, AnthropicLLM(model=args.model, effort=args.effort), policy=Policy(),
                          mode=args.approve, max_turns=args.max_turns, faults=Faults(args.crash_at))
            run_id = await (agent.resume(resume_id) if resume_id else agent.run(args.task))
    except SimulatedCrash as e:
        print(f"\n*** {e} - process killed without cleanup. Recover with: opsagent resume {store.latest_run_id()}", file=sys.stderr)
        os._exit(137)  # like SIGKILL: no atexit, no span cleanup
    except KeyboardInterrupt:
        print("\ninterrupted. The run is durable; continue with `opsagent resume`.", file=sys.stderr)
        return 130
    row = store.get_run(run_id)
    print(f"\n[{row['status']}] run {run_id}   (opsagent trace {run_id})")
    return 0 if row["status"] in ("done", "paused_for_approval") else 1


def cmd_run(a: argparse.Namespace) -> int:
    return asyncio.run(_go(a, None)) if _check_key() else 2


def cmd_resume(a: argparse.Namespace) -> int:
    a.run_id = a.run_id or _store(a).latest_run_id()
    if not a.run_id:
        print("no runs yet", file=sys.stderr)
        return 2
    return asyncio.run(_go(a, a.run_id)) if _check_key() else 2


def cmd_approve(a: argparse.Namespace) -> int:
    store = _store(a)
    run_id = a.run_id or store.latest_run_id()
    pending = store.awaiting(run_id)
    if not pending:
        print("nothing awaiting approval")
        return 0
    for r in pending:
        if a.call and not r["tool_use_id"].startswith(a.call):
            continue
        print(f"{r['tool_use_id']}  [{r['tier']}]  {r['name']}({r['args']})\n   why: {(r['reason'] or '').strip()[:200]}")
        if a.call or a.all:
            store.decide(run_id, r["tool_use_id"], not a.deny, "human:cli", a.message or "")
            print("   ->", "denied" if a.deny else "approved")
    if not (a.call or a.all):
        print(f"\ndecide with: opsagent approve {run_id} --all [--deny -m 'reason']   then   opsagent resume {run_id}")
    return 0


def cmd_runs(a: argparse.Namespace) -> int:
    for r in _store(a).list_runs():
        print(f"{r['id']}  {r['status']:<20} {r['task'][:70]}")
    return 0


def cmd_trace(a: argparse.Namespace) -> int:
    store = _store(a)
    run_id = a.run_id or store.latest_run_id()
    spans = store.spans(run_id)
    print(json.dumps(spans, indent=2) if a.json else render(spans, store.get_run(run_id)["model"]))
    return 0


def cmd_tools(a: argparse.Namespace) -> int:
    async def go() -> None:
        async with ToolHub() as hub:
            for t in hub.tools:
                tier = Policy().classify(t["name"], hub.annotations[t["name"]], {})
                print(f"{t['name']:<16} {tier:<12} {t['description'].splitlines()[0]}")
    asyncio.run(go())
    return 0


def cmd_world(a: argparse.Namespace) -> int:
    w = W.reset() if a.reset else W.load()
    if a.reset:
        print("world reset to the incident scenario")
    print(json.dumps(W.snapshot(w), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opsagent", description="Ops agent with MCP tools, approvals, crash recovery and tracing.")
    p.add_argument("--db", default=os.environ.get("OPSAGENT_DB", DEFAULT_DB), help="run store (SQLite)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def agent_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--approve", choices=MODES, default="ask", help="how risky calls are approved (default: ask)")
        sp.add_argument("--model", default=None)
        sp.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
        sp.add_argument("--max-turns", type=int, default=25)
        sp.add_argument("--crash-at", metavar="POINT[:TOOL]", default=None,
                        help="inject a crash to demo recovery: after_llm | after_approval | mid_tool | after_tools")

    r = sub.add_parser("run", help="start a run")
    r.add_argument("task", nargs="?", default=DEMO_TASK)
    agent_opts(r)
    r.set_defaults(fn=cmd_run)

    r = sub.add_parser("resume", help="continue a crashed, interrupted or paused run")
    r.add_argument("run_id", nargs="?")
    agent_opts(r)
    r.set_defaults(fn=cmd_resume)

    r = sub.add_parser("approve", help="decide pending approvals of a queued run")
    r.add_argument("run_id", nargs="?")
    r.add_argument("--call", help="tool_use id (prefix ok)")
    r.add_argument("--all", action="store_true")
    r.add_argument("--deny", action="store_true")
    r.add_argument("-m", "--message", help="note recorded with the decision (shown to the model on denial)")
    r.set_defaults(fn=cmd_approve)

    sub.add_parser("runs", help="list runs").set_defaults(fn=cmd_runs)
    r = sub.add_parser("trace", help="show the span tree of a run")
    r.add_argument("run_id", nargs="?")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_trace)
    sub.add_parser("tools", help="list MCP tools and their risk tier").set_defaults(fn=cmd_tools)
    r = sub.add_parser("world", help="show or reset the simulated environment")
    r.add_argument("--reset", action="store_true")
    r.set_defaults(fn=cmd_world)
    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    ensure_utf8()
    args = build_parser().parse_args(argv)
    return args.fn(args)
