"""Crash-recovery demo that needs no API key: a scripted 'model' drives the real agent, MCP server,
store and tracer.  The process is killed (simulated) right after the rollback took effect but before
its result was recorded; a fresh agent then resumes the run.

    python examples/offline_demo.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

tmp = Path(tempfile.mkdtemp())
os.environ["OPSAGENT_WORLD"] = str(tmp / "world.json")

from opsagent import world as W  # noqa: E402
from opsagent.agent import Agent  # noqa: E402
from opsagent.cli import ensure_utf8  # noqa: E402
from opsagent.faults import Faults, SimulatedCrash  # noqa: E402
from opsagent.hub import ToolHub  # noqa: E402
from opsagent.store import Store  # noqa: E402
from opsagent.tracing import render  # noqa: E402
from fakes import ScriptedLLM, say, tool_use  # noqa: E402

ensure_utf8()
W.reset()
store = Store(tmp / "ops.db")


async def attempt(llm, faults=None, resume=None):
    async with ToolHub() as hub:
        agent = Agent(store, hub, llm, mode="auto", faults=faults)
        return await (agent.resume(resume) if resume else agent.run("checkout is returning 5xx"))


print("=== process 1: investigates, then dies mid-rollback ===")
first = ScriptedLLM(
    [say("Checking status and recent errors."), tool_use("list_services"), tool_use("get_logs", service="checkout-api", grep="redis", lines=2)],
    [say("v2.4.1 moved REDIS_PORT to 6380 but redis listens on 6379. A restart can't fix config, so I'll roll back to 2.4.0."),
     tool_use("rollback_deploy", service="checkout-api", version="2.4.0")],
)
try:
    asyncio.run(attempt(first, Faults("mid_tool:rollback_deploy")))
except SimulatedCrash as e:
    print(f"\n*** {e}")
run_id = store.latest_run_id()

print("\n=== process 2: fresh start, resumes from the journal ===")
second = ScriptedLLM(
    [say("My rollback's outcome is unknown, so I'll check state before assuming anything."), tool_use("get_metrics", service="checkout-api")],
    [say("Verified: checkout-api is on 2.4.0, 0.1% errors, 3/3 replicas ready. The rollback did take effect; I did not repeat it.")],
)
asyncio.run(attempt(second, resume=run_id))

print("\n=== trace ===")
print(render(store.spans(run_id)))
print(f"\nrollbacks actually executed: {W.load()['audit'].count('rollback checkout-api 2.4.1->2.4.0')}")
