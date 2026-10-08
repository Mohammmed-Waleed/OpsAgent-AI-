"""Human-in-the-loop demo that needs no API key: a scripted 'model' drives the real agent, MCP server
and terminal approver.  You are the approver - try denying the restart with a reason, then approving
the rollback.

    python examples/approval_demo.py
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
from opsagent.hub import ToolHub  # noqa: E402
from opsagent.store import Store  # noqa: E402
from fakes import ScriptedLLM, say, tool_use  # noqa: E402


def after_restart_decision(messages):
    """Turn 3 reacts to whatever the human decided about the restart."""
    result = next(b for b in messages[-1]["content"] if b["type"] == "tool_result")
    if "DENIED" in result["content"]:
        lead = "Understood, no restart. Checking what v2.4.1 changed instead."
    else:
        lead = "Restarted, but the pods are still crash-looping, so the restart didn't help. Checking what v2.4.1 changed."
    return [say(lead), tool_use("list_deploys", service="checkout-api"),
            tool_use("get_logs", service="redis", lines=1)]


llm = ScriptedLLM(
    [say("Starting with service health and checkout-api's recent errors."),
     tool_use("list_services"), tool_use("get_logs", service="checkout-api", grep="redis", lines=2)],
    [say("checkout-api is crash-looping on v2.4.1 (41 restarts). I'll restart it to clear the bad state."),
     tool_use("restart_service", service="checkout-api")],
    after_restart_decision,
    [say("Found it: v2.4.1 moved Redis to port 6380, but redis listens on 6379, so every pod fails its startup "
         "check. Rolling back to 2.4.0 restores the working config."),
     tool_use("rollback_deploy", service="checkout-api", version="2.4.0")],
    [say("Verifying the fix."), tool_use("get_metrics", service="checkout-api"), tool_use("get_logs", service="web", lines=1)],
    [say("Resolved. Cause: the v2.4.1 deploy pointed checkout-api at redis:6380; redis serves 6379. "
         "Change: rolled checkout-api back to 2.4.0 (approved). Verified: 3/3 replicas ready, 0.1% errors, "
         "web upstream healthy. Follow-up for a human: fix REDIS_PORT in the 2.4.x branch before redeploying.")],
)


async def main():
    ensure_utf8()
    W.reset()
    store = Store(tmp / "ops.db")
    async with ToolHub() as hub:
        run_id = await Agent(store, hub, llm, mode="ask").run("Checkout is returning 5xx. Find out why and fix it.")
    print(f"\n[{store.get_run(run_id)['status']}] run {run_id}")


asyncio.run(main())
