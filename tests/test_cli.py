import asyncio

from opsagent import cli
from opsagent.agent import Agent
from opsagent.hub import ToolHub

from .fakes import ScriptedLLM, say, tool_use


def _queued_run(env):
    async def go():
        async with ToolHub() as hub:
            llm = ScriptedLLM([say("rolling back because 2.4.1 broke redis config"),
                               tool_use("rollback_deploy", service="checkout-api", version="2.4.0")])
            return await Agent(env, hub, llm, mode="queue", out=lambda s: None).run("fix checkout")
    return asyncio.run(go())


def test_approve_runs_trace_commands(env, tmp_path, capsys):
    run_id = _queued_run(env)
    db = str(env.path)

    assert cli.main(["--db", db, "runs"]) == 0
    listing = capsys.readouterr().out
    assert run_id in listing and "paused_for_approval" in listing

    assert cli.main(["--db", db, "approve", run_id]) == 0  # lists, decides nothing
    out = capsys.readouterr().out
    assert "rollback_deploy" in out and "2.4.1 broke redis config" in out
    assert len(env.awaiting(run_id)) == 1

    assert cli.main(["--db", db, "approve", run_id, "--all", "--deny", "-m", "change freeze"]) == 0
    row = env.db.execute("SELECT status, decision_note, decided_by FROM tool_calls").fetchone()
    assert (row["status"], row["decision_note"], row["decided_by"]) == ("denied", "change freeze", "human:cli")

    assert cli.main(["--db", db, "trace", run_id]) == 0
    assert "approval rollback_deploy: queued" in capsys.readouterr().out


def test_tools_command_lists_risk_tiers(capsys):
    assert cli.main(["tools"]) == 0
    tiers = {line.split()[0]: line.split()[1] for line in capsys.readouterr().out.splitlines()}
    assert tiers["delete_files"] == "destructive" and tiers["get_logs"] == "read" and tiers["rollback_deploy"] == "mutate"
