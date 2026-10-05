import asyncio

import pytest

from opsagent import store as S
from opsagent import world as W
from opsagent.agent import RECOVERY_NOTICE, Agent
from opsagent.faults import Faults, SimulatedCrash
from opsagent.hub import ToolHub
from opsagent.policy import Policy
from opsagent.tracing import render

from .fakes import ScriptedLLM, say, tool_use


def run(coro):
    return asyncio.run(coro)


def agent_for(env, llm, mode="auto", faults=None, policy=None):
    """Returns build(fn): runs fn(agent) with a live MCP server subprocess behind the agent."""
    async def build(fn):
        async with ToolHub() as hub:
            agent = Agent(env, hub, llm, mode=mode, faults=faults, policy=policy, out=lambda s: None)
            return await fn(agent)
    return build


def test_happy_path_tool_calling_and_history_shape(env):
    llm = ScriptedLLM(
        [say("Looking."), tool_use("list_services"), tool_use("get_logs", service="checkout-api", grep="redis")],
        [say("Rolling back."), tool_use("rollback_deploy", service="checkout-api", version="2.4.0")],
        [say("Fixed and verified.")],
    )
    run_id = run(agent_for(env, llm)(lambda a: a.run("fix it")))
    assert env.get_run(run_id)["status"] == "done"
    assert W.load()["services"]["checkout-api"]["status"] == "running"
    msgs = env.messages(run_id)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    # every tool_use is answered by a tool_result with the same id, all in one user message
    for a, u in ((1, 2), (3, 4)):
        ids = [b["id"] for b in msgs[a]["content"] if b["type"] == "tool_use"]
        assert ids == [b["tool_use_id"] for b in msgs[u]["content"]]
    tiers = {r["name"]: r["tier"] for r in env.db.execute("SELECT name, tier FROM tool_calls")}
    assert tiers == {"list_services": "read", "get_logs": "read", "rollback_deploy": "mutate"}


def test_denied_action_is_not_executed_and_model_is_told(env):
    llm = ScriptedLLM([tool_use("rollback_deploy", service="checkout-api", version="2.4.0")], [say("ok, escalating")])
    run_id = run(agent_for(env, llm, mode="deny")(lambda a: a.run("fix it")))
    assert W.load()["services"]["checkout-api"]["version"] == "2.4.1"
    (res,) = llm.results_at(1)
    assert res["is_error"] and "DENIED" in res["content"] and "NOT performed" in res["content"]
    assert env.get_run(run_id)["status"] == "done"


def test_queue_mode_pauses_durably_then_resumes_after_human_decision(env):
    llm = ScriptedLLM([tool_use("rollback_deploy", service="checkout-api", version="2.4.0")], [say("done")])
    build = agent_for(env, llm, mode="queue")
    run_id = run(build(lambda a: a.run("fix it")))
    assert env.get_run(run_id)["status"] == "paused_for_approval"
    assert W.load()["services"]["checkout-api"]["version"] == "2.4.1"
    (call,) = env.awaiting(run_id)
    env.decide(run_id, call["tool_use_id"], True, "human:test", "go")
    run(build(lambda a: a.resume(run_id)))
    assert env.get_run(run_id)["status"] == "done"
    assert W.load()["services"]["checkout-api"]["version"] == "2.4.0"
    assert llm.calls == 2  # the paused turn was not re-asked of the model


def test_queue_mode_denial_with_reason_reaches_the_model(env):
    llm = ScriptedLLM([tool_use("restart_service", service="web")], [say("understood")])
    build = agent_for(env, llm, mode="queue")
    run_id = run(build(lambda a: a.run("t")))
    (call,) = env.awaiting(run_id)
    env.decide(run_id, call["tool_use_id"], False, "human:test", "change freeze until 18:00")
    run(build(lambda a: a.resume(run_id)))
    assert "change freeze until 18:00" in llm.results_at(1)[0]["content"]


def test_cannot_decide_twice(env):
    llm = ScriptedLLM([tool_use("restart_service", service="web")], [say("x")])
    run_id = run(agent_for(env, llm, mode="queue")(lambda a: a.run("t")))
    (call,) = env.awaiting(run_id)
    env.decide(run_id, call["tool_use_id"], False, "h")
    with pytest.raises(ValueError):
        env.decide(run_id, call["tool_use_id"], True, "h")


def test_crash_after_effect_before_result_marks_outcome_unknown_and_never_reruns(env):
    first = ScriptedLLM([say("rolling back"), tool_use("rollback_deploy", service="checkout-api", version="2.4.0")])
    with pytest.raises(SimulatedCrash):
        run(agent_for(env, first, faults=Faults("mid_tool:rollback_deploy"))(lambda a: a.run("fix it")))
    run_id = env.latest_run_id()
    assert env.get_run(run_id)["status"] == "running"
    assert W.load()["services"]["checkout-api"]["version"] == "2.4.0"  # the effect happened...
    (row,) = env.db.execute("SELECT * FROM tool_calls").fetchall()
    assert row["status"] == S.STARTED and row["result"] is None        # ...but was never recorded

    second = ScriptedLLM([tool_use("list_services")], [say("verified after crash")])
    run(agent_for(env, second)(lambda a: a.resume(run_id)))
    assert W.load()["audit"].count("rollback checkout-api 2.4.1->2.4.0") == 1  # not executed twice
    (res,) = second.results_at(0)
    assert res["is_error"] and res["content"] == RECOVERY_NOTICE
    assert env.db.execute("SELECT status FROM tool_calls WHERE name='rollback_deploy'").fetchone()[0] == S.UNKNOWN
    assert env.get_run(run_id)["status"] == "done"


def test_crash_after_llm_resumes_without_asking_the_model_again(env):
    first = ScriptedLLM([tool_use("get_metrics", service="worker")])
    with pytest.raises(SimulatedCrash):
        run(agent_for(env, first, faults=Faults("after_llm"))(lambda a: a.run("look")))
    run_id = env.latest_run_id()
    second = ScriptedLLM([say("worker is degraded")])
    run(agent_for(env, second)(lambda a: a.resume(run_id)))
    assert second.calls == 1  # only the post-tool turn; the tool_use turn was replayed from the journal
    assert "error_rate_pct" in second.results_at(0)[0]["content"]


def test_in_flight_read_only_call_is_safely_reexecuted(env):
    first = ScriptedLLM([tool_use("get_metrics", service="worker")])
    with pytest.raises(SimulatedCrash):
        run(agent_for(env, first, faults=Faults("mid_tool"))(lambda a: a.run("look")))
    run_id = env.latest_run_id()
    second = ScriptedLLM([say("done")])
    run(agent_for(env, second)(lambda a: a.resume(run_id)))
    (res,) = second.results_at(0)
    assert not res.get("is_error") and "cpu_pct" in res["content"]


def test_crash_after_all_tools_ran_loses_nothing(env):
    first = ScriptedLLM([tool_use("get_metrics", service="web"), tool_use("get_metrics", service="redis")])
    with pytest.raises(SimulatedCrash):
        run(agent_for(env, first, faults=Faults("after_tools"))(lambda a: a.run("look")))
    run_id = env.latest_run_id()
    second = ScriptedLLM([say("fine")])
    run(agent_for(env, second)(lambda a: a.resume(run_id)))
    assert len(second.results_at(0)) == 2


def test_trace_shows_where_the_process_died(env):
    first = ScriptedLLM([tool_use("restart_service", service="web")])
    with pytest.raises(SimulatedCrash):
        run(agent_for(env, first, faults=Faults("mid_tool"))(lambda a: a.run("t")))
    run_id = env.latest_run_id()
    run(agent_for(env, ScriptedLLM([say("ok")]))(lambda a: a.resume(run_id)))
    tree = render(env.spans(run_id))
    assert "process died here" in tree and "recovery" in tree
    kinds = {s["kind"] for s in env.spans(run_id)}
    assert {"run", "turn", "llm", "tool", "approval", "recovery"} <= kinds


def test_blocked_tool_is_denied_by_policy_even_in_auto_mode(env):
    llm = ScriptedLLM([tool_use("delete_files", paths=["/tmp/export-2026-09-30.csv"])], [say("ok")])
    run(agent_for(env, llm, policy=Policy(blocked_tools={"delete_files"}))(lambda a: a.run("clean")))
    assert "/tmp/export-2026-09-30.csv" in W.load()["files"]
    assert "blocked by operator policy" in env.db.execute("SELECT decision_note FROM tool_calls").fetchone()[0]


def test_stateful_restart_escalates_to_destructive_tier(env):
    llm = ScriptedLLM([tool_use("restart_service", service="postgres")], [say("ok")])
    run(agent_for(env, llm, mode="deny")(lambda a: a.run("t")))
    assert env.db.execute("SELECT tier FROM tool_calls").fetchone()[0] == "destructive"


def test_server_side_guardrails_hold_even_when_approved_and_errors_reach_the_model(env):
    llm = ScriptedLLM(
        [tool_use("delete_files", paths=["/var/log/worker/worker.log"])],
        [tool_use("delete_files", paths=["/var/lib/postgresql/16/main/base"])],
        [say("understood")],
    )
    run(agent_for(env, llm)(lambda a: a.run("clean")))
    assert "live file" in llm.results_at(1)[0]["content"]
    assert "are deletable" in llm.results_at(2)[0]["content"]
    assert "/var/lib/postgresql/16/main/base" in W.load()["files"]


def test_worker_recovers_after_cleanup(env):
    llm = ScriptedLLM(
        [tool_use("delete_files", paths=["/var/log/worker/worker.log.1", "/var/log/worker/worker.log.2", "/tmp/export-2026-09-30.csv"])],
        [say("done")],
    )
    run(agent_for(env, llm)(lambda a: a.run("disk")))
    w = W.load()
    assert w["services"]["worker"]["status"] == "running" and W.disk_ok(w)


def test_max_turns_stops_a_looping_model(env):
    llm = ScriptedLLM(*[[tool_use("list_services")] for _ in range(5)])

    async def go(a):
        a.max_turns = 3
        return await a.run("loop")

    run_id = run(agent_for(env, llm)(go))
    assert env.get_run(run_id)["status"] == "max_turns" and llm.calls == 3
