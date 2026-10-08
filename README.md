# OpsAgent

[![ci](https://github.com/Mohammmed-Waleed/OpsAgent-AI-/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohammmed-Waleed/OpsAgent-AI-/actions/workflows/ci.yml)

An on-call SRE agent built around the parts that make agents safe to run for real:

- **Tool calling** — Claude investigates an incident through tools, not guesses.
- **Its own MCP server** — the ops tools live in an [MCP](https://modelcontextprotocol.io) server (`opsagent/mcp_server.py`) that any MCP client can use. The agent talks to it over stdio.
- **Approval steps before risky actions** — read-only calls run freely; anything that changes state waits for a human, who can approve, or deny *with a reason the model reads*.
- **Crash recovery** — every step is journaled to SQLite *before* it happens. Kill the process at any point and `resume` continues, without repeating a mutation it can't prove didn't already happen.
- **Tracing** — every LLM call, tool call, approval and recovery is a span; `opsagent trace` renders the tree with timings, tokens and cost.

The environment is a simulation (a JSON file), so you can let an agent loose on a "production" that cannot hurt you. The scenario: `checkout-api` v2.4.1 crash-loops because the deploy moved `REDIS_PORT` to 6380 while Redis listens on 6379 (a restart can't fix it; a rollback can), and the worker's `/var` disk is 97% full.

## Try it without an API key

```bash
pip install -e ".[dev]"
python examples/offline_demo.py
```

A scripted "model" drives the real agent, MCP server, store and tracer. The process dies right after the rollback took effect but before its result was recorded; a fresh process resumes:

```
*** simulated crash at mid_tool (rollback_deploy)

=== process 2: fresh start, resumes from the journal ===
resuming 20261005-180404-90ec (was: running)
  ? rollback_deploy(service='checkout-api', version='2.4.0') was in flight when the process died: outcome unknown

├─X run run  [never closed]  <-- process died here
│  ├─  turn tools (msg 1)  [173ms]
│  │  ├─  tool list_services() [read] -> ok  [156ms]
│  │  └─  tool get_logs(service='checkout-api', grep='redis', lines=2) [read] -> ok  [11ms]
│  └─X turn tools (msg 3)  [0ms]  <-- process died here
│     └─X tool rollback_deploy(service='checkout-api', version='2.4.0') [mutate]  [never closed]
│        └─  approval rollback_deploy: approved by policy:auto-approve  [2ms]
└─  run resume  [46ms]
   ├─  recovery orphan-spans: 3 span(s) left open by a dead process were marked crashed  [1ms]
   ├─  tool rollback_deploy(...) [mutate] -> unknown
   │     └─  recovery rollback_deploy: mutating call was in flight; NOT re-executing, outcome marked unknown
   └─  tool get_metrics(service='checkout-api') [read] -> ok  [13ms]

rollbacks actually executed: 1
```

To be the approver yourself (deny the restart with a reason, then approve the rollback):

```bash
python examples/approval_demo.py
```

## Run it for real

```bash
python -m venv .venv && .venv\Scripts\activate      # or source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                                  # add ANTHROPIC_API_KEY (or `ant auth login`)

opsagent tools                    # the MCP tools and their risk tier
opsagent world --reset            # (re)create the incident
opsagent run                      # investigate the demo incident; asks before every change
opsagent run "why is the worker unhealthy?" --approve queue
opsagent runs
opsagent trace                    # span tree of the latest run
```

Defaults: `claude-opus-5-5`, adaptive thinking, `medium` effort (override with `--model`, `--effort` or `OPSAGENT_MODEL` / `OPSAGENT_EFFORT`).

### Approval modes (`--approve`)

| mode | behaviour |
|---|---|
| `ask` (default) | prompt in the terminal; `y` approves, any other text denies and is shown to the model as the reason |
| `queue` | park the run durably and exit. Later: `opsagent approve <run> --all [--deny -m "change freeze"]`, then `opsagent resume <run>` — the async/Slack-style flow |
| `deny` | refuse every non-read call |
| `auto` | approve everything — for sandboxes and tests only |

### Crash it on purpose

```bash
opsagent run --approve auto --crash-at mid_tool:rollback_deploy   # killed after the rollback ran, before it was recorded
opsagent resume
opsagent trace
```

Crash points: `after_llm`, `after_approval`, `mid_tool`, `after_tools`.

## How it works

```
            ┌──────────────────────── agent.py ─────────────────────────┐
 task ─────▶│ state machine over the persisted message log               │
            │  last msg = user        → call Claude                      │
            │  last msg = tool_use    → resolve each call:               │
            │      policy ▸ approval ▸ write-ahead ▸ MCP call ▸ record   │
            │  last msg = assistant   → done                             │
            └───────┬──────────────────────────┬────────────────┬────────┘
                    │                          │                │
              store.py (SQLite)          hub.py (MCP client)   tracing.py
              messages, tool_calls,             │               spans → tree
              spans — durable                   │ stdio
                                         mcp_server.py ──▶ world.py (simulated prod)
```

**Risk comes from the server.** Each MCP tool declares `readOnlyHint` / `destructiveHint`. `policy.py` maps that to a tier (read / mutate / destructive; an unannotated tool is treated as destructive, per the MCP spec defaults) and can escalate on arguments — restarting a stateful service (`redis`, `postgres`) becomes destructive. Operators can hard-block tools with `OPSAGENT_BLOCK=delete_files`. The server also enforces its own guardrails *regardless of approval*: `delete_files` refuses anything outside `/var/log` and `/tmp`, and refuses live files.

**The write-ahead journal** (`store.py`) records, in order: assistant turn + its tool calls (one transaction) → approval decision → `started` *before* the tool runs → result *after* → all `tool_result`s as one user message. After a crash each call is in a known state:

| found in state | meaning | recovery |
|---|---|---|
| `pending` / `awaiting_approval` / `approved` | never ran | continue (re-ask if approval was pending) |
| `done` / `failed` / `denied` | finished | reuse the stored result; never re-run |
| `started`, read-only tool | ran or not; harmless to repeat | re-execute |
| `started`, mutating tool | **unknown outcome** | do *not* re-run; tell the model the outcome is unknown and to verify state first |

The model sees a `RECOVERY NOTICE` as the tool result and — as the system prompt instructs — inspects current state before deciding anything. The model is never asked again for a turn already journaled, so a resume costs no repeated reasoning, and message history is append-only (so replayed thinking blocks stay valid and the prompt cache stays warm).

**Tracing.** Spans are written on open, so a killed process leaves them open; on resume they are marked `crashed` and the tree says where it died. `opsagent trace --json` dumps raw spans for other tooling.

## Layout

```
opsagent/
  agent.py       loop, approval gate, recovery
  mcp_server.py  the ops tools as an MCP server (also runnable standalone)
  hub.py         MCP client → Claude tool definitions
  policy.py      risk tiers from MCP annotations + escalation + blocklist
  approval.py    CLI approver and decision types
  store.py       SQLite journal (runs, messages, tool_calls, spans)
  tracing.py     spans + tree renderer + cost estimate
  llm.py         Anthropic Messages API wrapper (adaptive thinking, caching)
  faults.py      crash injection
  world.py       the simulated environment
tests/           scripted-model tests: approvals, denial, queue/resume, every crash point
examples/offline_demo.py
```

```bash
pytest -q     # spins up the real MCP server per test; no API key needed
```

## Limits (by design, for now)

- The environment is simulated; swapping `world.py` for real infrastructure is the point where you'd want per-call idempotency keys passed through the MCP tools, so "unknown outcome" can be resolved by the server instead of by re-inspection.
- Tool calls within a turn run sequentially to keep approvals orderly.
- The SDK handles retries of transient API errors; if it gives up, the run stays `running` and `opsagent resume` picks it up.
- The model refusing or hitting `max_tokens` ends the run as `failed` (a truncated tool call is never executed).
