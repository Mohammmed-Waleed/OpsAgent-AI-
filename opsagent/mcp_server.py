"""OpsAgent's own MCP server: the ops tools, exposed over the Model Context Protocol.

Run standalone (``python -m opsagent.mcp_server``) and point any MCP client at it,
or let the agent spawn it over stdio.  Risk is declared with MCP tool annotations
(``readOnlyHint`` / ``destructiveHint``); the agent's approval policy reads them,
so the server - not the agent - owns what counts as dangerous.
"""

from __future__ import annotations

import functools
import json
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import world as W

mcp = MCPServer("opsagent-tools", instructions="Operate and inspect a small production environment.")

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
MUTATE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)

Service = Annotated[str, Field(description="Service name: web, checkout-api, worker, redis or postgres")]


def tool(annotations: ToolAnnotations):
    """Register an MCP tool; domain ValueErrors become ToolErrors so the model sees the message
    (any other exception is reported to the client only as a generic crash)."""
    def deco(fn):
        @functools.wraps(fn)
        def guarded(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ValueError as e:
                raise ToolError(str(e)) from e
        mcp.tool(annotations=annotations)(guarded)
        return fn
    return deco


def _mutating(fn):
    """Load world, apply the change, persist atomically."""
    def run(*args):
        w = W.load()
        out = fn(w, *args)
        W.save(w)
        return out
    return run


# ---------------------------------------------------------------------- read-only

@tool(READ)
def list_services() -> str:
    """List every service with its status, version, replica count and restart count."""
    w = W.load()
    return json.dumps({n: {k: s[k] for k in ("status", "version", "replicas", "restarts")}
                       for n, s in w["services"].items()}, indent=2)


@tool(READ)
def get_logs(service: Service, lines: Annotated[int, Field(ge=1, le=200)] = 40,
             grep: Annotated[str, Field(description="Only lines containing this text (case-insensitive)")] = "") -> str:
    """Return recent log lines for a service, optionally filtered."""
    out = W.logs(W.load(), service)
    if grep:
        out = [l for l in out if grep.lower() in l.lower()]
    return "\n".join(out[-lines:]) or "(no matching log lines)"


@tool(READ)
def get_metrics(service: Service) -> str:
    """Current CPU, memory, error rate, latency and replica health for a service."""
    return json.dumps(W.metrics(W.load(), service), indent=2)


@tool(READ)
def list_deploys(service: Service) -> str:
    """Deploy history for a service, newest first, with who shipped it and why."""
    w = W.load()
    W._svc(w, service)
    return json.dumps({"current": w["services"][service]["version"], "history": w["deploys"].get(service, [])}, indent=2)


@tool(READ)
def disk_usage() -> str:
    """Disk usage per mount point."""
    d = W.load()["disk"]
    return json.dumps({m: {**v, "pct": round(100 * v["used_gb"] / v["size_gb"])} for m, v in d.items()}, indent=2)


@tool(READ)
def list_files(prefix: Annotated[str, Field(description="Path prefix, e.g. /var/log or /tmp")]) -> str:
    """List files (with sizes in GB) whose path starts with a prefix."""
    files = {p: s for p, s in W.load()["files"].items() if p.startswith(prefix)}
    return json.dumps(files, indent=2) if files else "(no files under that prefix)"


# ----------------------------------------------------------------------- mutating

@tool(MUTATE)
def restart_service(service: Service) -> str:
    """Restart all replicas of a service. Causes a brief interruption."""
    return _mutating(W.restart)(service)


@tool(MUTATE)
def scale_service(service: Service, replicas: Annotated[int, Field(ge=1, le=10)]) -> str:
    """Set the replica count of a service."""
    return _mutating(W.scale)(service, replicas)


@tool(MUTATE)
def rollback_deploy(service: Service, version: Annotated[str, Field(description="A version from list_deploys")]) -> str:
    """Roll a service back to a previously deployed version."""
    return _mutating(W.rollback)(service, version)


# ---------------------------------------------------------------------- destructive

@tool(DESTROY)
def delete_files(paths: Annotated[list[str], Field(min_length=1, max_length=20, description="Absolute paths under /var/log or /tmp")]) -> str:
    """Permanently delete files. Only /var/log and /tmp are deletable; live log files are refused."""
    return _mutating(W.delete_files)(paths)


# --------------------------------------------------------------------- resources

@mcp.resource("ops://runbook")
def runbook() -> str:
    """Team runbook: how to triage and what is safe to do."""
    return (
        "# Runbook\n"
        "1. Triage with read-only tools first: list_services, get_metrics, get_logs.\n"
        "2. A service that crashes right after a deploy: compare list_deploys with the first error in the logs; "
        "rollback beats restart. A restart does not change the config that is crashing it.\n"
        "3. Disk > 90% on /var: delete only rotated logs (worker.log.N) and stale /tmp exports. Never the live log.\n"
        "4. postgres and redis hold state. Do not restart them unless nothing else explains the symptom.\n"
        "5. After any change, verify with get_metrics / get_logs before declaring the incident resolved.\n"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
