"""A small simulated production environment for the agent to operate on.

The world is a JSON file so that it survives process boundaries: the MCP server
runs as a subprocess, and a crashed agent run can be resumed against the state
its earlier (partially executed) tool calls left behind.

Scenario ("bad deploy + full disk"):
  * checkout-api v2.4.1 crash-loops because the deploy moved REDIS_PORT to 6380
    while redis listens on 6379.  Restarting does not help; rolling back does.
  * worker is degraded because /var is 97% full of rotated logs and old exports.
  * postgres is healthy and stateful - restarting it is a high-blast-radius act.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

SERVICES = ("web", "checkout-api", "worker", "redis", "postgres")

# Paths delete_files may touch at all, regardless of human approval.
DELETABLE_PREFIXES = ("/var/log/", "/tmp/")
PROTECTED_FILES = {"/var/log/worker/worker.log"}  # the live log file


def default_world() -> dict[str, Any]:
    return {
        "services": {
            "web": {"status": "running", "version": "1.25.3", "replicas": 2, "restarts": 0,
                    "stateful": False, "port": 443},
            "checkout-api": {"status": "crashloop", "version": "2.4.1", "replicas": 3, "restarts": 41,
                             "stateful": False, "port": 8080, "redis_port": 6380},
            "worker": {"status": "degraded", "version": "5.2.0", "replicas": 2, "restarts": 0,
                       "stateful": False, "port": 0},
            "redis": {"status": "running", "version": "7.2.4", "replicas": 1, "restarts": 0,
                      "stateful": True, "port": 6379},
            "postgres": {"status": "running", "version": "16.3", "replicas": 1, "restarts": 0,
                         "stateful": True, "port": 5432},
        },
        "deploys": {
            "checkout-api": [
                {"version": "2.4.1", "at": "2026-10-05T08:12:00Z", "by": "ci", "notes": "pool tuning, move redis to dedicated port"},
                {"version": "2.4.0", "at": "2026-09-28T14:03:00Z", "by": "ci", "notes": "tax rounding fix"},
                {"version": "2.3.9", "at": "2026-09-21T10:44:00Z", "by": "ci", "notes": "dependency bumps"},
            ],
            "web": [{"version": "1.25.3", "at": "2026-09-02T09:00:00Z", "by": "ci", "notes": "tls config"}],
            "worker": [{"version": "5.2.0", "at": "2026-09-30T16:20:00Z", "by": "ci", "notes": "batch export job"}],
            "redis": [{"version": "7.2.4", "at": "2026-08-12T09:00:00Z", "by": "ops", "notes": "patch"}],
            "postgres": [{"version": "16.3", "at": "2026-07-01T09:00:00Z", "by": "ops", "notes": "minor upgrade"}],
        },
        "disk": {
            "/": {"size_gb": 100, "used_gb": 41},
            "/var": {"size_gb": 200, "used_gb": 194},
            "/data": {"size_gb": 500, "used_gb": 212},
        },
        "files": {
            "/var/log/worker/worker.log": 14.2,
            "/var/log/worker/worker.log.1": 38.0,
            "/var/log/worker/worker.log.2": 36.5,
            "/var/log/worker/worker.log.3": 35.1,
            "/var/log/worker/worker.log.4": 34.8,
            "/tmp/export-2026-09-30.csv": 12.4,
            "/tmp/export-2026-10-01.csv": 12.9,
            "/var/lib/postgresql/16/main/base": 150.0,
        },
        "extra_logs": {},
        "audit": [],
    }


def world_path() -> Path:
    return Path(os.environ.get("OPSAGENT_WORLD", "data/world.json"))


def load() -> dict[str, Any]:
    p = world_path()
    if not p.exists():
        return default_world()
    return json.loads(p.read_text(encoding="utf-8"))


def save(world: dict[str, Any]) -> None:
    p = world_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(world, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def reset() -> dict[str, Any]:
    w = default_world()
    save(w)
    return w


def _svc(world: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in world["services"]:
        raise ValueError(f"unknown service {name!r}; valid services: {', '.join(SERVICES)}")
    return world["services"][name]


# --------------------------------------------------------------------------- reads

def logs(world: dict[str, Any], name: str) -> list[str]:
    s = _svc(world, name)
    base: list[str] = []
    if name == "checkout-api":
        if s["version"] == "2.4.1":
            for i in range(6):
                t = f"2026-10-05T09:{10 + i * 7:02d}:1{i}Z"
                base += [
                    f"{t} INFO  checkout-api v2.4.1 starting (pid {4100 + i})",
                    f"{t} INFO  loading config REDIS_HOST=redis REDIS_PORT=6380 POOL_SIZE=64",
                    f"{t} ERROR redis.exceptions.ConnectionError: Error 111 connecting to redis:6380. Connection refused.",
                    f"{t} FATAL startup health check failed, exiting with code 1",
                ]
        else:
            base += [
                "2026-10-05T09:58:02Z INFO  checkout-api v%s starting" % s["version"],
                "2026-10-05T09:58:03Z INFO  loading config REDIS_HOST=redis REDIS_PORT=6379 POOL_SIZE=16",
                "2026-10-05T09:58:03Z INFO  connected to redis:6379",
                "2026-10-05T09:58:04Z INFO  listening on :8080, health check OK",
            ]
    elif name == "web":
        base += [f"2026-10-05T09:5{i}:00Z WARN  upstream checkout-api: 502 Bad Gateway (no live upstreams)" for i in range(5)]
        if world["services"]["checkout-api"]["status"] == "running":
            base += ["2026-10-05T10:01:00Z INFO  upstream checkout-api healthy, 200 OK"]
    elif name == "worker":
        used = world["disk"]["/var"]["used_gb"] / world["disk"]["/var"]["size_gb"]
        if used > 0.9:
            base += [f"2026-10-05T09:4{i}:30Z WARN  /var/log/worker/worker.log: write failed: No space left on device" for i in range(5)]
            base += ["2026-10-05T09:45:00Z ERROR export job 8812 failed: cannot write /tmp/export-2026-10-05.csv (ENOSPC)"]
        else:
            base += ["2026-10-05T10:03:00Z INFO  disk pressure cleared, export job 8812 retried OK"]
    elif name == "redis":
        base += ["2026-10-05T00:00:01Z INFO  Ready to accept connections tcp on port 6379",
                 "2026-10-05T09:10:20Z WARN  connection refused attempts on port 6380 are not served here"]
    elif name == "postgres":
        base += ["2026-10-05T09:00:00Z LOG   checkpoint complete", "2026-10-05T09:30:00Z LOG   checkpoint complete"]
    return base + world["extra_logs"].get(name, [])


def metrics(world: dict[str, Any], name: str) -> dict[str, Any]:
    s = _svc(world, name)
    healthy = s["status"] == "running"
    if s["status"] == "crashloop":
        return {"cpu_pct": 3, "mem_pct": 11, "error_rate_pct": 100.0, "p95_ms": None,
                "restarts_total": s["restarts"], "replicas_ready": 0, "replicas_desired": s["replicas"]}
    return {"cpu_pct": 22 if healthy else 71, "mem_pct": 38 if healthy else 83,
            "error_rate_pct": 0.1 if healthy else 12.5, "p95_ms": 90 if healthy else 2300,
            "restarts_total": s["restarts"], "replicas_ready": s["replicas"], "replicas_desired": s["replicas"]}


def disk_ok(world: dict[str, Any]) -> bool:
    return world["disk"]["/var"]["used_gb"] / world["disk"]["/var"]["size_gb"] < 0.9


# ------------------------------------------------------------------------ mutations

def _audit(world: dict[str, Any], what: str) -> None:
    world["audit"].append(what)


def _refresh_worker(world: dict[str, Any]) -> None:
    w = world["services"]["worker"]
    if w["status"] == "degraded" and disk_ok(world):
        w["status"] = "running"


def restart(world: dict[str, Any], name: str) -> str:
    s = _svc(world, name)
    s["restarts"] += 1
    _audit(world, f"restart {name}")
    if name == "checkout-api" and s["version"] == "2.4.1":
        s["status"] = "crashloop"
        return "restarted checkout-api; pods started and exited again (still crash-looping on v2.4.1)"
    if name == "worker":
        s["status"] = "running" if disk_ok(world) else "degraded"
    else:
        s["status"] = "running"
    return f"restarted {name}; status={s['status']}"


def scale(world: dict[str, Any], name: str, replicas: int) -> str:
    s = _svc(world, name)
    if not 1 <= replicas <= 10:
        raise ValueError("replicas must be between 1 and 10")
    _audit(world, f"scale {name} {s['replicas']}->{replicas}")
    s["replicas"] = replicas
    return f"scaled {name} to {replicas} replicas; status={s['status']}"


def rollback(world: dict[str, Any], name: str, version: str) -> str:
    s = _svc(world, name)
    known = [d["version"] for d in world["deploys"].get(name, [])]
    if version not in known:
        raise ValueError(f"{name} has no deploy {version!r}; known versions: {', '.join(known)}")
    if version == s["version"]:
        return f"{name} is already on {version}; nothing to do"
    _audit(world, f"rollback {name} {s['version']}->{version}")
    s["version"] = version
    if name == "checkout-api":
        s["redis_port"] = 6379
        s["status"] = "running"
    else:
        s["status"] = "running"
    return f"rolled back {name} to {version}; rollout complete, status={s['status']}"


def delete_files(world: dict[str, Any], paths: list[str]) -> str:
    if not paths:
        raise ValueError("no paths given")
    freed = 0.0
    deleted: list[str] = []
    for p in paths:
        if p in PROTECTED_FILES:
            raise ValueError(f"refusing to delete {p}: it is a live file")
        if not p.startswith(DELETABLE_PREFIXES) or ".." in p:
            raise ValueError(f"refusing to delete {p}: only {', '.join(DELETABLE_PREFIXES)} are deletable")
        if p not in world["files"]:
            raise ValueError(f"no such file: {p}")
    for p in paths:
        size = world["files"].pop(p)
        freed += size
        deleted.append(p)
        # /tmp lives on /var in this world, which is why exports filled the disk
        world["disk"]["/var"]["used_gb"] = round(world["disk"]["/var"]["used_gb"] - size, 1)
    _audit(world, f"delete_files {deleted}")
    _refresh_worker(world)
    return f"deleted {len(deleted)} file(s), freed {freed:.1f} GB; /var is now {world['disk']['/var']['used_gb']}/{world['disk']['/var']['size_gb']} GB"


def snapshot(world: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({k: world[k] for k in ("services", "disk", "audit")})
