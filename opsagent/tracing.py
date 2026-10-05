"""Tracing: nested spans persisted to the run store, plus a tree renderer.

Span kinds: run > turn > llm | tool | approval | recovery.  Every span records wall time,
status and attributes (tokens, tool args, decisions).  Spans are written on open, so a
process that dies mid-call leaves an 'open' span behind; resuming marks it 'crashed' and
the trace shows exactly where the previous process stopped.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from .faults import SimulatedCrash
from .store import Store

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar("opsagent_span", default=None)

# $ per 1M tokens (input, output) for cost estimates in the trace summary
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-fable-5-1": (10.0, 50.0),
          "claude-haiku-4-5": (1.0, 5.0)}


class Tracer:
    def __init__(self, store: Store, run_id: str):
        self.store, self.run_id = store, run_id

    @contextmanager
    def span(self, kind: str, name: str, **attrs: Any) -> Iterator["SpanHandle"]:
        span_id = uuid.uuid4().hex[:12]
        parent = _current.get()
        token = _current.set(span_id)
        self.store.span_open(span_id, self.run_id, parent, kind, name, attrs)
        handle = SpanHandle()
        try:
            yield handle
        except SimulatedCrash:
            raise  # a killed process cannot close its spans; leave this one open on purpose
        except BaseException as e:
            self.store.span_close(span_id, "error", {**handle.attrs, "error": f"{type(e).__name__}: {e}"})
            raise
        else:
            self.store.span_close(span_id, handle.status, handle.attrs)
        finally:
            _current.reset(token)


class SpanHandle:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}
        self.status = "ok"

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def fail(self, **attrs: Any) -> None:
        self.status = "error"
        self.attrs.update(attrs)


def _dur(s: dict[str, Any]) -> str:
    if s["status"] == "crashed":
        return "never closed"
    return "…" if s["end_ts"] is None else f"{(s['end_ts'] - s['start_ts']) * 1000:,.0f}ms"


def _label(s: dict[str, Any]) -> str:
    a = s["attrs"]
    if s["kind"] == "llm":
        return f"llm {a.get('model', '')}  in={a.get('input_tokens', 0)} out={a.get('output_tokens', 0)} stop={a.get('stop_reason', '?')}"
    if s["kind"] == "tool":
        args = ", ".join(f"{k}={v!r}" for k, v in (a.get("args") or {}).items())
        return f"tool {s['name']}({args}) [{a.get('tier', '?')}]" + (f" -> {a['outcome']}" if "outcome" in a else "")
    if s["kind"] == "approval":
        return f"approval {s['name']}: {a.get('decision', 'pending')}" + (f" by {a['by']}" if "by" in a else "") + (f" ({a['note']})" if a.get("note") else "")
    if s["kind"] == "recovery":
        return f"recovery {s['name']}: {a.get('detail', '')}"
    return f"{s['kind']} {s['name']}"


def render(spans: list[dict[str, Any]], model: str | None = None) -> str:
    kids: dict[str | None, list[dict[str, Any]]] = {}
    for s in spans:
        kids.setdefault(s["parent_id"], []).append(s)
    lines: list[str] = []
    mark = {"ok": " ", "error": "!", "crashed": "X", "open": "?"}

    def walk(parent: str | None, prefix: str) -> None:
        group = kids.get(parent, [])
        for i, s in enumerate(group):
            last = i == len(group) - 1
            lines.append(f"{prefix}{'└─' if last else '├─'}{mark.get(s['status'], ' ')} {_label(s)}  [{_dur(s)}]"
                         + (f"  ERROR {s['attrs']['error']}" if "error" in s["attrs"] else "")
                         + ("  <-- process died here" if s["status"] == "crashed" else ""))
            walk(s["id"], prefix + ("   " if last else "│  "))

    walk(None, "")
    llm = [s for s in spans if s["kind"] == "llm"]
    tin = sum(s["attrs"].get("input_tokens", 0) + s["attrs"].get("cache_read_tokens", 0) + s["attrs"].get("cache_write_tokens", 0) for s in llm)
    tout = sum(s["attrs"].get("output_tokens", 0) for s in llm)
    summary = f"\n{len(llm)} llm calls · {sum(1 for s in spans if s['kind'] == 'tool')} tool calls · {tin:,} in / {tout:,} out tokens"
    if model in PRICES and (tin or tout):
        pi, po = PRICES[model]
        summary += f" · ≈${(tin * pi + tout * po) / 1e6:.3f}"
    crashed = sum(1 for s in spans if s["status"] == "crashed")
    if crashed:
        summary += f" · {crashed} span(s) lost to a crash"
    return "\n".join(lines) + summary + "\nlegend: ' ' ok  '!' error  'X' crashed  '?' still open"
