"""Durable run state (SQLite, WAL).  This is what makes crash recovery possible.

Write-ahead discipline, in the order the agent does things:

  1. assistant message + one tool_calls row per tool_use block   (one transaction)
  2. approval decision recorded                                  (awaiting_approval -> approved|denied)
  3. status 'started' written BEFORE the tool is executed        (the intent record)
  4. result written AFTER it returns                             (started -> done|failed)
  5. the user message carrying all tool_results                  (one transaction)

After a crash the tool_calls table says exactly how far each call got.  A call stuck
in 'started' has an unknown outcome: re-running a read is safe, re-running a mutation
is not, so the agent reports the uncertainty to the model instead of guessing.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, task TEXT NOT NULL, status TEXT NOT NULL, model TEXT NOT NULL,
  config TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, updated_at REAL NOT NULL, final_text TEXT
);
CREATE TABLE IF NOT EXISTS messages (
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL,
  PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS tool_calls (
  run_id TEXT NOT NULL, tool_use_id TEXT NOT NULL, idx INTEGER NOT NULL, msg_seq INTEGER NOT NULL,
  name TEXT NOT NULL, args TEXT NOT NULL, tier TEXT NOT NULL, status TEXT NOT NULL,
  reason TEXT, decided_by TEXT, decision_note TEXT, result TEXT, is_error INTEGER NOT NULL DEFAULT 0,
  started_at REAL, finished_at REAL,
  PRIMARY KEY (run_id, tool_use_id)
);
CREATE TABLE IF NOT EXISTS spans (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL,
  start_ts REAL NOT NULL, end_ts REAL, status TEXT NOT NULL DEFAULT 'open', attrs TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS spans_run ON spans(run_id, start_ts);
"""

# tool_call statuses
PENDING, AWAITING, APPROVED, DENIED, STARTED, DONE, FAILED, UNKNOWN = (
    "pending", "awaiting_approval", "approved", "denied", "started", "done", "failed", "unknown")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")  # a crash must not lose an acknowledged write
        self.db.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")

    def close(self) -> None:
        self.db.close()

    # ----------------------------------------------------------------- runs
    def create_run(self, task: str, model: str, config: dict[str, Any]) -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        now = time.time()
        with self.tx() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,NULL)",
                       (run_id, task, "running", model, json.dumps(config), now, now))
            self._add_message(db, run_id, 0, "user", [{"type": "text", "text": task}])
        return run_id

    def get_run(self, run_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        return row

    def set_status(self, run_id: str, status: str, final_text: str | None = None) -> None:
        self.db.execute("UPDATE runs SET status=?, updated_at=?, final_text=COALESCE(?, final_text) WHERE id=?",
                        (status, time.time(), final_text, run_id))

    def list_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

    def latest_run_id(self) -> str | None:
        row = self.db.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    # ------------------------------------------------------------- messages
    @staticmethod
    def _add_message(db: sqlite3.Connection, run_id: str, seq: int, role: str, content: list[dict]) -> None:
        db.execute("INSERT INTO messages VALUES (?,?,?,?,?)", (run_id, seq, role, json.dumps(content), time.time()))

    def messages(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT seq, role, content FROM messages WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        return [{"seq": r["seq"], "role": r["role"], "content": json.loads(r["content"])} for r in rows]

    def append_assistant(self, run_id: str, content: list[dict], calls: list[dict[str, Any]]) -> int:
        """Persist an assistant turn and its tool_call rows atomically (step 1)."""
        with self.tx() as db:
            seq = db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE run_id=?", (run_id,)).fetchone()[0]
            self._add_message(db, run_id, seq, "assistant", content)
            for i, c in enumerate(calls):
                db.execute("INSERT INTO tool_calls (run_id, tool_use_id, idx, msg_seq, name, args, tier, status, reason)"
                           " VALUES (?,?,?,?,?,?,?,?,?)",
                           (run_id, c["id"], i, seq, c["name"], json.dumps(c["input"]), c["tier"], PENDING, c.get("reason")))
            db.execute("UPDATE runs SET updated_at=? WHERE id=?", (time.time(), run_id))
        return seq

    def append_tool_results(self, run_id: str, results: list[dict]) -> int:
        """Persist the user message carrying all tool_results (step 5)."""
        with self.tx() as db:
            seq = db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE run_id=?", (run_id,)).fetchone()[0]
            self._add_message(db, run_id, seq, "user", results)
            db.execute("UPDATE runs SET updated_at=? WHERE id=?", (time.time(), run_id))
        return seq

    # ----------------------------------------------------------- tool calls
    def calls_for(self, run_id: str, msg_seq: int) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM tool_calls WHERE run_id=? AND msg_seq=? ORDER BY idx", (run_id, msg_seq)).fetchall()

    def set_call(self, run_id: str, tool_use_id: str, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE tool_calls SET {cols} WHERE run_id=? AND tool_use_id=?",
                        (*fields.values(), run_id, tool_use_id))

    def awaiting(self, run_id: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM tool_calls WHERE run_id=? AND status=? ORDER BY msg_seq, idx",
                               (run_id, AWAITING)).fetchall()

    def decide(self, run_id: str, tool_use_id: str, approve: bool, by: str, note: str = "") -> None:
        row = self.db.execute("SELECT status FROM tool_calls WHERE run_id=? AND tool_use_id=?", (run_id, tool_use_id)).fetchone()
        if row is None:
            raise KeyError(f"no such tool call: {tool_use_id}")
        if row["status"] != AWAITING:
            raise ValueError(f"call is {row['status']}, not awaiting approval")
        self.set_call(run_id, tool_use_id, status=APPROVED if approve else DENIED, decided_by=by, decision_note=note)

    # ----------------------------------------------------------------- spans
    def span_open(self, span_id: str, run_id: str, parent_id: str | None, kind: str, name: str, attrs: dict) -> None:
        self.db.execute("INSERT INTO spans (id, run_id, parent_id, kind, name, start_ts, attrs) VALUES (?,?,?,?,?,?,?)",
                        (span_id, run_id, parent_id, kind, name, time.time(), json.dumps(attrs, default=str)))

    def span_close(self, span_id: str, status: str, attrs: dict) -> None:
        row = self.db.execute("SELECT attrs FROM spans WHERE id=?", (span_id,)).fetchone()
        merged = {**json.loads(row["attrs"]), **attrs} if row else attrs
        self.db.execute("UPDATE spans SET end_ts=?, status=?, attrs=? WHERE id=?",
                        (time.time(), status, json.dumps(merged, default=str), span_id))

    def close_orphan_spans(self, run_id: str) -> int:
        """Spans still open when a run is resumed belong to a process that died."""
        cur = self.db.execute("UPDATE spans SET status='crashed', end_ts=COALESCE(end_ts, start_ts) WHERE run_id=? AND status='open'",
                              (run_id,))
        return cur.rowcount

    def spans(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM spans WHERE run_id=? ORDER BY start_ts, rowid", (run_id,)).fetchall()
        return [{**dict(r), "attrs": json.loads(r["attrs"])} for r in rows]
