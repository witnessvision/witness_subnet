"""Durable v5 epoch and dispatch journal. No automatic task retries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


class EventsState:
    def __init__(self, root: Path, *, netuid: int, target_hotkey: str, start_after_epoch: int = -1):
        if isinstance(start_after_epoch, bool) or not isinstance(start_after_epoch, int) or start_after_epoch < -1:
            raise ValueError("invalid_initial_epoch_floor")
        if root.exists() and not (root / "scheduler.sqlite3").exists():
            if any(p.name != ".owner.lock" for p in root.iterdir()):
                raise ValueError("v5_requires_a_fresh_history_root")
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.path = root / "scheduler.sqlite3"
        self.db = sqlite3.connect(self.path)
        self.path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY CHECK(id=1), epoch INTEGER NOT NULL);
            INSERT OR IGNORE INTO cursor VALUES(1,-1);
            CREATE TABLE IF NOT EXISTS rounds (id TEXT PRIMARY KEY, epoch INTEGER UNIQUE NOT NULL,
                status TEXT NOT NULL, started REAL NOT NULL, finished REAL, finish_epoch INTEGER);
            CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, round_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, original TEXT NOT NULL, payload TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT, UNIQUE(round_id,ordinal), UNIQUE(round_id,original));
        """)
        identity = json.dumps({"netuid": netuid, "target_hotkey": target_hotkey,
                               "schema_version": "5.0", "tasks_per_epoch": 5}, sort_keys=True)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO identity VALUES(1,?)", (identity,))
        if self.db.execute("SELECT value FROM identity").fetchone()[0] != identity:
            self.db.close()
            raise ValueError("scheduler_identity_mismatch")
        # Deployment records the last epoch owned by the previous evaluator.
        # Persist this floor once; restarting in a later epoch must not move it.
        with self.db:
            self.db.execute("UPDATE cursor SET epoch=? WHERE id=1 AND epoch=-1", (start_after_epoch,))

    def active(self) -> dict | None:
        row = self.db.execute("SELECT * FROM rounds WHERE status='running'").fetchone()
        return dict(row) if row else None

    def eligible(self, epoch: int) -> bool:
        return epoch > self.db.execute("SELECT epoch FROM cursor").fetchone()[0]

    def skip_epoch(self, epoch: int) -> None:
        """Consume an unavailable preparation epoch without manufacturing tasks."""
        if self.active():
            raise ValueError('cannot_skip_active_round')
        with self.db:
            self.db.execute('UPDATE cursor SET epoch=MAX(epoch,?) WHERE id=1',(epoch,))

    def begin(self, epoch: int, jobs: list[dict]) -> str | None:
        if len(jobs) != 5 or len({j["original"] for j in jobs}) != 5:
            raise ValueError("round_requires_five_distinct_originals")
        if self.active():
            raise ValueError("round_already_active")
        round_id = "v5-" + uuid.uuid4().hex
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if not self.eligible(epoch):
                return None
            self.db.execute("UPDATE cursor SET epoch=? WHERE id=1", (epoch,))
            self.db.execute("INSERT INTO rounds VALUES(?,?,'running',?,NULL,NULL)",
                            (round_id, epoch, time.time()))
            for i, job in enumerate(jobs):
                self.db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,'planned',NULL)",
                    (uuid.uuid4().hex, round_id, i, job["original"], json.dumps(job),))
        return round_id

    def tasks(self, round_id: str) -> list[dict]:
        return [{**dict(r), "payload": json.loads(r["payload"]),
                 "result": json.loads(r["result"]) if r["result"] else None}
                for r in self.db.execute("SELECT * FROM tasks WHERE round_id=? ORDER BY ordinal", (round_id,))]

    def claim(self, task_id: str) -> bool:
        with self.db:
            # Commit BEFORE handing control to the network, so uncertain sends
            # following a crash are recorded, never retried.
            row = self.db.execute("UPDATE tasks SET status='dispatching' WHERE id=? AND status='planned'", (task_id,))
            return row.rowcount == 1

    def finish_task(self, task_id: str, result: dict) -> None:
        with self.db:
            self.db.execute("UPDATE tasks SET status='finished',result=? WHERE id=? AND status='dispatching'",
                            (json.dumps(result, allow_nan=False), task_id))

    def recover(self) -> None:
        with self.db:
            for row in self.db.execute("SELECT id FROM tasks WHERE status='dispatching'").fetchall():
                self.db.execute("UPDATE tasks SET status='finished',result=? WHERE id=?",
                    (json.dumps({"task_id":row[0],"status":"interrupted_unknown","send_confirmed":None,
                                 "evaluation":None,"miner_elapsed_s":None}),row[0]))

    def finish_round(self, round_id: str, finish_epoch: int) -> None:
        if any(t["status"] != "finished" for t in self.tasks(round_id)):
            raise ValueError("unfinished_tasks")
        with self.db:
            self.db.execute("UPDATE rounds SET status='completed',finished=?,finish_epoch=? WHERE id=?",
                            (time.time(), finish_epoch, round_id))
            self.db.execute("UPDATE cursor SET epoch=MAX(epoch,?) WHERE id=1", (finish_epoch,))

    def close(self):
        self.db.close()
