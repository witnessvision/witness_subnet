"""Durable 5.2 rounds and hotkey EMA; incomplete comparisons never rank."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
import uuid

from witness.events import content_hash


class ProductionState:
    def __init__(self, path: Path, identity: dict, *, start_after_epoch=-1):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY, epoch INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS rounds (id TEXT PRIMARY KEY, epoch INTEGER UNIQUE NOT NULL,
                status TEXT NOT NULL, jobs TEXT NOT NULL, endpoints TEXT NOT NULL, report TEXT);
            CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, round_id TEXT NOT NULL,
                hotkey TEXT NOT NULL, uid INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                status TEXT NOT NULL, result TEXT, UNIQUE(round_id,hotkey,ordinal));
            CREATE TABLE IF NOT EXISTS ranking (hotkey TEXT PRIMARY KEY, ema REAL NOT NULL,
                rounds INTEGER NOT NULL, last_round TEXT NOT NULL);
        """)
        self.identity = identity
        value = json.dumps(identity, sort_keys=True)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO identity VALUES(1,?)", (value,))
            self.db.execute("INSERT OR IGNORE INTO cursor VALUES(1,?)", (start_after_epoch,))
        if self.db.execute("SELECT value FROM identity WHERE id=1").fetchone()[0] != value:
            self.db.close()
            raise ValueError("new_evaluator_requires_new_ranking_series")

    def eligible(self, epoch):
        return epoch > self.db.execute("SELECT epoch FROM cursor WHERE id=1").fetchone()[0]

    def active(self):
        row = self.db.execute("SELECT * FROM rounds WHERE status='running'").fetchone()
        return self.decode_round(row) if row else None

    def decode_round(self, row):
        return {**dict(row), "jobs": json.loads(row["jobs"]), "endpoints": json.loads(row["endpoints"])}

    def skip(self, epoch):
        with self.db:
            self.db.execute("UPDATE cursor SET epoch=MAX(epoch,?) WHERE id=1", (epoch,))

    def begin(self, epoch, jobs, endpoints):
        if len(jobs) != 5 or len({j["original"] for j in jobs}) != 5:
            raise ValueError("five_distinct_clips_required")
        if len({e["hotkey"] for e in endpoints}) != len(endpoints) or len({e["uid"] for e in endpoints}) != len(endpoints):
            raise ValueError("duplicate_endpoint_identity")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.active() or not self.eligible(epoch):
                return None
            rid = uuid.uuid4().hex
            self.db.execute("INSERT INTO rounds VALUES(?,?,'running',?,?,NULL)",
                            (rid, epoch, json.dumps(jobs), json.dumps(endpoints)))
            self.db.execute("UPDATE cursor SET epoch=? WHERE id=1", (epoch,))
            for e in endpoints:
                for ordinal in range(5):
                    self.db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,'planned',NULL)",
                                    (uuid.uuid4().hex, rid, e["hotkey"], e["uid"], ordinal))
        return self.active()

    def tasks(self, rid):
        return [{**dict(r), "result": json.loads(r["result"]) if r["result"] else None}
                for r in self.db.execute("SELECT * FROM tasks WHERE round_id=? ORDER BY uid,ordinal", (rid,))]

    def claim(self, tid):
        with self.db:
            return self.db.execute("UPDATE tasks SET status='dispatching' WHERE id=? AND status='planned'", (tid,)).rowcount == 1

    def record(self, tid, result, *, status="finished"):
        with self.db:
            row = self.db.execute("UPDATE tasks SET result=?,status=? WHERE id=? AND status IN ('dispatching','evaluating')",
                                  (json.dumps(result, allow_nan=False), status, tid))
            if row.rowcount != 1:
                raise ValueError("task_not_claimed")

    def recover(self, epoch):
        # Never resend an ambiguous request or resume an old epoch as a burst.
        with self.db:
            rows = self.db.execute("""SELECT t.* FROM tasks t JOIN rounds r ON t.round_id=r.id
                WHERE t.status IN ('dispatching','evaluating') OR (t.status='planned' AND r.epoch<?)""", (epoch,)).fetchall()
            for row in rows:
                result = json.loads(row["result"]) if row["result"] else {}
                result.update(status="incomplete", evaluation_status="interrupted",
                              reward=None, previous_status=row["status"])
                self.db.execute("UPDATE tasks SET status='finished',result=? WHERE id=?", (json.dumps(result), row["id"]))

    def finish(self, rid, finish_epoch):
        rows = self.tasks(rid)
        if any(r["status"] != "finished" for r in rows):
            raise ValueError("unfinished_tasks")
        round_row = self.db.execute("SELECT * FROM rounds WHERE id=?", (rid,)).fetchone()
        if round_row["status"] != "running":
            return json.loads(round_row["report"])
        complete = bool(rows) and all(r["result"].get("reward") is not None for r in rows)
        miners = []
        for endpoint in json.loads(round_row["endpoints"]):
            results = [r["result"] for r in rows if r["hotkey"] == endpoint["hotkey"]]
            known = len(results) == 5 and all(r.get("reward") is not None for r in results)
            miners.append({**endpoint, "planned": 5,
                "sent": sum(r.get("dispatch_attempted", False) for r in results),
                "valid": sum(r.get("response_valid", False) for r in results),
                "scored": sum(r.get("evaluation_status") == "complete" for r in results),
                "mean_f1": sum(r.get("f1", 0.) for r in results)/5 if known else None,
                "mean_score": sum(r["reward"] for r in results)/5 if known else None,
                "max_elapsed_s": max((r.get("miner_elapsed_s", 0.) for r in results), default=None),
                "eligible": known and any(r.get("f1", 0.) > 0 for r in results)})
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if complete:
                for miner in miners:
                    old = self.db.execute("SELECT ema,rounds FROM ranking WHERE hotkey=?", (miner["hotkey"],)).fetchone()
                    ema = .2 * miner["mean_score"] + .8 * (old["ema"] if old else 0.)
                    self.db.execute("INSERT OR REPLACE INTO ranking VALUES(?,?,?,?)",
                                    (miner["hotkey"], ema, old["rounds"]+1 if old else 1, rid))
                    miner["ema"] = ema
            candidates = [m for m in miners if m["eligible"]] if complete else []
            winner = min(candidates, key=lambda m: (-m["ema"], m["uid"])) if candidates else None
            report = {"round_id": rid, "epoch": round_row["epoch"], "finish_epoch": finish_epoch,
                      "complete": complete, "identity": self.identity, "miners": miners,
                      "planned": len(rows), "winner": winner, "finished_unix": time.time(),
                      "tasks_hash": content_hash(rows)}
            self.db.execute("UPDATE rounds SET status=?,report=? WHERE id=?",
                            ("complete" if complete else "incomplete", json.dumps(report), rid))
            self.db.execute("UPDATE cursor SET epoch=MAX(epoch,?) WHERE id=1", (finish_epoch,))
        return report

    def latest(self):
        row = self.db.execute("SELECT status,report FROM rounds ORDER BY epoch DESC LIMIT 1").fetchone()
        return json.loads(row["report"]) if row and row["report"] else None

    def close(self):
        self.db.close()
