"""Persistent UTC daily reservations; provider and credential independent.

Each production role has exactly one ledger on its owning host. Fixed role
ceilings also bound the sum across hosts. Calibration uses the validator ledger.
Unsettled calls retain their ceiling, including after cancellation or restart.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import json
import sqlite3
import time
import uuid

NANO = 1_000_000_000
LIMITS = {"validator": 9 * NANO, "miner": NANO}


class BudgetUnavailable(RuntimeError):
    pass


def nanos(usd):
    value = Decimal(str(usd))
    if not value.is_finite() or value < 0:
        raise ValueError("invalid_cost")
    return int((value * NANO).to_integral_value(rounding=ROUND_CEILING))


class DailyBudget:
    def __init__(self, path: Path, *, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS reservations (
                    id TEXT PRIMARY KEY, day TEXT NOT NULL, role TEXT NOT NULL,
                    provider TEXT NOT NULL, input_hash TEXT NOT NULL,
                    reserved INTEGER NOT NULL, settled INTEGER, metadata TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS daily_role ON reservations(day, role);
                CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            """)
            policy = json.dumps(LIMITS, sort_keys=True)
            db.execute("INSERT OR IGNORE INTO policy VALUES(1, ?)", (policy,))
            if db.execute("SELECT value FROM policy WHERE id=1").fetchone()[0] != policy:
                raise BudgetUnavailable("budget_policy_changed")
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def day(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat()

    def reserve(self, *, role, provider, input_hash, upper_usd, call_id=None):
        if role not in LIMITS or provider not in ("saygm", "openai"):
            raise ValueError("invalid_budget_identity")
        amount, day = nanos(upper_usd), self.day()
        call_id = call_id or uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM reservations WHERE id=?", (call_id,)).fetchone():
                raise BudgetUnavailable("reservation_already_used")
            total, own = db.execute("""SELECT COALESCE(SUM(COALESCE(settled,reserved)),0),
                COALESCE(SUM(CASE WHEN role=? THEN COALESCE(settled,reserved) ELSE 0 END),0)
                FROM reservations WHERE day=?""", (role, day)).fetchone()
            if own + amount > LIMITS[role] or total + amount > sum(LIMITS.values()):
                raise BudgetUnavailable("daily_budget_exhausted")
            db.execute("INSERT INTO reservations VALUES(?,?,?,?,?,?,NULL,?)",
                       (call_id, day, role, provider, input_hash, amount, "{}"))
        return call_id

    def settle(self, call_id, cost_usd, metadata=None):
        cost = nanos(cost_usd)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved,settled FROM reservations WHERE id=?", (call_id,)).fetchone()
            if row is None:
                raise BudgetUnavailable("unknown_reservation")
            if row[1] is not None and row[1] != cost:
                raise BudgetUnavailable("settlement_changed")
            db.execute("UPDATE reservations SET settled=?,metadata=? WHERE id=?",
                       (cost, json.dumps(metadata or {}, allow_nan=False), call_id))
        if cost > row[0]:
            # Record the actual debt and stop accepting a wrongly priced result.
            raise BudgetUnavailable("provider_exceeded_reserved_ceiling")

    def totals(self, day=None):
        with self.connect() as db:
            rows = db.execute("""SELECT role,COUNT(*),COALESCE(SUM(settled),0),
                COALESCE(SUM(CASE WHEN settled IS NULL THEN reserved ELSE 0 END),0)
                FROM reservations WHERE day=? GROUP BY role""", (day or self.day(),)).fetchall()
        return {role: {"calls": count, "settled_usd": settled/NANO, "reserved_usd": reserved/NANO}
                for role, count, settled, reserved in rows}
