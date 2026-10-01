"""Durable one-shot hotkeys and round-robin coldkeys (no challenge fee)."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time

from .submission import challenge_id, parse_submission

TERMINAL = ('done', 'rejected')


class Triggers:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS triggers (
            id INTEGER PRIMARY KEY, hotkey TEXT NOT NULL, value TEXT NOT NULL, block INTEGER NOT NULL,
            status TEXT NOT NULL, reason TEXT, observed_unix REAL, finished_unix REAL, result TEXT,
            UNIQUE(hotkey, value, block))''')
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(triggers)')}
        for name, spec in {'coldkey': 'TEXT', 'uid': 'INTEGER', 'model_id': 'TEXT', 'window_id': 'INTEGER',
                           'retry_block': 'INTEGER DEFAULT 0', 'attempts': 'INTEGER DEFAULT 0'}.items():
            if name not in columns:
                self.db.execute(f'ALTER TABLE triggers ADD COLUMN {name} {spec}')
        self.db.execute('CREATE TABLE IF NOT EXISTS coldkey_turns (coldkey TEXT PRIMARY KEY, turn INTEGER NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS king (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS admission_reset (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS admission_archive (trigger_id INTEGER PRIMARY KEY, record TEXT NOT NULL)')
        # Legacy completed uses are retained; old HF pending entries cannot run as v2.
        self.db.execute("UPDATE triggers SET status='legacy', reason='p2p_resubmission_required' "
                        "WHERE value NOT LIKE 'wm2|%' AND status IN ('queued','running','failed')")
        path.chmod(0o600)

    def apply_reset(self, reset):
        if not reset:
            return
        with self.lock:
            saved = self.db.execute('SELECT value FROM admission_reset WHERE id=1').fetchone()
            if saved:
                if json.loads(saved[0]) != reset:
                    raise ValueError('hotkey_reset_changed')
                return
            self.db.execute('BEGIN IMMEDIATE')
            try:
                for row in self.db.execute('SELECT * FROM triggers WHERE block<?', (reset['block'],)).fetchall():
                    self.db.execute('INSERT INTO admission_archive VALUES (?,?)', (row['id'], json.dumps(dict(row))))
                    self.db.execute("UPDATE triggers SET status='reset' WHERE id=?", (row['id'],))
                self.db.execute('INSERT INTO admission_reset VALUES (1,?)', (json.dumps(reset),))
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise

    def observe(self, commitments, registered, max_per_hotkey=1, *, coldkeys=None, uids=None):
        if max_per_hotkey != 1:
            raise ValueError('mainnet_requires_one_challenge_per_hotkey')
        coldkeys, uids = coldkeys or {}, uids or {}
        added = []
        rows = ([{'hotkey': key, **row} for key, row in commitments.items()]
                if isinstance(commitments, dict) else commitments)
        with self.lock:
            reset = self.db.execute('SELECT value FROM admission_reset WHERE id=1').fetchone()
            minimum_block = json.loads(reset[0])['block'] if reset else 0
            for row in sorted(rows, key=lambda r: (r['block'], r['hotkey'])):
                hotkey, value, block = row['hotkey'], row['value'], row['block']
                if hotkey not in registered or block < minimum_block:
                    continue
                try:
                    submission = parse_submission(value)
                except ValueError:
                    continue
                if self.db.execute("SELECT 1 FROM triggers WHERE hotkey=? AND status IN "
                                   "('queued','running','done','rejected','failed')", (hotkey,)).fetchone():
                    continue  # immutable first binding; no new database row for spam updates
                coldkey = coldkeys.get(hotkey)
                if coldkey is None or hotkey not in uids:
                    continue  # ownership must come from finalized chain, never the miner
                self.db.execute('BEGIN IMMEDIATE')
                try:
                    turn = self.db.execute('SELECT COALESCE(MAX(turn),0)+1 FROM coldkey_turns').fetchone()[0]
                    self.db.execute('INSERT OR IGNORE INTO coldkey_turns VALUES (?,?)', (coldkey, turn))
                    self.db.execute('''INSERT OR IGNORE INTO triggers
                        (hotkey,value,block,status,observed_unix,coldkey,uid,model_id)
                        VALUES (?,?,?,'queued',?,?,?,?)''',
                                    (hotkey, value, block, time.time(), coldkey, uids[hotkey], challenge_id(hotkey, submission.model_id)))
                    self.db.execute('COMMIT')
                except BaseException:
                    self.db.execute('ROLLBACK')
                    raise
                added.append({'hotkey': hotkey, 'coldkey': coldkey, 'model_id': challenge_id(hotkey, submission.model_id),
                              'block': block, 'value': value})
        return added

    def pending(self, count=1, *, before_block=2**63-1, current_block=2**63-1):
        """A preview of actual fair dispatch order; no rotation until start()."""
        with self.lock:
            rows = self.db.execute('''SELECT t.*, c.turn FROM triggers t JOIN coldkey_turns c USING(coldkey)
                WHERE status IN ('queued','running','failed') AND block < ?
                ORDER BY c.turn,t.block,t.hotkey''', (before_block,)).fetchall()
        groups = {}
        for r in rows:
            groups.setdefault(r['coldkey'], []).append(self._decode(r))
        # An infrastructure retry at the head cannot be overtaken by its owner's tail.
        for key in list(groups):
            if groups[key][0]['retry_block'] > current_block:
                del groups[key]
        result = []
        while groups and len(result) < count:
            for key in list(groups):
                result.append(groups[key].pop(0))
                if not groups[key]:
                    del groups[key]
                if len(result) == count:
                    break
        return result

    def start(self, trigger_id, window=None):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                row = self.db.execute('SELECT coldkey,status FROM triggers WHERE id=?', (trigger_id,)).fetchone()
                if row is None or row['status'] in (*TERMINAL, 'reset'):
                    raise ValueError('hotkey_already_consumed')
                turn = self.db.execute('SELECT COALESCE(MAX(turn),0)+1 FROM coldkey_turns').fetchone()[0]
                self.db.execute('UPDATE coldkey_turns SET turn=? WHERE coldkey=?', (turn, row['coldkey']))
                self.db.execute("UPDATE triggers SET status='running',window_id=?,attempts=attempts+1 WHERE id=?",
                                (window, trigger_id))
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise

    def defer(self, trigger_id, reason, retry_block=0):
        with self.lock:
            self.db.execute("UPDATE triggers SET status='queued',reason=?,retry_block=? WHERE id=? AND status='running'",
                            (reason, retry_block, trigger_id))

    def finish(self, trigger_id, status, reason=None, result=None):
        if status not in TERMINAL:
            raise ValueError('use_defer_for_infrastructure_failures')
        with self.lock:
            self.db.execute('''UPDATE triggers SET status=?,reason=?,finished_unix=?,result=?
                               WHERE id=? AND status NOT IN ('done','rejected','reset')''',
                            (status, reason, time.time(), json.dumps(result) if result else None, trigger_id))

    def reconcile(self, hotkey, result, *, window, rejected=False):
        with self.lock:
            row = self.db.execute("SELECT id FROM triggers WHERE hotkey=? AND status IN ('queued','running','failed')",
                                  (hotkey,)).fetchone()
            if row:
                self.db.execute('UPDATE triggers SET window_id=? WHERE id=?', (window, row['id']))
                self.finish(row['id'], 'rejected' if rejected else 'done', 'finalized_on_chain', result)

    @staticmethod
    def _decode(row):
        value = dict(row)
        if isinstance(value.get('result'), str):
            value['result'] = json.loads(value['result'])
        value['usage'] = 'consumed' if value['status'] in TERMINAL else 'reserved'
        return value

    def rows(self):
        with self.lock:
            return [self._decode(r) for r in self.db.execute('SELECT * FROM triggers ORDER BY block,hotkey')]

    @property
    def king(self):
        with self.lock:
            row = self.db.execute('SELECT value FROM king WHERE id=1').fetchone()
            return json.loads(row[0]) if row else None

    @king.setter
    def king(self, value):
        with self.lock:
            if value is None:
                self.db.execute('DELETE FROM king')
            else:
                self.db.execute('INSERT INTO king VALUES (1,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value',
                                (json.dumps(value),))

    def close(self):
        self.db.close()
