"""Replayable finalized-chain ledger. No HTTP dashboard/report input is accepted."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading

from .protocol import BASELINE, EARLY_STOP, REJECTED, Result, WINDOW_EPOCHS, decide, LEGACY_POLICY, MEDIA_RECOVERY_WINDOW, FIVE_VIDEO_POLICY, TEN_VIDEO_WINDOW, policy_identity
from .submission import challenge_id, parse_submission


class Ledger:
    def __init__(self, path: Path, *, activation_block: int, activation_epoch: int, policy: str):
        if activation_block < 1 or activation_epoch < 0:
            raise ValueError('explicit_activation_block_and_epoch_required')
        self.activation_block, self.activation_epoch, self.policy = activation_block, activation_epoch, policy
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS blocks (block INTEGER PRIMARY KEY, hash TEXT NOT NULL, epoch INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS commitments (block INTEGER, hotkey TEXT, value TEXT,
                PRIMARY KEY(block,hotkey,value));
            CREATE INDEX IF NOT EXISTS commitments_hotkey ON commitments(hotkey,block);
            CREATE TABLE IF NOT EXISTS submissions (hotkey TEXT PRIMARY KEY, model TEXT NOT NULL, row TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS windows (id INTEGER PRIMARY KEY, opening TEXT NOT NULL, decision TEXT);
            CREATE TABLE IF NOT EXISTS uses (evaluator TEXT, hotkey TEXT, window INTEGER, result TEXT,
                PRIMARY KEY(evaluator,hotkey));
            CREATE TABLE IF NOT EXISTS bootstrap_pending (evaluator TEXT, window INTEGER, model TEXT, result TEXT,
                PRIMARY KEY(evaluator,window,model));''')
        expected = {'block': activation_block, 'epoch': activation_epoch, 'policy': policy}
        self.versioned_media = policy == policy_identity()
        previous = self.get('activation')
        if (previous and previous['block'] == activation_block and previous['epoch'] == activation_epoch
                and previous['policy'] in (LEGACY_POLICY, FIVE_VIDEO_POLICY) and self.versioned_media):
            self._migrate_media_policy(expected, previous['policy'])
        if self.get('activation') not in (None, expected):
            raise ValueError('ledger_activation_or_policy_changed_use_explicit_migration')
        self.set('activation', expected)
        path.chmod(0o600)

    def policy_for_window(self, window):
        if not self.versioned_media:
            return self.policy
        if window < MEDIA_RECOVERY_WINDOW:
            return LEGACY_POLICY
        return FIVE_VIDEO_POLICY if window < TEN_VIDEO_WINDOW else self.policy

    def _migrate_media_policy(self, expected, previous_policy):
        """The sole supported transition; reject any already reported affected window.

        Closed pre-transition decisions, king, uses, commitments and cursor remain
        byte-for-byte intact. Fresh replay uses the same per-window policy schedule.
        """
        first_window = MEDIA_RECOVERY_WINDOW if previous_policy == LEGACY_POLICY else TEN_VIDEO_WINDOW
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for row in self.db.execute('SELECT value FROM commitments'):
                try:
                    result = Result.parse(row[0])
                except ValueError:
                    continue
                if result.window >= first_window:
                    raise ValueError('media_migration_requires_unreported_windows')
            affected = list(self.db.execute('SELECT id,opening,decision FROM windows WHERE id>=?',
                                           (first_window,)))
            for row in affected:
                if row['decision']:
                    raise ValueError('media_migration_requires_unclosed_windows')
                opening = json.loads(row['opening'])
                opening['policy_hash'] = self.policy_for_window(row['id'])
                self.db.execute('UPDATE windows SET opening=? WHERE id=?', (json.dumps(opening), row['id']))
                if self.active and self.active['id'] == row['id']:
                    self.set('active', opening)
            self.set('policy_migration', {'from': previous_policy, 'to': self.policy,
                                         'first_window': first_window, 'cursor': self.cursor})
            self.set('activation', expected)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def get(self, key, default=None):
        with self.lock:
            row = self.db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute('INSERT INTO state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                        (key, json.dumps(value)))

    @property
    def cursor(self):
        return self.get('cursor', self.activation_block - 1)

    @property
    def active(self):
        return self.get('active')

    def submissions(self):
        with self.lock:
            return sorted((json.loads(r[0]) for r in self.db.execute('SELECT row FROM submissions')),
                          key=lambda r: (r['block'], r['hotkey']))

    def last_commitment_block(self, hotkey):
        with self.lock:
            return self.db.execute('SELECT MAX(block) FROM commitments WHERE hotkey=?', (hotkey,)).fetchone()[0]

    def usage(self, evaluator):
        with self.lock:
            return {r['hotkey']: {'window': r['window'], 'result': json.loads(r['result'])}
                    for r in self.db.execute('SELECT * FROM uses WHERE evaluator=?', (evaluator,))}

    def history(self, start, end):
        with self.lock:
            return [dict(r) for r in self.db.execute('SELECT * FROM commitments WHERE block>=? AND block<? '
                                                    'ORDER BY block,hotkey,value', (start, end))]

    def _voting_history(self, active, end):
        rows = []
        for row in self.history(active['start_block'], end):
            try:
                result = Result.parse(row['value'])
            except ValueError:
                continue
            if result.flags & BASELINE:
                rows.append(row)
                continue
            candidate = active['candidates'].get(result.model_id)
            if candidate is None:
                continue
            usage = self.usage(row['hotkey']).get(candidate['hotkey'])
            if result.flags & EARLY_STOP and (not usage or usage['window'] == active['id']):
                rows.append(row)
                continue
            if (usage and usage['window'] == active['id']
                    and usage['result'].get('commitment') == row['value']):
                rows.append(row)
        return rows

    def _open(self, snapshot, window):
        retired = set(self.get('bootstrap_retired', []))
        candidates = {}
        for row in self.submissions():
            if row['block'] < snapshot['block'] and row['hotkey'] in snapshot['uids'] and row['hotkey'] not in retired:
                row['uid'] = snapshot['uids'][row['hotkey']]
                candidates.setdefault(row['model_id'], row)
        king = self.get('king')
        if king and king['hotkey'] not in snapshot['uids']:
            king = None
        if king:
            king = {**king, 'uid': snapshot['uids'][king['hotkey']]}
            candidates.pop(king['model_id'], None)
        else:
            # Bootstrap rotates fair pairs every window: an offline submission
            # cannot indefinitely hold the entire subnet's first coronation.
            groups = {}
            for row in sorted(candidates.values(), key=lambda r: (r['block'], r['hotkey'])):
                groups.setdefault(row['coldkey'], []).append(row)
            order = []
            while groups:
                for owner in list(groups):
                    order.append(groups[owner].pop(0))
                    if not groups[owner]:
                        del groups[owner]
            after = self.get('bootstrap_after')
            index = next((i + 1 for i, r in enumerate(order) if r['hotkey'] == after), 0)
            order = order[index:] + order[:index]
            pair = order[:2] if len(order) >= 2 else []
            candidates = {r['model_id']: r for r in pair}
            if pair:
                self.set('bootstrap_after', pair[-1]['hotkey'])
        opening = {'id': window, 'start_block': snapshot['block'],
                   'start_epoch': self.activation_epoch + WINDOW_EPOCHS * window,
                   'end_epoch': self.activation_epoch + WINDOW_EPOCHS * (window + 1),
                   'king': king, 'candidates': candidates, 'state': 'open', 'policy_hash': self.policy_for_window(window)}
        self.db.execute('INSERT INTO windows VALUES (?,?,NULL)', (window, json.dumps(opening)))
        self.set('active', opening)

    def ingest(self, snapshot, rows):
        """One contiguous finalized block, with commitments at that exact block hash.

        The first block of the next window closes the old one BEFORE accepting its
        new commitments. Late results cannot slide into a different window.
        """
        block, epoch = snapshot['block'], snapshot['epoch_index']
        with self.lock:
            if block != self.cursor + 1:
                raise ValueError('noncontiguous_finalized_history')
            if epoch < self.activation_epoch:
                raise ValueError('activation_epoch_not_reached')
            window = (epoch - self.activation_epoch) // WINDOW_EPOCHS
            self.db.execute('BEGIN IMMEDIATE')
            try:
                active = self.active
                if active and window != active['id']:
                    decision = decide(self._voting_history(active, block), window=active['id'],
                                      policy=active['policy_hash'], king=active['king'],
                                      candidates=active['candidates'], snapshot=snapshot)
                    decision.update(block=block, block_hash=snapshot['block_hash'], end_epoch=epoch)
                    self.db.execute('UPDATE windows SET decision=? WHERE id=?', (json.dumps(decision), active['id']))
                    self.set('king', decision['king'])
                    self.set('decision', decision)
                    # A bounded loss consumes the attempt only after the final
                    # stake-weighted comparison. An unresolved partial can retry.
                    for item in decision['early_losses']:
                        partial = Result.parse(item['value'])
                        candidate = active['candidates'][partial.model_id]
                        value = json.dumps({**partial.public(), 'commitment': item['value']})
                        self.db.execute('INSERT OR IGNORE INTO uses VALUES (?,?,?,?)',
                                        (item['hotkey'], candidate['hotkey'], active['id'], value))
                    if not active['king'] and not decision['king']:
                        pair = {r['hotkey'] for r in active['candidates'].values()}
                        evaluated = any(pair and pair <= set(self.usage(v)) for v in snapshot['validators'])
                        if evaluated:
                            self.set('bootstrap_retired', sorted(set(self.get('bootstrap_retired', [])) | pair))
                if active is None or window != active['id']:
                    self._open(snapshot, window)
                active = self.active
                for row in sorted(rows, key=lambda r: (r['block'], r['hotkey'], r['value'])):
                    if row['block'] != block:
                        raise ValueError('commitment_not_from_this_block')
                    self.db.execute('INSERT OR IGNORE INTO commitments VALUES (?,?,?)',
                                    (block, row['hotkey'], row['value']))
                    try:
                        submission = parse_submission(row['value'])
                    except ValueError:
                        submission = None
                    if submission and row['hotkey'] in snapshot['uids']:
                        own = {**row, 'coldkey': snapshot['coldkeys'][row['hotkey']],
                               'uid': snapshot['uids'][row['hotkey']],
                               'model_id': challenge_id(row['hotkey'], submission.model_id)}
                        self.db.execute('INSERT OR IGNORE INTO submissions VALUES (?,?,?)',
                                        (row['hotkey'], submission.model_id, json.dumps(own)))
                    try:
                        result = Result.parse(row['value'])
                    except ValueError:
                        continue
                    candidate = active['candidates'].get(result.model_id)
                    if (result.window != window or result.policy != active['policy_hash'][:24] or result.flags & (BASELINE | EARLY_STOP)
                            or candidate is None or result.uid != candidate['uid']
                            or row['hotkey'] not in snapshot['validators']):
                        continue
                    previous = self.db.execute('SELECT window FROM uses WHERE evaluator=? AND hotkey=?',
                                               (row['hotkey'], candidate['hotkey'])).fetchone()
                    if previous is not None:
                        # Audit history remains intact; only the first terminal result votes.
                        continue
                    value = json.dumps({**result.public(), 'commitment': row['value']})
                    if active['king']:
                        self.db.execute('INSERT INTO uses VALUES (?,?,?,?)',
                                        (row['hotkey'], candidate['hotkey'], window, value))
                    else:
                        self.db.execute('INSERT OR IGNORE INTO bootstrap_pending VALUES (?,?,?,?)',
                                        (row['hotkey'], window, result.model_id, value))
                if not active['king'] and len(active['candidates']) == 2:
                    panels = {}
                    for r in self.db.execute('SELECT * FROM bootstrap_pending WHERE window=?', (window,)):
                        panels.setdefault(r['evaluator'], {})[r['model']] = r['result']
                    for evaluator, panel in panels.items():
                        if set(panel) != set(active['candidates']):
                            continue
                        for model, value in panel.items():
                            self.db.execute('INSERT OR IGNORE INTO uses VALUES (?,?,?,?)',
                                            (evaluator, active['candidates'][model]['hotkey'], window, value))
                self.db.execute('INSERT INTO blocks VALUES (?,?,?)', (block, snapshot['block_hash'], epoch))
                self.set('cursor', block)
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise

    def is_closed(self, window):
        with self.lock:
            row = self.db.execute('SELECT decision FROM windows WHERE id=?', (window,)).fetchone()
            return bool(row and row[0])

    def close(self):
        self.db.close()
