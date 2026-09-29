"""Operational scheduling and cache lifecycle outside versioned scoring."""
from __future__ import annotations

from contextlib import nullcontext
import json
import time

from witness.storage import write_private
from .compression import CompressionRequired
from .evaluator import Evaluator
from .protocol import Result
from .triggers import Triggers


class MinerUnavailable(RuntimeError):
    """Miner transport failed; withdraw locally without publishing a result."""


class ScheduledTriggers(Triggers):
    """Dispatch the active window fairly, without penalizing planned cancellation."""

    def __init__(self, path, *, candidates, interruption, current_block=None):
        super().__init__(path)
        self.candidates, self.interruption = candidates, interruption
        self.current_block = current_block
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(triggers)')}
        for name in ('removed_block', 'readmission_block'):
            if name not in columns:
                self.db.execute(f'ALTER TABLE triggers ADD COLUMN {name} INTEGER')
        self.db.execute('CREATE TABLE IF NOT EXISTS queue_cursor (id INTEGER PRIMARY KEY, block INTEGER)')

    def observe(self, commitments, registered, *args, **kwargs):
        # Keep the first binding even while a transport is excluded.
        with self.lock:
            withdrawn = list(self.db.execute("SELECT * FROM triggers WHERE status='withdrawn'"))
            rows = ([{'hotkey': key, **row} for key, row in commitments.items()]
                    if isinstance(commitments, dict) else commitments)
            for old in withdrawn:
                fresh = [r for r in rows if r['hotkey'] == old['hotkey']
                         and r['value'] == old['value'] and r['block'] > old['removed_block']
                         and r['hotkey'] in registered]
                if fresh:
                    block = min(r['block'] for r in fresh)
                    self.db.execute("UPDATE triggers SET status='queued',reason='resubmitted',"
                                    "readmission_block=?,retry_block=0 WHERE id=?", (block, old['id']))
            excluded = {r['hotkey'] for r in self.excluded()}
            # Keep the immutable first binding, including withdrawn records.
            blocked = {r['hotkey'] for r in withdrawn}
            return super().observe(rows, set(registered) - excluded - blocked, *args, **kwargs)

    def rows(self):
        return [row for row in super().rows() if row['status'] not in ('excluded', 'withdrawn')]

    def excluded(self):
        with self.lock:
            return [self._decode(r) for r in self.db.execute(
                "SELECT * FROM triggers WHERE status='excluded' ORDER BY retry_block,block,hotkey")]

    def compression_retry(self, trigger_id, *, ready, block):
        with self.lock:
            self.db.execute("UPDATE triggers SET status=?,reason=?,retry_block=? "
                            "WHERE id=? AND status='excluded'", (
                                'queued' if ready else 'excluded',
                                'compression_ready' if ready else 'compression_required',
                                0 if ready else block + 25, trigger_id))

    def reconcile(self, hotkey, result, *, window, rejected=False):
        # Only the finalized ledger can turn an excluded entry into a used one.
        with self.lock:
            self.db.execute("UPDATE triggers SET status='queued' WHERE hotkey=? AND status IN ('excluded','withdrawn')",
                            (hotkey,))
            return super().reconcile(hotkey, result, window=window, rejected=rejected)

    def recover_interrupted(self):
        # One immediate retry after restart for legacy timeout/cancellation rows.
        # Preserve attempt counts, coldkey turns, immutable bindings and results.
        with self.lock:
            rows = [dict(r) for r in self.db.execute(
                "SELECT id,retry_block,reason FROM triggers WHERE status='queued' AND retry_block>0 "
                "AND reason IN ('TimeoutError','InterruptedError')")]
            self.db.execute("UPDATE triggers SET retry_block=0,reason='retry_after_restart' "
                            "WHERE status='queued' AND retry_block>0 "
                            "AND reason IN ('TimeoutError','InterruptedError')")
            if self.current_block:
                cap = self.current_block() + 5
                rows.extend(dict(r) for r in self.db.execute(
                    "SELECT id,retry_block,reason FROM triggers WHERE status='queued' AND retry_block>? "
                    "AND COALESCE(reason,'') != 'DownloadBudgetExceeded'", (cap,)))
                self.db.execute("UPDATE triggers SET retry_block=? WHERE status='queued' AND retry_block>? "
                                "AND COALESCE(reason,'') != 'DownloadBudgetExceeded'", (cap, cap))
            return rows

    def pending(self, count=1, *, before_block=2**63-1, current_block=2**63-1):
        candidates = self.candidates()
        with self.lock:
            rows = self.db.execute('''SELECT t.*, c.turn FROM triggers t JOIN coldkey_turns c USING(coldkey)
                WHERE status IN ('queued','running','failed') AND COALESCE(readmission_block,block) < ?
                ORDER BY c.turn,COALESCE(t.readmission_block,t.block),t.hotkey''', (before_block,)).fetchall()
        groups = {}
        for row in rows:
            if row['reason'] == 'DownloadBudgetExceeded':
                continue  # Parked acquisition does not block an eligible sibling.
            candidate = candidates.get(row['model_id'])
            if candidate and candidate['hotkey'] == row['hotkey']:
                groups.setdefault(row['coldkey'], []).append(self._decode(row))
        # Only an eligible head can hold back its own coldkey in this window.
        groups = {key: rows for key, rows in groups.items() if rows[0]['retry_block'] <= current_block}
        result = []
        while groups and len(result) < count:
            for key in list(groups):
                result.append(groups[key].pop(0))
                if not groups[key]:
                    del groups[key]
                if len(result) == count:
                    break
        return result

    def defer(self, trigger_id, reason, retry_block=0):
        if reason == 'MinerUnavailable':
            with self.lock:
                row = self.db.execute('SELECT window_id FROM triggers WHERE id=?', (trigger_id,)).fetchone()
                if row and not self.interruption(row['window_id']):
                    self.db.execute("UPDATE triggers SET status='withdrawn',reason='miner_unavailable',"
                                    "removed_block=?,retry_block=0 WHERE id=? AND status='running'",
                                    (self.current_block() if self.current_block else 0, trigger_id))
                    return
        if reason == 'CompressionRequired':
            with self.lock:
                self.db.execute("UPDATE triggers SET status='excluded',reason='compression_required',"
                                "retry_block=? WHERE id=? AND status='running'",
                                ((self.current_block() if self.current_block else 0) + 25, trigger_id))
            return
        if reason == 'DownloadBudgetExceeded':
            # Durable local parking only; no terminal result or hotkey use.
            return super().defer(trigger_id, reason, 2**63 - 1)
        with self.lock:
            row = self.db.execute('SELECT window_id FROM triggers WHERE id=?', (trigger_id,)).fetchone()
        if row:
            planned = self.interruption(row['window_id'])
            if planned:
                reason, retry_block = planned, 0
        if self.current_block:
            # Total attempts also count successful partial turns, not just
            # failures. Do not turn one retryable error into an hour's wait.
            retry_block = min(retry_block, self.current_block() + 5)
        super().defer(trigger_id, reason, retry_block)


class ManagedEvaluator(Evaluator):
    """Run weight-cache housekeeping on the same worker between evaluations.

    Keep the scoring/replay source identity unchanged: cache deletion neither
    creates results nor decides which challenger wins.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cleanup_window = None
        self.cleanup_retry_at = 0.
        self.hold_windows = set()
        self.compression_check = None
        self.triggers.close()
        self.triggers = ScheduledTriggers(self.root / 'triggers.sqlite3',
            candidates=lambda: (self.ledger.active or {}).get('candidates', {}),
            interruption=self._interruption, current_block=lambda: self.ledger.cursor)
        recovered = self.triggers.recover_interrupted()
        if recovered:
            write_private(self.root / 'queue-recovery.json', {'unix': time.time(), 'triggers': recovered})

    @classmethod
    def from_config(cls, config, root, hotkey, ledger, keypair):
        holds = config.get('evaluation_hold_windows', [])
        if not isinstance(holds, list) or any(type(window) is not int or window < 0 for window in holds):
            raise ValueError('invalid_evaluation_hold_windows')
        worker = super().from_config(config, root, hotkey, ledger, keypair)
        worker.hold_windows = set(holds)
        from .contract import InfrastructureError
        from .p2p import ModelClient
        from .submission import parse_submission

        def check(entry, snapshot, cancelled, timeout_s):
            address = snapshot.get('endpoints', {}).get(entry['hotkey'])
            if address is None:
                raise InfrastructureError('miner_endpoint_unavailable')
            client = ModelClient(keypair, entry['hotkey'], address['host'], address['port'],
                                 parse_submission(entry['value']))
            client.cancelled = cancelled
            client.remaining_s = lambda: timeout_s
            client.require_compression()

        worker.compression_check = check
        return worker

    def update(self, snapshot, *, start_worker=True):
        # The consensus ledger retains first bindings, so reread fresh finalized
        # commitments separately for explicit readmission of the SAME model.
        with self.triggers.lock:
            row = self.triggers.db.execute('SELECT block FROM queue_cursor WHERE id=1').fetchone()
            cursor = row[0] if row else 0
            head = self.ledger.cursor
            rows = self.ledger.history(cursor + 1, head + 1)
            self.triggers.observe(rows, set(snapshot['uids']),
                                  coldkeys=snapshot['coldkeys'], uids=snapshot['uids'])
            self.triggers.db.execute('INSERT OR REPLACE INTO queue_cursor VALUES (1,?)', (head,))
        return super().update(snapshot, start_worker=start_worker)

    def _miner_io(self, callback, window, *, preflight=False):
        from http.client import HTTPException
        from ssl import SSLError
        from .download_budget import DownloadBudgetExceeded
        from .contract import InfrastructureError
        try:
            return callback()
        except Exception as error:
            remote = not isinstance(error, (CompressionRequired, DownloadBudgetExceeded)) and (
                      isinstance(error, (ConnectionError, TimeoutError, HTTPException, SSLError))
                      or isinstance(error, OSError) and str(error).startswith(('model_server_status_', 'incomplete_model_'))
                      or preflight and isinstance(error, (OSError, InterruptedError))
                      or preflight and isinstance(error, ValueError)
                      and str(error) in ('miner_tls_pin_mismatch', 'unsafe_model_endpoint')
                      or isinstance(error, InfrastructureError) and str(error) == 'miner_endpoint_unavailable')
            if remote and not self._cancelled(window):
                raise MinerUnavailable('miner_unavailable') from error
            raise

    def _evaluate(self, entry, window, snapshot, rows, selected, **kwargs):
        original = self.download
        # Attribute only challenger download failures, never king/judge/GPU failures.
        def guarded(*args):
            if window['king'] and entry['model_id'] == window['king']['model_id']:
                return original(*args)
            return self._miner_io(lambda: original(*args), window)
        self.download = guarded
        try:
            return super()._evaluate(entry, window, snapshot, rows, selected, **kwargs)
        finally:
            self.download = original

    def _attempt(self, entry, window, snapshot, partials):
        # Recheck the authoritative opening immediately before any miner I/O.
        # Queue growth or a stale/tampered dispatch must never expand this panel.
        with self.ledger.lock:
            opening = self.ledger.active
            candidate = (opening or {}).get('candidates', {}).get(entry['model_id'])
            if (not opening or opening['id'] != window['id']
                    or opening['start_block'] != window['start_block']
                    or not candidate or candidate['block'] >= opening['start_block']
                    or (entry.get('readmission_block') or entry['block']) >= opening['start_block']
                    or any(entry.get(key) != candidate.get(key)
                           for key in ('hotkey', 'model_id', 'value', 'block', 'uid'))):
                raise InterruptedError('not_admitted_to_window')
        if self.compression_check is None:
            raise RuntimeError('compression_checker_required')
        self._miner_io(lambda: self.compression_check(
            entry, snapshot, lambda: self._cancelled(window), 30.), window, preflight=True)
        return super()._attempt(entry, window, snapshot, partials)

    def _recover_compression(self, window, snapshot):
        if self.compression_check is None:
            return
        rows = self.triggers.excluded()
        if not rows or rows[0]['retry_block'] > snapshot['block']:
            return
        row = rows[0]  # At most one five-second check per worker pass.
        ready = False
        try:
            self.compression_check(row, snapshot,
                lambda: self.stopping.is_set() or (self.ledger.active or {}).get('id') != window['id'], 5.)
            ready = True
        except Exception:
            pass  # Still excluded; no score, result commitment or hotkey consumption.
        self.triggers.compression_retry(row['id'], ready=ready, block=snapshot['block'])

    def _interruption(self, window):
        with self.lock:
            context = self.context
            if self.stopping.is_set():
                return 'validator_stopping'
            if context and context[0] and context[0]['id'] != window:
                return 'window_changed'
            if context and self.attempt_epoch is not None and context[1]['epoch_index'] != self.attempt_epoch:
                return 'epoch_changed'
            if self.deadline is not None and time.monotonic() >= self.deadline:
                return 'attempt_budget_elapsed'
        return None

    def _retired_models(self):
        """Cache eviction candidates from closed decisions, not local scores."""
        with self.ledger.lock:
            models = {json.loads(r[0])['model_id'] for r in self.ledger.db.execute(
                'SELECT u.result FROM uses u JOIN windows w ON u.window=w.id '
                'WHERE u.evaluator=? AND w.decision IS NOT NULL', (self.hotkey,))}
            # A king may have been downloaded only as a baseline by this validator.
            for row in self.ledger.db.execute('SELECT opening FROM windows WHERE decision IS NOT NULL'):
                king = json.loads(row[0]).get('king')
                if king:
                    models.add(king['model_id'])
            for king in (self.ledger.get('king'), (self.ledger.active or {}).get('king')):
                if king:
                    models.discard(king['model_id'])
            return models

    def _cleanup_models(self, window):
        """Run only on the evaluator worker, between attempts and after consensus."""
        if self.cleanup_window == window['id'] and time.monotonic() < self.cleanup_retry_at:
            return
        from .model_cache import remove_models
        status = {'window': window['id'], 'deleted_downloads': [], 'deleted_gpu': []}
        try:
            with self.lock, self.ledger.lock:
                active = self.ledger.active
                if not active or active['id'] != window['id']:
                    return
                models = self._retired_models()
                # Keep any result still being published, and all retryable challengers.
                models.difference_update(Result.parse(item['value']).model_id for item in self.outbox)
                models.difference_update(row['model_id'] for row in self.triggers.rows()
                                         if row['usage'] != 'consumed')
                for view in (window, self.context[0] if self.context else None):
                    if view and view.get('king'):
                        models.discard(view['king']['model_id'])
            if models:
                # Downloads and runner copies have one worker. The GPU lease also
                # protects cooperating local jobs; no chain-thread filesystem work.
                with self.gpu.lease() if self.gpu else nullcontext():
                    # Taking the lease can wait on a hardware probe. If consensus
                    # advanced meanwhile, recompute retention on the next pass.
                    if (self.ledger.active or {}).get('id') != window['id']:
                        return
                    status['deleted_downloads'] = remove_models(self.root / 'models', sorted(models))
                    if self.gpu:
                        status['deleted_gpu'] = self.gpu.remove_models(sorted(models))
            status['status'] = 'complete'
        except Exception as error:
            # A failed cleanup must not invalidate a score or consume a hotkey.
            status.update(status='retry', error=type(error).__name__)
        self.cleanup_window = window['id']
        self.cleanup_retry_at = time.monotonic() + 60.
        write_private(self.root / 'model-cache-cleanup.json', {**status, 'unix': time.time()})

    def run_once(self, window, snapshot):
        if window['id'] in self.hold_windows:
            self.progress('waiting', window)
            return  # Chain replay/dashboard/weights continue; no new scores for this window.
        self._recover_compression(window, snapshot)
        self._cleanup_models(window)
        return super().run_once(window, snapshot)
