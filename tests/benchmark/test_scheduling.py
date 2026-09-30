"""Dispatch regressions: planned cancellations, window rotation and coldkey FIFO."""
import time

import pytest

from witness.benchmark.managed_evaluator import ScheduledTriggers
from test_model_cache import evaluator_at, advance
from test_validator import MODELS, MINERS, entry, snapshot


def queue(tmp_path, candidates, interruption=lambda window: None):
    triggers = ScheduledTriggers(tmp_path/'triggers.sqlite3', candidates=lambda: candidates,
                                 interruption=interruption)
    rows = [entry(i) for i in range(3)]
    triggers.observe(rows, set(MINERS[:3]),
                     coldkeys={MINERS[0]: 'first', MINERS[1]: 'shared', MINERS[2]: 'shared'},
                     uids={key: i for i, key in enumerate(MINERS[:3])})
    return triggers


def test_ineligible_head_does_not_block_eligible_sibling(tmp_path):
    active = {MODELS[i]: entry(i) for i in (0, 2)}
    triggers = queue(tmp_path, active)
    head = next(row for row in triggers.rows() if row['model_id'] == MODELS[1])
    triggers.start(head['id'], 6)
    triggers.defer(head['id'], 'TimeoutError', 500)
    assert [row['model_id'] for row in triggers.pending(10, current_block=100)] == [MODELS[0], MODELS[2]]
    # The original FIFO/backoff is preserved if both siblings are eligible.
    active[MODELS[1]] = entry(1)
    assert [row['model_id'] for row in triggers.pending(10, current_block=100)] == [MODELS[0]]
    assert len(triggers.pending(10, current_block=500)) == 3


def test_window_filter_keeps_binding_cutoff_and_fair_coldkey_rotation(tmp_path):
    active = {MODELS[i]: entry(i) for i in range(3)}
    triggers = queue(tmp_path, active)
    assert not triggers.pending(10, before_block=1)
    first = triggers.pending(1)[0]
    triggers.start(first['id'], 2)
    triggers.defer(first['id'], 'attempt_budget_elapsed', 0)
    assert [row['model_id'] for row in triggers.pending(3)] == [MODELS[1], MODELS[0], MODELS[2]]
    active[MODELS[1]] = {**entry(1), 'hotkey': 'wrong-binding'}
    assert MODELS[1] not in {row['model_id'] for row in triggers.pending(10)}


def test_panel_stays_frozen_while_queue_grows_and_after_restart(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    opening = ledger.active
    advance(worker, ledger, 10, {9: [entry(3, block=9)], 10: [entry(4, block=10)]})
    assert ledger.active == opening
    assert {MODELS[3], MODELS[4]} <= {r['model_id'] for r in worker.triggers.rows()}
    eligible = lambda: {r['model_id'] for r in worker.triggers.pending(
        100, before_block=ledger.active['start_block'], current_block=ledger.cursor)}
    assert not {MODELS[3], MODELS[4]} & eligible()
    worker.triggers.close()
    worker.triggers = ScheduledTriggers(tmp_path/'triggers.sqlite3',
        candidates=lambda: ledger.active['candidates'], interruption=lambda window: None)
    assert not {MODELS[3], MODELS[4]} & eligible()
    advance(worker, ledger, 12)
    assert {MODELS[3], MODELS[4]} <= eligible()


@pytest.mark.parametrize('field,value', [
    ('model_id', MODELS[3]), ('hotkey', MINERS[3]), ('value', entry(3)['value']),
    ('block', 8), ('uid', 3),
])
def test_dispatch_guard_blocks_unfrozen_or_changed_entry_without_io_or_consumption(
        tmp_path, monkeypatch, field, value):
    from witness.benchmark.managed_evaluator import ManagedEvaluator
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    window = ledger.active
    challenger = {**window['candidates'][MODELS[2]], field: value}
    before = ledger.usage(worker.hotkey)
    def forbidden(*args):
        pytest.fail('unadmitted model must not reach transport, batch or inference')
    worker.compression_check = worker.batch_factory = worker.download = worker.runner = forbidden
    with pytest.raises(InterruptedError, match='not_admitted_to_window'):
        ManagedEvaluator._attempt(worker, challenger, window, snapshot(8), set())
    assert ledger.usage(worker.hotkey) == before and not worker.outbox


def test_dispatch_guard_uses_ledger_panel_not_modified_dispatch_copy(tmp_path, monkeypatch):
    from witness.benchmark.managed_evaluator import ManagedEvaluator
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    window = ledger.active
    forged = entry(3, block=7)
    window['candidates'][MODELS[3]] = forged
    worker.compression_check = lambda *args: pytest.fail('forged panel reached miner')
    with pytest.raises(InterruptedError, match='not_admitted_to_window'):
        ManagedEvaluator._attempt(worker, forged, window, snapshot(8), set())


def test_admitted_dispatch_keeps_compression_and_evaluation_path(tmp_path, monkeypatch):
    from witness.benchmark.managed_evaluator import ManagedEvaluator
    from witness.benchmark.evaluator import Evaluator
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    calls = []
    worker.compression_check = lambda *args: calls.append('compression')
    monkeypatch.setattr(Evaluator, '_attempt', lambda *args: calls.append('evaluation'))
    window = ledger.active
    ManagedEvaluator._attempt(worker, window['candidates'][MODELS[2]], window, snapshot(8), set())
    assert calls == ['compression', 'evaluation']


def test_unavailable_miner_requires_fresh_same_model_commitment_and_next_window(tmp_path):
    active = {MODELS[i]: entry(i) for i in range(3)}
    triggers = queue(tmp_path, active)
    triggers.current_block = lambda: 100
    row = triggers.pending()[0]
    triggers.start(row['id'], 6)
    triggers.defer(row['id'], 'MinerUnavailable')
    assert row['id'] not in {r['id'] for r in triggers.rows()}
    assert row['id'] not in {r['id'] for r in triggers.pending(100)}
    saved = dict(triggers.db.execute('SELECT * FROM triggers WHERE id=?', (row['id'],)).fetchone())
    assert saved['result'] is None and saved['finished_unix'] is None
    triggers.close()
    triggers = ScheduledTriggers(tmp_path/'triggers.sqlite3', candidates=lambda: active,
                                 interruption=lambda window: None)
    def submit(value, block):
        triggers.observe([{**entry(0), 'value': value, 'block': block}], {MINERS[0]},
                         coldkeys={MINERS[0]: 'first'}, uids={MINERS[0]: 0})
    submit(entry(0)['value'], 1)  # Old chain history cannot requeue.
    submit(entry(1)['value'], 101)  # Binding cannot change.
    assert row['id'] not in {r['id'] for r in triggers.rows()}
    submit(entry(0)['value'], 102)
    assert row['id'] not in {r['id'] for r in triggers.pending(100, before_block=102)}
    revived = next(r for r in triggers.pending(100, before_block=103) if r['id'] == row['id'])
    assert revived['usage'] == 'reserved' and revived['readmission_block'] == 102
    assert all(revived[k] == row[k] for k in ('hotkey','model_id','value','block'))


def test_chain_replay_reactivates_withdrawn_binding_only_after_new_commit(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    row = worker.triggers.pending()[0]
    worker.triggers.start(row['id'], ledger.active['id'])
    worker.triggers.defer(row['id'], 'MinerUnavailable')
    worker.update(snapshot(8), start_worker=False)
    assert row['id'] not in {r['id'] for r in worker.triggers.rows()}
    advance(worker, ledger, 9, {9: [entry(2, block=9)]})
    assert next(r for r in worker.triggers.rows() if r['id'] == row['id'])['readmission_block'] == 9
    assert not worker.triggers.pending(100, before_block=ledger.active['start_block'])
    advance(worker, ledger, 12)
    assert row['id'] in {r['id'] for r in worker.triggers.pending(100, before_block=ledger.active['start_block'])}


@pytest.mark.parametrize('error,removed', [
    (ConnectionRefusedError(), True), (TimeoutError(), True),
    (RuntimeError('judge unavailable'), False), (OSError('insufficient_model_cache_space'), False),
])
def test_only_miner_transport_errors_withdraw_challenger(tmp_path, monkeypatch, error, removed):
    from witness.benchmark.managed_evaluator import MinerUnavailable
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    def fail():
        raise error
    before = ledger.usage(worker.hotkey)
    with pytest.raises(MinerUnavailable if removed else type(error)):
        worker._miner_io(fail, ledger.active)
    assert ledger.usage(worker.hotkey) == before and not worker.outbox


def test_planned_cancellation_preserves_miner_reservation(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    worker.stopping.set()
    def fail():
        raise ConnectionRefusedError()
    with pytest.raises(ConnectionRefusedError):
        worker._miner_io(fail, ledger.active, preflight=True)


def test_readmitted_model_cannot_use_old_panel_in_current_window(tmp_path, monkeypatch):
    from witness.benchmark.managed_evaluator import ManagedEvaluator
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    window = ledger.active
    entry = {**window['candidates'][MODELS[2]], 'readmission_block': 9}
    worker.compression_check = lambda *args: pytest.fail('readmission bypassed frozen cutoff')
    with pytest.raises(InterruptedError, match='not_admitted_to_window'):
        ManagedEvaluator._attempt(worker, entry, window, snapshot(9), set())


@pytest.mark.parametrize('baseline', [True, False])
def test_download_failure_attributed_to_challenger_not_king(tmp_path, monkeypatch, baseline):
    from witness.benchmark.managed_evaluator import ManagedEvaluator, MinerUnavailable
    from witness.benchmark.evaluator import Evaluator
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    def download(*args):
        raise ConnectionRefusedError()
    worker.download = download
    monkeypatch.setattr(Evaluator, '_evaluate', lambda self, *a, **k: self.download())
    window = ledger.active
    entry = window['king'] if baseline else window['candidates'][MODELS[2]]
    with pytest.raises(ConnectionRefusedError if baseline else MinerUnavailable):
        ManagedEvaluator._evaluate(worker, entry, window, snapshot(8), [], [])
    assert worker.download is download and not worker.outbox


@pytest.mark.parametrize('kind', ['window', 'epoch', 'budget', 'stop'])
@pytest.mark.parametrize('exception', [InterruptedError, TimeoutError])
def test_expected_cancellation_does_not_back_off_or_consume(tmp_path, monkeypatch, kind, exception):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    old_window = ledger.active
    def interrupted(*args):
        if kind == 'window':
            worker.context = ({**old_window, 'id': old_window['id'] + 1}, snapshot(8))
        elif kind == 'epoch':
            worker.context = (old_window, snapshot(8, epoch_index=worker.attempt_epoch + 1))
        elif kind == 'budget':
            worker.deadline = time.monotonic() - 1
        else:
            worker.stopping.set()
        raise exception('planned stop')
    monkeypatch.setattr(worker, '_attempt', interrupted)
    before = ledger.usage(worker.hotkey)
    with pytest.raises(exception):
        worker.run_once(old_window, snapshot(8))
    row = next(row for row in worker.triggers.rows() if row['model_id'] == MODELS[2])
    expected = {'window': 'window_changed', 'epoch': 'epoch_changed',
                'budget': 'attempt_budget_elapsed', 'stop': 'validator_stopping'}[kind]
    assert row['retry_block'] == 0 and row['reason'] == expected
    assert row['attempts'] == 1 and row['usage'] == 'reserved'
    assert ledger.usage(worker.hotkey) == before
    assert not worker.outbox


def test_real_network_timeout_keeps_backoff(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    def timeout(*args):
        assert worker.deadline > time.monotonic()
        raise TimeoutError('socket timeout')
    monkeypatch.setattr(worker, '_attempt', timeout)
    with pytest.raises(TimeoutError):
        worker.run_once(ledger.active, snapshot(8))
    row = next(row for row in worker.triggers.rows() if row['model_id'] == MODELS[2])
    assert 8 < row['retry_block'] <= ledger.cursor + 5 and row['reason'] == 'TimeoutError'
    assert row['usage'] == 'reserved'


def test_restart_recovers_only_queued_timeout_delays(tmp_path):
    active = {MODELS[i]: entry(i) for i in range(3)}
    triggers = queue(tmp_path, active)
    for row, reason in zip(triggers.rows(), ('TimeoutError', 'InterruptedError', 'OSError')):
        triggers.start(row['id'], 6)
        triggers.defer(row['id'], reason, 500)
    before = triggers.rows()
    turns = list(triggers.db.execute('SELECT * FROM coldkey_turns ORDER BY coldkey'))
    recovered = triggers.recover_interrupted()
    after = triggers.rows()
    assert len(recovered) == 2 and not triggers.recover_interrupted()
    assert [r['retry_block'] for r in after] == [0, 0, 500]
    assert [r['attempts'] for r in before] == [r['attempts'] for r in after]
    assert list(triggers.db.execute('SELECT * FROM coldkey_turns ORDER BY coldkey')) == turns
    assert all(r['usage'] == 'reserved' for r in after)


def test_exhausted_download_does_not_block_sibling_or_consume_hotkey(tmp_path):
    active = {MODELS[i]: entry(i) for i in range(3)}
    triggers = queue(tmp_path, active)
    head = next(row for row in triggers.rows() if row['model_id'] == MODELS[1])
    triggers.start(head['id'], 6)
    triggers.defer(head['id'], 'DownloadBudgetExceeded', 500)
    triggers.recover_interrupted()
    assert [r['model_id'] for r in triggers.pending(10)] == [MODELS[0], MODELS[2]]
    parked = next(r for r in triggers.rows() if r['id'] == head['id'])
    assert parked['status'] == 'queued' and parked['usage'] == 'reserved'
    assert parked['retry_block'] == 2**63 - 1


def test_exhausted_download_never_creates_zero_score_or_outbox(tmp_path, monkeypatch):
    from witness.benchmark.download_budget import DownloadBudgetExceeded
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    def exhausted(*args):
        raise DownloadBudgetExceeded('model_download_time_budget_exhausted')
    monkeypatch.setattr(worker, '_attempt', exhausted)
    before = ledger.usage(worker.hotkey)
    with pytest.raises(DownloadBudgetExceeded):
        worker.run_once(ledger.active, snapshot(8))
    row = next(r for r in worker.triggers.rows() if r['model_id'] == MODELS[2])
    assert row['status'] == 'queued' and row['usage'] == 'reserved'
    assert ledger.usage(worker.hotkey) == before and not worker.outbox


@pytest.mark.parametrize('reason', ['TimeoutError', 'OSError', 'RemoteDisconnected', 'InfrastructureError'])
def test_many_normal_turns_do_not_magnify_a_retryable_failure(tmp_path, reason):
    active = {MODELS[i]: entry(i) for i in range(3)}
    triggers = queue(tmp_path, active)
    triggers.current_block = lambda: 1000
    head = triggers.pending(1)[0]
    for _ in range(8):
        triggers.start(head['id'], 7)
        triggers.defer(head['id'], 'attempt_budget_elapsed', 0)
    triggers.start(head['id'], 7)
    triggers.defer(head['id'], reason, 1300)
    row = next(r for r in triggers.rows() if r['id'] == head['id'])
    assert row['attempts'] == 9 and row['retry_block'] == 1005
    assert row['usage'] == 'reserved'


def test_restart_caps_legacy_io_backoff_but_preserves_exhausted_budget(tmp_path):
    active = {MODELS[i]: entry(i) for i in range(3)}
    triggers = queue(tmp_path, active)
    rows = triggers.rows()
    for row, reason in zip(rows[:2], ('OSError', 'DownloadBudgetExceeded')):
        triggers.start(row['id'], 7)
        triggers.defer(row['id'], reason, 1500)
    triggers.current_block = lambda: 1000
    recovered = triggers.recover_interrupted()
    after = triggers.rows()
    assert len(recovered) == 1 and after[0]['retry_block'] == 1005
    assert after[1]['retry_block'] == 2**63 - 1
    assert all(row['usage'] == 'reserved' for row in after)


def test_compression_exclusion_survives_restart_and_preserves_first_binding(tmp_path):
    active = {MODELS[i]: entry(i) for i in range(3)}
    triggers = queue(tmp_path, active)
    triggers.current_block = lambda: 100
    head = next(r for r in triggers.rows() if r['model_id'] == MODELS[1])
    triggers.start(head['id'], 6)
    triggers.defer(head['id'], 'CompressionRequired')
    assert MODELS[1] not in {r['model_id'] for r in triggers.pending(10)}
    assert MODELS[2] in {r['model_id'] for r in triggers.pending(10)}
    assert MODELS[1] not in {r['model_id'] for r in triggers.rows()}
    excluded = triggers.excluded()[0]
    assert excluded['usage'] == 'reserved' and excluded['result'] is None
    assert excluded['finished_unix'] is None and excluded['retry_block'] == 125
    triggers.close()
    triggers = ScheduledTriggers(tmp_path/'triggers.sqlite3', candidates=lambda: active,
                                 interruption=lambda window: None, current_block=lambda: 101)
    triggers.recover_interrupted()
    spam = {**entry(1), 'block': 101, 'value': entry(2)['value']}
    assert not triggers.observe([spam], {MINERS[1]}, coldkeys={MINERS[1]: 'shared'}, uids={MINERS[1]: 1})
    assert len(triggers.excluded()) == 1
    triggers.compression_retry(head['id'], ready=True, block=125)
    revived = next(r for r in triggers.rows() if r['id'] == head['id'])
    assert all(revived[key] == head[key] for key in ('hotkey', 'model_id', 'block', 'value'))
    assert revived['reason'] == 'compression_ready' and revived['retry_block'] == 0
    assert revived['status'] == 'queued' and revived['usage'] == 'reserved'
    assert not triggers.excluded()


def test_compression_preflight_blocks_cached_challenger_before_batch_or_score(tmp_path, monkeypatch):
    from witness.benchmark.managed_evaluator import ManagedEvaluator
    from witness.benchmark.compression import CompressionRequired
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    monkeypatch.setattr(worker, '_attempt', ManagedEvaluator._attempt.__get__(worker))
    calls = []
    def check(entry, snapshot, cancelled, timeout_s):
        calls.append(entry['model_id'])
        raise CompressionRequired('zstandard_required')
    worker.compression_check = check
    def forbidden(*args):
        pytest.fail('compression preflight must precede batch, download, inference and score')
    worker.batch_factory = worker.download = worker.runner = forbidden
    before = ledger.usage(worker.hotkey)
    with pytest.raises(CompressionRequired):
        worker.run_once(ledger.active, snapshot(8))
    assert calls == [MODELS[2]]
    assert ledger.usage(worker.hotkey) == before and not worker.outbox
    row = worker.triggers.excluded()[0]
    assert row['model_id'] == MODELS[2] and row['usage'] == 'reserved'
    assert not list(tmp_path.glob('windows/*/models/*/score.json'))
    assert not worker.triggers.pending(10)
    # Recovery does not run early, then requeues exactly the same entry.
    worker._recover_compression(ledger.active, snapshot(32))
    assert len(calls) == 1
    worker.compression_check = lambda *args: None
    worker._recover_compression(ledger.active, snapshot(33))
    assert not worker.triggers.excluded()
    assert worker.triggers.pending()[0]['id'] == row['id']
    assert ledger.usage(worker.hotkey) == before and not worker.outbox


def test_failed_compression_recovery_is_bounded_and_never_scores(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    row = worker.triggers.pending()[0]
    worker.triggers.start(row['id'], ledger.active['id'])
    worker.triggers.defer(row['id'], 'CompressionRequired')
    calls = []
    def unavailable(entry, snapshot, cancelled, timeout_s):
        calls.append(timeout_s)
        raise TimeoutError('unreachable')
    worker.compression_check = unavailable
    worker._recover_compression(ledger.active, snapshot(33))
    excluded = worker.triggers.excluded()[0]
    assert calls == [5.] and excluded['retry_block'] == 58
    assert excluded['result'] is None and excluded['usage'] == 'reserved' and not worker.outbox
    worker._recover_compression(ledger.active, snapshot(34))
    assert calls == [5.]


def test_miner_download_limit_withdraws_without_hotkey_use_and_preserves_sibling(tmp_path, monkeypatch):
    from witness.benchmark.managed_evaluator import MinerDownloadExpired
    from witness.benchmark.download_budget import DownloadBudgetExceeded
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    window = ledger.active
    before = ledger.usage(worker.hotkey)
    original = worker._evaluate
    def acquisition(*args):
        raise DownloadBudgetExceeded('model_download_time_budget_exhausted')
    monkeypatch.setattr(worker, 'download', acquisition)
    candidate = worker.triggers.pending(10)[0]
    worker.triggers.start(candidate['id'], window['id'])
    with pytest.raises(MinerDownloadExpired):
        worker._miner_io(lambda: acquisition(), window)
    worker.triggers.defer(candidate['id'], 'MinerDownloadExpired')
    saved = dict(worker.triggers.db.execute('SELECT * FROM triggers WHERE id=?', (candidate['id'],)).fetchone())
    assert saved['status'] == 'withdrawn' and saved['reason'] == 'download_time_limit'
    assert saved['result'] is None and saved['finished_unix'] is None
    assert ledger.usage(worker.hotkey) == before and worker.outbox == []
    assert candidate['id'] not in {r['id'] for r in worker.triggers.pending(100)}
    ledger.close()


def test_round_budget_parks_only_same_window_survives_restart_and_releases_sibling(tmp_path):
    active = {MODELS[i]: entry(i) for i in range(3)}
    current = [6]
    interruption = lambda w: 'window_changed' if w != current[0] else None
    triggers = queue(tmp_path, active, interruption)
    triggers.current_block = lambda: 100
    head = next(r for r in triggers.rows() if r['model_id'] == MODELS[1])
    triggers.start(head['id'], 6)
    triggers.defer(head['id'], 'EvaluationRoundBudgetExceeded')
    triggers.recover_interrupted()
    assert MODELS[1] not in {r['model_id'] for r in triggers.pending(10)}
    assert MODELS[2] in {r['model_id'] for r in triggers.pending(10)}
    triggers.close()
    triggers = ScheduledTriggers(tmp_path/'triggers.sqlite3', candidates=lambda:active,
                                 interruption=interruption, current_block=lambda:100)
    triggers.recover_interrupted()
    assert MODELS[1] not in {r['model_id'] for r in triggers.pending(10)}
    current[0] = 7
    assert MODELS[1] in {r['model_id'] for r in triggers.pending(10)}
    row = next(r for r in triggers.rows() if r['id'] == head['id'])
    assert row['result'] is None and row['usage'] == 'reserved'
