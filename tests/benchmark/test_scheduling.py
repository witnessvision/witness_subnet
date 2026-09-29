"""Dispatch regressions: planned cancellations, window rotation and coldkey FIFO."""
import time

import pytest

from witness.benchmark.managed_evaluator import ScheduledTriggers
from test_model_cache import evaluator_at
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
