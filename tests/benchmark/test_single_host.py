import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from witness.storage import write_private
from witness.budget import DailyBudget, BudgetUnavailable
from witness.benchmark.execution import run_process
from witness.benchmark.gpu import LocalGpu
from witness.benchmark.protocol import weight_vector, weights_match
from witness.benchmark.validator import submit_weights


def test_archive_mirror_cannot_change_source_or_leave_archive():
    from witness.benchmark.pool import source_url
    video = {'identifier': 'source-1', 'file': 'clip one.mp4'}
    assert source_url(video) == 'https://archive.org/download/source-1/clip%20one.mp4'
    good = 'https://dn1.ca.archive.org/0/items/source-1/clip%20one.mp4'
    assert source_url(video, {'source-1': good}) == good
    for bad in ('https://evil.example/0/items/source-1/clip%20one.mp4',
                good.replace('source-1', 'other'), good.replace('https:', 'http:'),
                good + '?redirect=elsewhere', good.replace('dn1.', 'user:password@dn1.'),
                good.replace('.org/', '.org:8443/')):
        with pytest.raises(ValueError):
            source_url(video, {'source-1': bad})


def test_weights_use_finalized_chain_required_version():
    calls = []
    def query(module, name, params, **kwargs):
        assert (module, name, params, kwargs) == (
            'SubtensorModule', 'WeightsVersionKey', [20], {'block_hash': 'finalized'})
        return SimpleNamespace(value=1022)
    def send(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=True, error=None, extrinsic_receipt=SimpleNamespace(
            extrinsic_hash='tx', block_hash='included'))
    chain = SimpleNamespace(subtensor=SimpleNamespace(set_weights=send, substrate=SimpleNamespace(
        get_chain_finalised_head=lambda: 'finalized', query=query)))
    assert submit_weights(chain, 'wallet', 20, [240], [1.])['success']
    assert calls == [dict(wallet='wallet', netuid=20, uids=[240], weights=[1.], version_key=1022,
                          wait_for_inclusion=True, wait_for_finalization=True)]


def failed_load_worker(job, pipe):
    pipe.send({'loading': True, 'hardware_id': 'same-gpu'})
    pipe.send({'exit': 2, 'error': 'model_load_oom'})


def stalled_load_worker(job, pipe):
    pipe.send({'loading': True, 'hardware_id': 'same-gpu'})
    time.sleep(30)


@pytest.mark.parametrize('worker', [failed_load_worker, stalled_load_worker])
def test_rejected_model_preserves_gpu_identity_and_cleans_worker(tmp_path, worker):
    import multiprocessing
    from witness.benchmark.pod_runtime import supervise
    before = {p.pid for p in multiprocessing.active_children()}
    tasks = [{'task_id': str(i), 'deadline_s': 60.} for i in range(2)]
    output = tmp_path / 'rejected.jsonl'
    assert supervise({'tasks': tasks, 'load_timeout_s': .1}, output, worker=worker,
                     context=multiprocessing.get_context('fork')) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2 and all(r['status'] == 'invalid' and r['hardware_id'] == 'same-gpu' for r in rows)
    assert not ({p.pid for p in multiprocessing.active_children()} - before)


def test_burn_allocation_and_real_u16_encoding():
    snapshot = {'uids': {'king': 170, 'burn': 240}, 'burn_uid': 240}
    burn = weight_vector(None, snapshot)
    assert burn['uids'] == [240] and burn['weights'] == [1.]
    mixed = weight_vector({'hotkey': 'king'}, snapshot)
    assert mixed['uids'] == [170, 240] and mixed['weights'] == pytest.approx([.3, .7])
    assert weights_match([(240, 65535), (170, 28086)], mixed)
    assert not weights_match([(240, 65535)], mixed)
    assert not weights_match([(170, 65535), (240, 28086)], mixed)
    assert not weights_match([(170, 28086), (240, 65535), (99, 1)], mixed)
    assert weight_vector({'hotkey': 'departed'}, snapshot) == burn
    assert weight_vector({'hotkey': 'king'}, {**snapshot, 'burn_uid': None})['uids'] == []
    collision = weight_vector({'hotkey': 'burn'}, snapshot)
    assert collision['uids'] == [240] and collision['weights'] == [1.]


def test_unlimited_api_accounting_and_operator_limit_are_separate(tmp_path):
    path = tmp_path / 'spend.sqlite3'
    budget = DailyBudget(path, daily_limit_usd=None)
    claim = budget.reserve(role='validator', provider='openai', input_hash='x', upper_usd=100.)
    budget.settle(claim, 50.)
    assert DailyBudget(path, daily_limit_usd=None).totals()['validator']['settled_usd'] == 50.
    with pytest.raises(BudgetUnavailable, match='policy_changed'):
        DailyBudget(path, daily_limit_usd=9.)




def test_preprocessing_deadline_and_gpu_lock_do_not_kill_foreign_jobs(tmp_path, monkeypatch):
    started = time.monotonic()
    with pytest.raises(InterruptedError):
        run_process([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=30, remaining_s=lambda: .05)
    assert time.monotonic()-started < 2
    gpu = LocalGpu({'workspace': str(tmp_path)}, tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 0, '', ''))
    with gpu.lease():
        with pytest.raises(RuntimeError, match='gpu_busy'):
            with gpu.lease():
                pass
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 0, '9876\n', ''))
    with pytest.raises(RuntimeError, match='gpu_busy'):
        with gpu.lease():
            pass
