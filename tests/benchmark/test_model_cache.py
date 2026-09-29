"""Real cache deletion driven by finalized decisions; no scores or reports lost."""
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading

import pytest

from witness.benchmark.managed_evaluator import ManagedEvaluator as Evaluator
from witness.benchmark.gpu import LocalGpu, RunPodGpu, SshGpu
from witness.benchmark.ledger import Ledger
from witness.benchmark.model_cache import remove_models
from witness.benchmark.protocol import BASELINE, CONTROLS_OK
from witness.benchmark.contract import InfrastructureError
from test_validator import MINERS, MODELS, POLICY, VALS, entry, result, snapshot


@pytest.mark.parametrize('backend', ['local', 'ssh'])
def test_decoder_host_resource_failure_is_not_a_model_zero(tmp_path, monkeypatch, backend):
    error = ('video_reader_backend torchvision error: [swscaler] Failed initializing scaling graph '
             '(Resource temporarily unavailable)')
    fake = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, '', error)
    if backend == 'local':
        from witness.benchmark import execution
        monkeypatch.setattr(execution, 'run_process', fake)
        gpu = LocalGpu({'workspace': str(tmp_path / 'gpu')}, tmp_path)
    else:
        from witness.benchmark import gpu as module
        monkeypatch.setattr(module.subprocess, 'run', fake)
        gpu = SshGpu({'host': 'test.invalid', 'ssh_key': str(tmp_path / 'key')}, tmp_path)
    with pytest.raises(InfrastructureError, match='gpu_video_decoder_resource_exhausted'):
        gpu.run(['timeout', '600', 'python', '/workspace/pod_runtime.py', 'spec.json'], timeout=610)
    assert gpu.run(['python', 'unrelated_script.py'], timeout=1).returncode == 0


def test_ordinary_model_failure_keeps_its_existing_classification(tmp_path, monkeypatch):
    from witness.benchmark import execution
    monkeypatch.setattr(execution, 'run_process', lambda *a, **k:
                        subprocess.CompletedProcess(a, 2, '', 'ValueError: incompatible model weights'))
    gpu = LocalGpu({'workspace': str(tmp_path / 'gpu')}, tmp_path)
    assert gpu.run(['python', '/workspace/pod_runtime.py'], timeout=10).returncode == 2


def test_held_window_never_completes_a_faulty_bootstrap_pair_and_expires(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    worker.hold_windows = {ledger.active['id']}
    calls = []
    monkeypatch.setattr(worker, '_attempt', lambda *args: calls.append(args))
    before = worker.triggers.rows()
    worker.run_once(ledger.active, snapshot(8))
    assert worker.triggers.rows() == before and not calls and not worker.outbox
    advance(worker, ledger, 12)
    worker.run_once(ledger.active, snapshot(12))
    assert len(calls) == 1


def cached(root, model, *, partial=False):
    directory = root / 'models' / model
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('weights.safetensors.part' if partial else 'weights.safetensors')
    path.write_bytes(b'cached model fixture')
    return path


def evaluator_at(tmp_path, monkeypatch, *, block=6):
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    for height in range(1, block + 1):
        events = [entry(i) for i in range(3)] if height == 1 else []
        if height in (5, 6):
            index = height - 5
            events = [result(0, index, 20000 + index * 20000, 15000 + index * 20000,
                             block=height, kq=0, kr=0)]
        ledger.ingest(snapshot(height), events)
    gpu = LocalGpu({'workspace': str(tmp_path/'gpu')}, tmp_path)
    monkeypatch.setattr(gpu, 'lease', nullcontext)  # No GPU required for filesystem tests.
    worker = Evaluator(root=tmp_path, hotkey=VALS[0], ledger=ledger, gpu=gpu,
                       policy=None, judge=None, batch_factory=None, download=None, runner=None)
    monkeypatch.setattr(worker, '_attempt', lambda *args: None)
    worker.update(snapshot(block), start_worker=False)
    return worker, ledger, gpu


def advance(worker, ledger, block, events=None):
    for height in range(ledger.cursor + 1, block + 1):
        ledger.ingest(snapshot(height), (events or {}).get(height, []))
    worker.update(snapshot(block), start_worker=False)


def test_close_removes_loser_both_copies_and_keeps_king_retries_and_evidence(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch)
    paths = {i: [cached(root, MODELS[i], partial=i == 2) for root in (tmp_path, Path(gpu.workspace))]
             for i in range(3)}
    unknown = cached(tmp_path, 'f' * 64)
    evidence = [tmp_path/'reports/result.json', tmp_path/'windows/1/models'/MODELS[0]/'score.json',
                tmp_path/'windows/1/clips/clip.mp4', tmp_path/'content-owners.json']
    for path in evidence:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preserved evidence')
    usage = ledger.usage(VALS[0])
    worker.run_once(ledger.active, snapshot(6))
    assert all(path.exists() for copies in paths.values() for path in copies)
    advance(worker, ledger, 8)
    worker.run_once(ledger.active, snapshot(8))
    assert ledger.get('king')['model_id'] == MODELS[1]
    assert not any(path.exists() for path in paths[0])
    assert all(path.exists() for i in (1, 2) for path in paths[i])
    assert unknown.exists() and all(path.read_bytes() == b'preserved evidence' for path in evidence)
    assert ledger.usage(VALS[0]) == usage
    assert ledger.history(5, 7)
    status = json.loads((tmp_path/'model-cache-cleanup.json').read_text())
    assert status['status'] == 'complete'
    assert status['deleted_downloads'] == status['deleted_gpu'] == [MODELS[0]]


def test_new_king_kept_and_replaced_king_removed_on_next_closed_decision(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    old = cached(tmp_path, MODELS[1])
    new = cached(tmp_path, MODELS[2])
    advance(worker, ledger, 10, {
        9: [result(0, 1, 40000, 35000, window=2, block=9, flags=CONTROLS_OK | BASELINE)],
        10: [result(0, 2, 60000, 55000, window=2, block=10, kq=40000, kr=35000)]})
    worker.run_once(ledger.active, snapshot(10))
    assert old.exists() and new.exists()  # An acknowledged result is not a coronation.
    advance(worker, ledger, 12)
    worker.run_once(ledger.active, snapshot(12))
    assert ledger.get('king')['model_id'] == MODELS[2]
    assert not old.exists() and new.exists()


def test_window_change_during_gpu_lease_rechecks_the_new_king(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    revived = cached(tmp_path, MODELS[0])
    @contextmanager
    def lease():
        advance(worker, ledger, 12, {
            9: [result(1, 1, 40000, 35000, window=2, block=9, flags=CONTROLS_OK | BASELINE)],
            10: [result(1, 0, 60000, 55000, window=2, block=10, kq=40000, kr=35000)]})
        yield
    monkeypatch.setattr(gpu, 'lease', lease)
    worker._cleanup_models(ledger.active)
    assert ledger.get('king')['model_id'] == MODELS[0]
    assert revived.exists()


def test_restart_reclaims_retired_weights_without_another_evaluation(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    old = cached(tmp_path, MODELS[0], partial=True)
    king = cached(tmp_path, MODELS[1])
    worker.triggers.close()
    ledger.close()
    reopened = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    worker = Evaluator(root=tmp_path, hotkey=VALS[0], ledger=reopened, policy=None, judge=None,
                       batch_factory=None, download=None, runner=None)
    worker.update(snapshot(8), start_worker=False)
    worker._cleanup_models(reopened.active)
    assert not old.exists() and king.exists()


def test_chain_update_never_deletes_a_running_workers_model(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    weights = cached(tmp_path, MODELS[2])
    entered, release = threading.Event(), threading.Event()
    def running(*args):
        entered.set()
        assert release.wait(5)
        assert weights.exists()
    monkeypatch.setattr(worker, '_attempt', running)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_once, ledger.active, snapshot(8))
        try:
            assert entered.wait(5)
            advance(worker, ledger, 12, {
                9: [result(0, 1, 40000, 35000, window=2, block=9, flags=CONTROLS_OK | BASELINE)],
                10: [result(0, 2, 1000, 1000, window=2, block=10)]})
            assert weights.exists()
        finally:
            release.set()
        future.result(timeout=5)
    worker.run_once(ledger.active, snapshot(12))
    assert not weights.exists()


def test_cleanup_failure_retries_without_losing_result_or_redownloading(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    local = cached(tmp_path, MODELS[0])
    remote = cached(Path(gpu.workspace), MODELS[0])
    usage = ledger.usage(VALS[0])
    original = gpu.remove_models
    def offline(models):
        raise OSError('offline')
    monkeypatch.setattr(gpu, 'remove_models', offline)
    worker._cleanup_models(ledger.active)
    assert not local.exists() and remote.exists()
    assert json.loads((tmp_path/'model-cache-cleanup.json').read_text())['status'] == 'retry'
    assert ledger.usage(VALS[0]) == usage
    monkeypatch.setattr(gpu, 'remove_models', original)
    worker.cleanup_retry_at = 0.
    worker._cleanup_models(ledger.active)
    assert not remote.exists() and ledger.usage(VALS[0]) == usage


def test_pending_publication_keeps_an_otherwise_retired_cache(tmp_path, monkeypatch):
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    weights = cached(tmp_path, MODELS[0])
    worker.outbox = [{'value': result(0, 0, 20000, 15000)['value']}]
    worker._cleanup_models(ledger.active)
    assert weights.exists()
    worker.outbox = []
    worker.cleanup_retry_at = 0.
    worker._cleanup_models(ledger.active)
    assert not weights.exists()


@pytest.mark.parametrize('inconclusive', [False, True])
def test_early_stopped_weights_wait_for_final_consensus(tmp_path, monkeypatch, inconclusive):
    from test_stopping import make_evaluator
    from witness.benchmark.protocol import quantize
    def transport_ready(**kwargs):
        worker = Evaluator(**kwargs)
        worker.compression_check = lambda *args: None
        return worker
    monkeypatch.setattr('test_stopping.Evaluator', transport_ready)
    worker, ledger, calls = make_evaluator(tmp_path, monkeypatch)
    weights = cached(tmp_path, MODELS[2])
    king = cached(tmp_path, MODELS[1])
    worker.cleanup_retry_at = 0.
    worker._cleanup_models(ledger.active)
    assert weights.exists()  # Published partial is still continuable.
    stakes = {VALS[0]: 1, VALS[1]: 9} if inconclusive else snapshot()['validators']
    other = ([result(1, 1, quantize(.4), quantize(.4), window=2, block=11,
                     flags=CONTROLS_OK | BASELINE),
              result(1, 2, 65535, 65535, window=2, block=11, kq=quantize(.4), kr=quantize(.4))]
             if inconclusive else [])
    ledger.ingest(snapshot(11, validators=stakes), other)
    ledger.ingest(snapshot(12, validators=stakes), [])
    worker.update(snapshot(12, validators=stakes), start_worker=False)
    worker._cleanup_models(ledger.active)
    assert weights.exists() == inconclusive
    assert king.exists() and len(calls) == 16


@pytest.mark.parametrize('invalid', ['..', '../outside', '/tmp/model', 'a'*63, 'A'*64])
def test_invalid_id_is_rejected_before_any_deletion(tmp_path, invalid):
    weights = cached(tmp_path, MODELS[0])
    with pytest.raises(ValueError, match='invalid_model_cache_id'):
        remove_models(tmp_path/'models', [MODELS[0], invalid])
    assert weights.exists()


def test_symlink_roots_and_model_directories_never_delete_outside(tmp_path):
    outside = tmp_path/'outside'
    weights = cached(outside, MODELS[0])
    (tmp_path/'models').symlink_to(outside/'models', target_is_directory=True)
    with pytest.raises(ValueError, match='model_cache_symlink'):
        remove_models(tmp_path/'models', [MODELS[0]])
    (tmp_path/'models').unlink()
    (tmp_path/'models').mkdir()
    (tmp_path/'models'/MODELS[0]).symlink_to(weights.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe_model_cache_directory'):
        remove_models(tmp_path/'models', [MODELS[0]])
    assert weights.read_bytes() == b'cached model fixture'


def test_nested_symlink_is_unlinked_without_following_it(tmp_path):
    outside = tmp_path/'outside'
    outside.mkdir()
    sentinel = outside/'keep'
    sentinel.write_bytes(b'keep')
    weights = cached(tmp_path, MODELS[0])
    (weights.parent/'nested').symlink_to(outside, target_is_directory=True)
    assert remove_models(tmp_path/'models', [MODELS[0]]) == [MODELS[0]]
    assert sentinel.read_bytes() == b'keep'
    assert remove_models(tmp_path/'models', [MODELS[0]]) == []


def test_ssh_helper_deletes_same_scoped_tree_without_ensure(tmp_path, monkeypatch):
    gpu = SshGpu({'host': 'unused', 'ssh_key': 'unused', 'workspace': str(tmp_path/'gpu')}, tmp_path)
    weights = cached(Path(gpu.workspace), MODELS[0])
    def run(command, *, stdin, timeout):
        assert timeout == 30
        return subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=timeout)
    monkeypatch.setattr(gpu, 'run', run)
    monkeypatch.setattr(gpu, 'ensure', lambda: pytest.fail('cleanup must not provision compute'))
    assert gpu.remove_models([MODELS[0]]) == [MODELS[0]]
    assert not weights.exists()


def test_offline_runpod_cleanup_never_starts_or_rents_compute(tmp_path):
    gpu = RunPodGpu.__new__(RunPodGpu)
    gpu.state = {}
    with pytest.raises(RuntimeError, match='waiting_for_running_host'):
        gpu.remove_models([MODELS[0]])


def test_media_recovery_has_explicit_versioned_policy_identity():
    from witness.benchmark.protocol import policy_identity
    from witness.benchmark.protocol import LEGACY_POLICY, MEDIA_RECOVERY_WINDOW
    assert policy_identity() != LEGACY_POLICY
    assert MEDIA_RECOVERY_WINDOW == 9
