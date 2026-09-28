"""Finite-batch error bound, real process cancellation and partial-chain lifecycle."""
from itertools import combinations, product
import json
import multiprocessing
from types import SimpleNamespace
import threading
import time

import pytest

from witness.benchmark.contract import Policy
from witness.benchmark.evaluator import Evaluator
from witness.benchmark.ledger import Ledger
from witness.benchmark.pod_runtime import supervise
from witness.benchmark.protocol import BASELINE, CONTROLS_OK, EARLY_STOP, Result, decide, quantize
from witness.benchmark.stopping import futility
from test_validator import MINERS, MODELS, POLICY, VALS, entry, fake_worker, result, snapshot


def test_exact_90_percent_bound_over_all_random_three_video_samples():
    # Exhaust every ordered five-video population on this score grid, not Monte
    # Carlo. At most one of the ten possible prefixes can falsely stop a winner.
    for population in product((0., .2, .6, 1.), repeat=5):
        actual = sum(population) / 5
        for king in (.2, .4, .6, .8):
            if quantize(actual) - quantize(king) <= .02 * 65535:
                continue
            false = sum(futility([{'quality': population[i], 'reward': population[i]} for i in prefix], king)
                        is not None for prefix in combinations(range(5), 3))
            assert false <= 1


def test_single_statistical_look_and_deterministic_completion():
    low = {'quality': .1, 'reward': .1}
    assert futility([low] * 2, .4) is None
    cut = futility([low] * 3, .4)
    assert cut['method'] == 'finite_batch_90' and cut['confidence'] == .9
    assert futility([low] * 4, .4)['method'] == 'best_possible_completion'
    assert futility([low] * 5, .4) is None
    assert futility([{'quality': .8, 'reward': .8}] * 3, .4) is None


def test_partial_cannot_crown_or_drop_its_stake_and_full_supersedes():
    baseline = [result(v, 0, 40000, 30000, flags=CONTROLS_OK | BASELINE) for v in (0, 1)]
    partial = result(0, 1, 40000, 20000, flags=CONTROLS_OK | EARLY_STOP)
    full = result(1, 1, 60000, 60000)
    kwargs = dict(window=1, policy=POLICY, king=entry(0), candidates={MODELS[1]: entry(1)}, snapshot=snapshot())
    out = decide(baseline + [partial, full], **kwargs)
    assert out['king']['hotkey'] == MINERS[0] and not out['inconclusive']
    assert out['early_losses'] == [{'hotkey': partial['hotkey'], 'value': partial['value']}]
    wrong_policy = result(0, 1, 10000, 10000, flags=CONTROLS_OK | EARLY_STOP, block=4, policy='e'*64)
    assert decide([wrong_policy] + baseline + [partial, full], **kwargs)['early_losses'] == out['early_losses']
    kwargs['snapshot'] = snapshot(validators={VALS[0]: 1, VALS[1]: 9})
    out = decide(baseline + [partial, full], **kwargs)
    assert out['king']['hotkey'] == MINERS[0] and out['inconclusive'] == [MODELS[1]]
    assert not out['early_losses']
    completed = result(0, 1, 50000, 45000, block=6)
    out = decide(baseline + [partial, full, completed], **kwargs)
    assert out['king']['hotkey'] == MINERS[1] and not out['inconclusive']
    with pytest.raises(ValueError, match='early_stop_flags'):
        result(0, 0, 40000, 30000, flags=CONTROLS_OK | BASELINE | EARLY_STOP)


def test_sixty_second_clip_limit_and_budget_kills_actual_worker(tmp_path):
    policy = Policy(judge_id='test', runtime_hash='a'*64, preprocessing_hash='b'*64)
    assert [policy.deadline_s(d) for d in (10., 20., 30.)] == [60.] * 3
    job = {'load_timeout_s': 2, 'job_timeout_s': .15,
           'tasks': [{'task_id': 'hang', 'deadline_s': 5}, {'task_id': 'next', 'deadline_s': 1}]}
    before = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    assert supervise(job, tmp_path/'out.jsonl', worker=fake_worker,
                     context=multiprocessing.get_context('fork')) == 75
    assert time.monotonic() - started < 2
    assert not (set(p.pid for p in multiprocessing.active_children()) - before)
    assert (tmp_path/'out.jsonl').read_text() == ''


def test_remote_cancel_marker_interrupts_warmup(tmp_path):
    cancel = tmp_path/'cancel'
    job = {'load_timeout_s': 2, 'cancel_path': str(cancel),
           'tasks': [{'task_id': 'hang', 'deadline_s': 5}]}
    timer = threading.Timer(.15, cancel.touch)
    timer.start()
    started = time.monotonic()
    try:
        assert supervise(job, tmp_path/'out.jsonl', worker=fake_worker,
                         context=multiprocessing.get_context('fork')) == 75
        assert time.monotonic() - started < 2
    finally:
        timer.join()


def make_evaluator(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    for block in range(1, 9):
        rows = [entry(i) for i in range(3)] if block == 1 else []
        if block == 5: rows = [result(0, 0, 20000, 15000, kq=0, kr=0)]
        if block == 6: rows = [result(0, 1, 40000, 35000, kq=0, kr=0, block=6)]
        ledger.ingest(snapshot(block), rows)
    rows = [{'video': f'v{i//2}', 'index': i%2, 'file': f'c{i}.mp4', 'start': 0.} for i in range(10)]
    selected = [(SimpleNamespace(task=SimpleNamespace(id=f'c{i}', clip_sha256=f'{i:064x}', duration=20.),
                                 media_path=f'c{i}.mp4', source_group=f'v{i//2}'), []) for i in range(10)]
    calls = []
    def runner(items, job):
        def run(models, tasks, paths):
            calls.extend((models[0], t.id) for t in tasks)
            return {models[0]: {t.id: object() for t in tasks}}
        run.hardware_id = 'test-gpu'
        return run
    def grade(case, refs, execution, model, **kwargs):
        score = .4 if model == MODELS[1] else .1 if int(case.task.id[1:]) < 6 else 1.
        return {'video': case.source_group, 'quality': score, 'reward': score}
    monkeypatch.setattr('witness.benchmark.evaluator.grade', grade)
    evaluator = Evaluator(root=tmp_path, hotkey=VALS[0], ledger=ledger, policy=None, judge=None,
                          batch_factory=None, download=lambda e,s,c: {'content_id': e['model_id']}, runner=runner)
    monkeypatch.setattr(evaluator, '_batch', lambda window: (rows, selected))
    evaluator.update(snapshot(8), start_worker=False)
    evaluator.run_once(ledger.active, snapshot(8))
    assert len([c for c in calls if c[0] == MODELS[1]]) == 10
    assert len([c for c in calls if c[0] == MODELS[2]]) == 6  # four GPU inferences genuinely not scheduled
    for block, queued in zip((9, 10), list(evaluator.outbox)):
        ledger.ingest(snapshot(block), [{'hotkey': VALS[0], 'block': block, 'value': queued['value']}])
    evaluator.update(snapshot(10), start_worker=False)
    return evaluator, ledger, calls


def test_early_loss_is_not_consumed_until_close_and_report_is_explicit(tmp_path, monkeypatch):
    evaluator, ledger, calls = make_evaluator(tmp_path, monkeypatch)
    assert MINERS[2] not in ledger.usage(VALS[0])
    evaluator.run_once(ledger.active, snapshot(10))
    assert len(calls) == 16  # no repeated look/run on polling
    ledger.ingest(snapshot(11), [])
    ledger.ingest(snapshot(12), [])
    usage = ledger.usage(VALS[0])[MINERS[2]]
    assert usage['result']['flags'] & EARLY_STOP
    report = json.loads((tmp_path/'reports'/f"{usage['result']['evidence']}.json").read_text())
    assert report['evaluation_status'] == 'early_stop'
    assert report['total']['reward'] is None and report['total']['reward_upper'] > 0
    ledger.close()


def test_other_validator_can_require_remaining_videos_without_repeating_first_three(tmp_path, monkeypatch):
    evaluator, ledger, calls = make_evaluator(tmp_path, monkeypatch)
    stakes = {VALS[0]: 1, VALS[1]: 9}
    other = [result(1, 1, quantize(.4), quantize(.4), flags=CONTROLS_OK | BASELINE, window=2, block=11),
             result(1, 2, 65535, 65535, kq=quantize(.4), kr=quantize(.4), window=2, block=11)]
    ledger.ingest(snapshot(11, validators=stakes), other)
    evaluator.update(snapshot(11, validators=stakes), start_worker=False)
    evaluator.run_once(ledger.active, snapshot(11, validators=stakes))
    candidate_calls = [c for c in calls if c[0] == MODELS[2]]
    assert len(candidate_calls) == 10 and len(set(candidate_calls)) == 10
    completed = Result.parse(evaluator.outbox[-1]['value'])
    assert not completed.flags & EARLY_STOP
    assert completed.reward == quantize(.46)
    ledger.close()


def test_unresolved_partial_stays_retryable_after_close_and_restart(tmp_path, monkeypatch):
    evaluator, ledger, calls = make_evaluator(tmp_path, monkeypatch)
    stakes = {VALS[0]: 1, VALS[1]: 9}
    other = [result(1, 1, quantize(.4), quantize(.4), flags=CONTROLS_OK | BASELINE, window=2, block=11),
             result(1, 2, 65535, 65535, kq=quantize(.4), kr=quantize(.4), window=2, block=11)]
    ledger.ingest(snapshot(11, validators=stakes), other)
    ledger.ingest(snapshot(12, validators=stakes), [])
    assert ledger.get('decision')['inconclusive'] == [MODELS[2]]
    assert ledger.get('king')['hotkey'] == MINERS[1]
    assert MINERS[2] not in ledger.usage(VALS[0])
    evaluator.update(snapshot(12, validators=stakes), start_worker=False)
    assert next(r for r in evaluator.triggers.rows() if r['hotkey'] == MINERS[2])['usage'] == 'reserved'
    ledger.close()
    reopened = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    assert MINERS[2] not in reopened.usage(VALS[0])
    assert reopened.get('decision')['inconclusive'] == [MODELS[2]]
    reopened.close()


def test_epoch_change_and_expired_attempt_cancel_without_a_score(tmp_path, monkeypatch):
    evaluator, ledger, calls = make_evaluator(tmp_path, monkeypatch)
    window = ledger.active
    evaluator.deadline = time.monotonic() + 10
    evaluator.attempt_epoch = 5
    assert not evaluator._cancelled(window)
    evaluator.context = (window, snapshot(11, epoch_index=6))
    assert evaluator._cancelled(window)
    evaluator.context = (window, snapshot(10))
    evaluator.deadline = time.monotonic() - 1
    assert evaluator._cancelled(window)
    ledger.close()


def test_failed_continuation_keeps_complete_videos_and_resumes_without_another_look(tmp_path, monkeypatch):
    evaluator, ledger, calls = make_evaluator(tmp_path, monkeypatch)
    import witness.benchmark.evaluator as module
    original_grade = module.grade
    failed = False
    def interrupted(case, *args, **kwargs):
        nonlocal failed
        if case.task.id == 'c8' and not failed:
            failed = True
            raise ValueError('judge_unavailable')
        return original_grade(case, *args, **kwargs)
    monkeypatch.setattr(module, 'grade', interrupted)
    window = ledger.active
    evaluator.attempt_epoch = snapshot(10)['epoch_index']
    evaluator.deadline = time.monotonic() + 900
    rows, selected = evaluator._batch(window)
    baseline = json.loads((tmp_path/'windows/2/models'/MODELS[1]/'score.json').read_text())
    with pytest.raises(ValueError, match='judge_unavailable'):
        evaluator._evaluate(entry(2), window, snapshot(10), rows, selected, baseline=baseline, resume=True)
    progress = json.loads((tmp_path/'windows/2/models'/MODELS[2]/'progress.json').read_text())
    assert len(progress['grades']) == 8 and progress['continue_to_full']
    completed = evaluator._evaluate(entry(2), window, snapshot(10), rows, selected, baseline=baseline, resume=True)
    assert len(completed['grades']) == 10 and not completed.get('early_stop')
    assert completed['reward'] == pytest.approx(.46)
    candidate_calls = [clip for model, clip in calls if model == MODELS[2]]
    assert all(candidate_calls.count(f'c{i}') == 1 for i in range(8))
    ledger.close()
