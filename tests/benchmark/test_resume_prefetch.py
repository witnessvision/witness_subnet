"""Recover exact first answers, bounded prefetch, and measured timing records."""
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time
import pytest

from witness.benchmark.contract import InfrastructureError, Policy, Task
from witness.benchmark.runner import PodRunner, _run_job
from witness.benchmark.prefetch import Prefetch
from witness.benchmark.performance import Performance
from witness.storage import write_private
from witness.benchmark.protocol import TEN_VIDEO_POLICY, RESUME_WINDOW, policy_identity
from witness.benchmark.ledger import Ledger
from witness.benchmark.managed_evaluator import ManagedEvaluator
from test_scheduling import evaluator_at
from test_validator import snapshot, MODELS, entry


def make_runner(tmp_path, monkeypatch, job='attempt1'):
    import witness.benchmark.runner as module
    gpu=SimpleNamespace(root=tmp_path,workspace='/gpu',put=lambda *a:None,
                        run=lambda *a,**k:SimpleNamespace(returncode=0,stdout=''),read=lambda p:'')
    monkeypatch.setattr(module,'prepare',lambda *a:'/gpu/jobs')
    monkeypatch.setattr(module,'verify_directory',lambda *a:None)
    policy=Policy(judge_id='test',runtime_hash='a'*64,preprocessing_hash='b'*64)
    runner=PodRunner(gpu,{'c'*64:{'path':str(tmp_path),'manifest':{'arch':'qwen2.5-omni','files':[]}}},policy,job)
    runner.execution_cache=tmp_path/'executions'
    tasks=[Task(id=f'clip{i}',clip_sha256=f'{i:064x}',duration=20.) for i in range(3)]
    return runner,tasks


def row(task, hardware='GPU-fixed'):
    return {'task_id':task.id,'status':'ok','elapsed_s':2.,'raw':'first answer','hardware_id':hardware}


def test_partial_job_recovers_only_flushed_answers_and_restart_skips_them(tmp_path,monkeypatch):
    import witness.benchmark.runner as module
    first,tasks=make_runner(tmp_path,monkeypatch)
    def interrupted(gpu,script,env,spec,artifact,*args,**kwargs):
        write_private(artifact.with_suffix('.output.json'),{'lines':json.dumps(row(tasks[0]))+'\n{"task_id":'})
        raise InfrastructureError('evaluation_cancelled_or_budget_expired')
    monkeypatch.setattr(module,'_run_job',interrupted)
    with pytest.raises(InfrastructureError): first(['c'*64],tasks,[Path('x')]*3)
    assert len(list((tmp_path/'executions').glob('*.json')))==1
    restarted,tasks=make_runner(tmp_path,monkeypatch,'attempt2');seen=[]
    def complete(gpu,script,env,spec,*args,**kwargs):
        seen.extend(t['task_id'] for t in spec['tasks'])
        return 0,'\n'.join(json.dumps(row(t)) for t in tasks[1:])
    monkeypatch.setattr(module,'_run_job',complete)
    results=restarted(['c'*64],tasks,[Path('x')]*3)['c'*64]
    assert seen==['clip1','clip2'] and len(results)==3
    assert results['clip0'].raw=='first answer' and results['clip0'].elapsed_s==2.
    assert restarted.reused_task_ids=={('c'*64,'clip0')}
    # A completed batch survives a judge failure without another GPU command.
    last,_=make_runner(tmp_path,monkeypatch,'attempt3')
    monkeypatch.setattr(module,'_run_job',lambda *a,**k:pytest.fail('repeated inference'))
    assert last(['c'*64],tasks,[Path('x')]*3)['c'*64]==results


def test_cache_binding_changes_with_task_policy_or_model_and_rejects_hardware_mix(tmp_path,monkeypatch):
    import witness.benchmark.runner as module
    runner,tasks=make_runner(tmp_path,monkeypatch)
    runner._save_lines('c'*64,tasks,json.dumps(row(tasks[0])))
    assert not runner._saved('c'*64,[tasks[0].model_copy(update={'clip_sha256':'d'*64})])
    runner.policy=runner.policy.model_copy(update={'runtime_hash':'e'*64})
    assert not runner._saved('c'*64,tasks)
    runner,_=make_runner(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'_run_job',lambda *a,**k:(0,'\n'.join(json.dumps(row(t,'differentGPU')) for t in tasks[1:])))
    with pytest.raises(InfrastructureError,match='gpu_changed'):runner(['c'*64],tasks,[Path('x')]*3)


def test_cancelled_runtime_flush_is_preserved_before_raising(tmp_path):
    gpu=SimpleNamespace(workspace='/gpu',put=lambda *a:None,
                        run=lambda *a,**k:SimpleNamespace(returncode=75,stderr=''),
                        read=lambda p:json.dumps({'task_id':'done'})+'\n')
    artifact=tmp_path/'job.json'
    with pytest.raises(InfrastructureError,match='cancelled'):
        _run_job(gpu,'pod_runtime.py','omni',{},artifact,'/gpu/job',1.)
    assert 'done' in json.loads(artifact.with_suffix('.output.json').read_text())['lines']


def test_prefetch_is_parallel_bounded_and_cancelled_independently():
    entered=threading.Event();release=threading.Event()
    def acquire(cancel,remaining):
        entered.set()
        while not release.wait(.01):
            if cancel():raise InterruptedError('cancelled')
        return {'verified':True}
    job=Prefetch({'model_id':'a'*64}, {'id':13},acquire,lambda:False,timeout_s=1.)
    assert entered.wait(1) and job.thread.is_alive()
    release.set();assert job.wait(lambda:False)=={'verified':True}
    cancelled=Prefetch({'model_id':'a'*64},{'id':13},acquire,lambda:True,timeout_s=1.)
    cancelled.thread.join(1)
    # This acquisition checks cancellation between requests, as the production client does.
    assert not cancelled.thread.is_alive()


def test_prefetch_only_frozen_candidate_one_at_time_and_disk_reserve(tmp_path,monkeypatch):
    worker,ledger,gpu=evaluator_at(tmp_path,monkeypatch,block=8)
    gate=threading.Event();calls=[]
    worker.prefetch_download=lambda e,s,c,**k: (calls.append(e['model_id']),gate.wait(1),{'verified':True})[-1]
    worker.compression_check=lambda *a:None
    monkeypatch.setattr('witness.benchmark.managed_evaluator.shutil.disk_usage',lambda p:SimpleNamespace(free=300*10**9))
    window=ledger.active;active=next(iter(window['candidates'].values()))
    usage_before=ledger.usage(worker.hotkey)
    worker._start_prefetch(active,window,snapshot(8))
    assert worker.prefetch is not None
    first=worker.prefetch
    assert first.entry['block']<window['start_block'] and first.entry['model_id']!=active['model_id']
    worker._start_prefetch(active,window,snapshot(8));assert worker.prefetch is first
    gate.set();first.thread.join(2)
    assert len(calls)==1 and ledger.usage(worker.hotkey)==usage_before
    worker.prefetch=None
    monkeypatch.setattr('witness.benchmark.managed_evaluator.shutil.disk_usage',lambda p:SimpleNamespace(free=10*10**9))
    worker._start_prefetch(active,window,snapshot(8));assert worker.prefetch is None
    ledger.close()


def test_performance_records_cumulative_active_wall_and_percentiles(tmp_path):
    e=entry(0);window={'id':13}
    first=Performance(tmp_path,e,window)
    with first.phase('download'):time.sleep(.01)
    first.clips=[{'model_id':'m','task_id':'a','elapsed_s':2.,'status':'ok','reused':False}]
    a=first.finish('deferred',RuntimeError())
    time.sleep(.01)
    second=Performance(tmp_path,e,window)
    second.clips=[{'model_id':'m','task_id':'a','elapsed_s':2.,'status':'ok','reused':True},{'model_id':'m','task_id':'b','elapsed_s':3.,'status':'ok','reused':False}]
    b=second.finish('evaluation_ready_for_publication')
    assert b['attempts']==2 and b['active_s']>a['active_s']
    assert b['wall_s']>b['active_s'] and b['clip_latency_s']=={'count':2,'p50':2.,'p95':3.,'max':3.}
    assert b['cumulative_phases_s']['download']>=.01


def test_epoch_boundary_does_not_cancel_new_attempt_but_window_and_deadline_do(tmp_path,monkeypatch):
    worker,ledger,gpu=evaluator_at(tmp_path,monkeypatch,block=8)
    window={**ledger.active,'id':RESUME_WINDOW}
    worker.context=(window,snapshot(8,epoch_index=5));worker.attempt_epoch=4
    worker.deadline=time.monotonic()+1800
    assert not worker._cancelled(window) and worker._interruption(RESUME_WINDOW) is None
    worker.deadline=time.monotonic()-1
    assert worker._cancelled(window) and worker._interruption(RESUME_WINDOW)=='attempt_budget_elapsed'
    worker.deadline=None;worker.context=({**window,'id':RESUME_WINDOW+1},snapshot(9))
    assert worker._cancelled(window) and worker._interruption(RESUME_WINDOW)=='window_changed'
    ledger.close()


def test_resume_policy_migration_keeps_reported_ten_video_window(tmp_path):
    p=tmp_path/'chain.sqlite3'
    old=Ledger(p,activation_block=1,activation_epoch=0,policy=TEN_VIDEO_POLICY)
    old.set('cursor',25);old.set('king',{'uid':92});old.close()
    new=Ledger(p,activation_block=1,activation_epoch=0,policy=policy_identity())
    assert new.cursor==25 and new.get('king')=={'uid':92}
    assert new.policy_for_window(12)==TEN_VIDEO_POLICY and new.policy_for_window(13)==policy_identity()
    assert new.get('policy_migration')['first_window']==13
    new.close()


def test_cumulative_round_budget_caps_next_attempt_and_prevents_repeated_work(tmp_path, monkeypatch):
    from witness.benchmark.managed_evaluator import EvaluationRoundBudgetExceeded
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    window=ledger.active; candidate=window['candidates'][MODELS[2]]
    record=tmp_path/'performance'/str(window['id'])/(candidate['model_id']+'.json')
    write_private(record,{'active_s':2600.})
    worker.deadline=time.monotonic()+1800
    def check(*args):
        assert 95 < worker.deadline-time.monotonic() <= 100
    monkeypatch.setattr(worker,'_attempt_checked',check)
    ManagedEvaluator._attempt(worker,candidate,window,snapshot(8),set())
    write_private(record,{'active_s':2700.})
    monkeypatch.setattr(worker,'_attempt_checked',lambda *a:pytest.fail('exhausted round reran'))
    with pytest.raises(EvaluationRoundBudgetExceeded):
        ManagedEvaluator._attempt(worker,candidate,window,snapshot(8),set())
    assert not worker.outbox
    ledger.close()


def test_exhausted_round_deadline_is_parked_but_window_change_is_not(tmp_path, monkeypatch):
    from witness.benchmark.managed_evaluator import EvaluationRoundBudgetExceeded
    worker, ledger, gpu = evaluator_at(tmp_path, monkeypatch, block=8)
    window=ledger.active; candidate=window['candidates'][MODELS[2]]
    record=tmp_path/'performance'/str(window['id'])/(candidate['model_id']+'.json')
    worker.context=(window,snapshot(8))
    def expired(*args):
        worker.deadline=time.monotonic()-1
        raise InfrastructureError('evaluation_cancelled_or_budget_expired')
    monkeypatch.setattr(worker,'_attempt_checked',expired)
    write_private(record,{'active_s':2600.}); worker.deadline=time.monotonic()+1800
    with pytest.raises(EvaluationRoundBudgetExceeded):
        ManagedEvaluator._attempt(worker,candidate,window,snapshot(8),set())
    worker.context=({**window,'id':window['id']+1},snapshot(8))
    write_private(record,{'active_s':2600.}); worker.deadline=time.monotonic()+1800
    with pytest.raises(InfrastructureError):
        ManagedEvaluator._attempt(worker,candidate,window,snapshot(8),set())
    ledger.close()
