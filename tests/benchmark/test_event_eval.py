import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from witness.benchmark.contract import Claim, Fact, Reference, Response, Policy, InfrastructureError
from witness.benchmark.scoring import Assessment, Decision, Part
from witness.benchmark.reward import combine, reward
from witness.benchmark.ledger import Ledger
from witness.benchmark.protocol import policy_identity
from witness.benchmark.novel_review import NovelReviewer
from witness.benchmark.runner import _run_job
from witness.benchmark.gpu import LocalGpu
from witness.benchmark.performance import Performance
from witness.storage import write_private


def claim(id='a',start=0,end=12,description='A bell rings',modality='sound'):
    return Claim(id=id,start=start,end=end,subject='Bell',description=description,modality=modality)


def policy():
    return Policy(judge_id='judge',runtime_hash='a'*64,preprocessing_hash='b'*64,scoring_version='events-v2')


def ref(*claims):
    return Reference(kind='machine',duration=12,clip_sha256='c'*64,annotators=['d'*64],
                     facts=[Fact(claim=c,salience='core') for c in claims])


def score(reference,claims,links,**kwargs):
    response=Response(claims=claims)
    assessment=Assessment(judge_id='judge',decisions=[Decision.whole(c,links[c.id][0],links[c.id][1],'test') for c in claims])
    return combine([reference],response,[assessment],policy(),**kwargs)


def test_repeating_paraphrasing_or_splitting_an_event_does_not_increase_score():
    r=ref(claim())
    whole=score(r,[claim()],{'a':('supported',['a'])})
    split=[claim('a',0,4),claim('b',4,8,description='Bell sounds'),claim('c',8,12)]
    measured=score(r,split,{c.id:('supported',['a']) for c in split})
    assert measured['quality']==whole['quality']==1
    flooded=[claim(str(i),description=f'Paraphrase {i}') for i in range(64)]
    assert score(r,flooded,{c.id:('supported',['a']) for c in flooded})['quality']==1
    unknown=claim('u',description='A dog barks')
    honest=score(r,[claim(),unknown],{'a':('supported',['a']),'u':('unresolved',[])})
    flood=score(r,flooded[:63]+[unknown],{**{c.id:('supported',['a']) for c in flooded[:63]},'u':('unresolved',[])})
    assert flood['quality']==honest['quality']


def test_fact_weighting_not_equal_modality_and_duplicate_labels_have_no_weight():
    visual=[claim(str(i),description=f'Object {i}',modality='visual') for i in range(4)]
    r=ref(*visual,claim('sound'))
    measured=score(r,visual,{c.id:('supported',[c.id]) for c in visual})
    assert measured['recall']==.8
    r.facts.append(Fact(claim=visual[0].model_copy(update={'id':'duplicate'}),salience='core'))
    assert score(r,visual,{c.id:('supported',[c.id]) for c in visual})['recall']==.8


def test_duration_overclaim_loses_precision_and_partial_coverage_loses_recall():
    r=ref(claim(end=6))
    long=score(r,[claim()],{'a':('supported',['a'])})
    assert long['precision']==pytest.approx(7/12)
    partial=score(ref(claim()),[claim(end=3)],{'a':('supported',['a'])})
    assert partial['precision']==1 and partial['recall']==pytest.approx(4/12)


def test_unique_events_and_capacity_with_multiple_citations():
    from witness.benchmark.event_scoring import _credit
    assert _credit([(('f',),1),(('f',),1)])==1
    assert _credit([(('f','g'),1),(('f',),1)])==2
    assert _credit([(('f',),.5),(('f',),.5)])==1
    r=ref(claim('a',0,2),claim('b',8,10))
    assert score(r,[claim('x',0,2)],{'x':('supported',['a'])})['recall']==.5


def test_visual_review_repairs_precision_without_moving_recall_target():
    visual=claim('v',modality='visual')
    extra=claim('x',description='A red coat',modality='visual')
    before=score(ref(visual),[visual,extra],{'v':('supported',['v']),'x':('unresolved',[])})
    repaired=score(ref(visual),[visual,extra],{'v':('supported',['v']),'x':('unresolved',[])},
                   reviews={('x','a red coat'):{'status':'supported','event':'coat','start':0,'end':12}})
    assert repaired['recall']==before['recall']==1
    assert repaired['precision']==1>before['precision']
    packet=[{'id':'0','start':0,'end':12}]
    bad={'decisions':[{'id':'0','status':'supported','frames':[],'start':0,'end':12,'event':'coat'}]}
    validated=NovelReviewer._validate(bad,packet,12)
    assert validated['decisions'][0]['status']=='unresolved'
    good={'decisions':[{'id':'0','status':'supported','frames':[4], 'event':'coat'}]}
    validated=NovelReviewer._validate(good,packet,12)
    assert validated['decisions'][0]['spans']==[(1.75,2.25)]


def test_reward_ninety_ten_and_hard_invalid():
    assert reward(1,30,12,policy(),valid=True)['reward']==pytest.approx(.95)
    assert reward(1,61,12,policy(),valid=False)['reward']==0


def test_explicit_upgrade_preserves_old_windows_and_restarts(tmp_path):
    old='e'*64
    p=tmp_path/'ledger.sqlite3'
    ledger=Ledger(p,activation_block=1,activation_epoch=0,policy=old)
    ledger.db.execute('INSERT INTO windows VALUES (?,?,?)',(20,json.dumps({'id':20}),'closed'))
    ledger.close()
    upgrade={'first_window':21,'previous_policy':old}
    ledger=Ledger(p,activation_block=1,activation_epoch=0,policy=policy_identity(),policy_upgrade=upgrade)
    assert ledger.policy_for_window(20)==old
    assert ledger.policy_for_window(21)==policy_identity()
    assert ledger.db.execute('SELECT decision FROM windows WHERE id=20').fetchone()[0]=='closed'
    ledger.close()
    Ledger(p,activation_block=1,activation_epoch=0,policy=policy_identity(),policy_upgrade=upgrade).close()
    with pytest.raises(ValueError,match='schedule_changed'):
        Ledger(p,activation_block=1,activation_epoch=0,policy=policy_identity())


def test_upgrade_rejects_opened_window(tmp_path):
    p=tmp_path/'ledger.sqlite3';old='e'*64
    ledger=Ledger(p,activation_block=1,activation_epoch=0,policy=old)
    ledger.db.execute('INSERT INTO windows VALUES (?,?,NULL)',(21,'{}'));ledger.close()
    with pytest.raises(ValueError,match='unopened_window'):
        Ledger(p,activation_block=1,activation_epoch=0,policy=policy_identity(),
               policy_upgrade={'first_window':21,'previous_policy':old})


def test_stream_delivers_before_job_end_and_continuation_is_explicit(tmp_path):
    workspace=tmp_path/'gpu';remote=workspace/'jobs'/'test';remote.mkdir(parents=True)
    seen=[]
    gpu=LocalGpu({'workspace':str(workspace)},tmp_path)
    gpu.ensure=lambda:None
    def run(command,**kwargs):
        output=Path(command[-1]);spec=json.loads(Path(command[-2]).read_text())
        with output.open('w') as stream:
            stream.write(json.dumps({'task_id':'first'})+'\n');stream.flush()
            deadline=time.monotonic()+3
            while not Path(spec['continue_path']).exists():
                assert time.monotonic()<deadline
                time.sleep(.01)
            assert seen==['first']
            assert json.loads(Path(spec['continue_path']).read_text())=={'continue':True}
            stream.write(json.dumps({'task_id':'second'})+'\n');stream.flush()
        return SimpleNamespace(returncode=0,stderr='')
    gpu.run=run
    status,_=_run_job(gpu,'pod_runtime.py','omni',{},tmp_path/'spec.json',str(remote),4,
                      on_row=lambda row:seen.append(row['task_id']),pause_after=1,continue_batch=lambda:True)
    assert status==0 and seen==['first','second']


def test_parallel_phase_seconds_are_wall_time_not_sum(tmp_path):
    meter=Performance(tmp_path,{'model_id':'m','uid':1},{'id':1})
    barrier=threading.Barrier(2)
    def call():
        with meter.phase('judge'):
            barrier.wait();time.sleep(.03)
    threads=[threading.Thread(target=call) for _ in range(2)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert meter.phase_work['judge'] > meter.phases['judge']*1.5


@pytest.mark.parametrize('early',[False,True])
def test_stream_grades_in_draw_order_and_reuses_worker_at_barrier(tmp_path,monkeypatch,early):
    from witness.benchmark.evaluator import Evaluator
    from concurrent.futures import ThreadPoolExecutor
    model='a'*64
    worker=Evaluator(root=tmp_path,hotkey='validator',ledger=SimpleNamespace(),policy=policy(),
                     judge=None,batch_factory=None,download=None,runner=None)
    worker._cancelled=lambda window:False
    finished=[]
    def grade(case,*args,**kwargs):
        i=int(case.task.id)
        if i==0:time.sleep(.07)  # Later judgments finish first.
        finished.append(i)
        q=.1 if early else .9
        return {'video':case.source_group,'quality':q,'reward':q,'valid':True}
    monkeypatch.setattr('witness.benchmark.evaluator.grade',grade)
    selected=[(SimpleNamespace(task=SimpleNamespace(id=str(i),clip_sha256=f'{i:064x}',duration=12),
                               source_group=str(i//2),media_path=Path(str(i))),[]) for i in range(20)]
    rows=[{'start':0,'file':str(i)} for i in range(20)]
    class Runner:
        hardware_id='hardware'
        calls=0
        def stream(self,models,tasks,paths,*,on_execution,pause_after,continue_batch):
            self.calls+=1
            for i,task in enumerate(tasks):
                if i==pause_after and not continue_batch():break
                on_execution(SimpleNamespace(task_id=task.id))
    runner=Runner()
    result=worker._stream_evaluate(runner,model,{'id':30},rows,selected,[],{'reward':.9},False,
                                  tmp_path,tmp_path/'score.json',tmp_path/'progress.json',tmp_path/'hardware.json')
    expected=6 if early else 20
    assert runner.calls==1 and len(result['grades'])==expected
    assert [g['id'] for g in result['grades']]==[f'{i:064x}' for i in range(expected)]
    assert finished[0]!=0 and bool(result.get('early_stop'))==early


def test_api_parallelism_is_bounded_and_identical_calls_are_single_flight(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from witness.providers import ApiText
    api=ApiText('gpt-5.6-terra',tmp_path)
    lock=threading.Lock();active=0;maximum=0;calls=[]
    def call(prompt,value,**kwargs):
        nonlocal active,maximum
        with lock:
            active+=1;maximum=max(maximum,active);calls.append(value['x'])
        time.sleep(.02)
        with lock:active-=1
        return value
    monkeypatch.setattr(api,'_call',call)
    with ThreadPoolExecutor(12) as pool:
        list(pool.map(lambda i:api('p',{'x':i},schema={}),range(12)))
    assert 1<maximum<=4
    # Identical requests cannot enter the adapter simultaneously. Real _call
    # performs the cache lookup inside this lock before reserving any budget.
    maximum=0
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i:api('same',{'x':1},schema={}),range(8)))
    assert maximum==1
