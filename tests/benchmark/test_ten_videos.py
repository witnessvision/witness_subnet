"""Ten-video activation, replay preservation and exact finite-population risk."""
from itertools import combinations
import json

import pytest

from witness.benchmark import pool
from witness.benchmark.ledger import Ledger
from witness.benchmark.protocol import FIVE_VIDEO_POLICY, LEGACY_POLICY, TEN_VIDEO_POLICY, Result, policy_identity
from witness.benchmark.reward import eval_for_window
from witness.benchmark.stopping import futility
from test_media_recovery import fake_preparation, replay


def test_ten_video_selection_stable_restart_and_grouped(tmp_path, monkeypatch):
    fake_preparation(monkeypatch, failures={}, calls=[])
    # Fake preparation writes metadata only; exercise the real window sampler.
    def load(target, policy, **kwargs):
        rows = [json.loads(x) for x in (target/'clips.jsonl').read_text().splitlines()]
        return list(reversed([r for r in rows if r['status']=='ok']))
    monkeypatch.setattr(pool, 'load_pool', load)
    rows = pool.window_batch(tmp_path,12,'validator',gpu=None,api=None,policy=None)
    assert len(rows)==20 and len({r['video'] for r in rows})==10
    assert [r['index'] for r in rows]==[0,1]*10
    assert pool.window_batch(tmp_path,12,'validator',gpu=None,api=None,policy=None)==rows
    assert eval_for_window(11).videos==5 and eval_for_window(12).videos==10


def test_ten_video_migration_preserves_reported_history_and_restart(tmp_path):
    path=tmp_path/'old.sqlite3'
    old=replay(path,FIVE_VIDEO_POLICY)
    record=Result(9,0,'a'*64,FIVE_VIDEO_POLICY[:24],'b'*64,0,0,0,0,1)
    old.db.execute('INSERT INTO commitments VALUES (?,?,?)',(19,'validator',record.commitment))
    history=[tuple(r) for r in old.db.execute('SELECT * FROM windows')]
    active=old.active; cursor=old.cursor; old.close()
    updated=Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity())
    assert [tuple(r) for r in updated.db.execute('SELECT * FROM windows')]==history
    assert updated.active==active and updated.cursor==cursor
    assert updated.policy_for_window(8)==LEGACY_POLICY
    assert updated.policy_for_window(11)==FIVE_VIDEO_POLICY
    assert updated.policy_for_window(12)==TEN_VIDEO_POLICY
    assert updated.policy_for_window(13)==policy_identity()
    updated.close()
    restarted=Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity())
    assert restarted.active==active
    restarted.close()


def test_ten_video_migration_refuses_already_reported_affected_window(tmp_path):
    path=tmp_path/'old.sqlite3'; old=replay(path,FIVE_VIDEO_POLICY)
    record=Result(12,0,'a'*64,FIVE_VIDEO_POLICY[:24],'b'*64,0,0,0,0,1)
    old.db.execute('INSERT INTO commitments VALUES (?,?,?)',(25,'validator',record.commitment)); old.close()
    with pytest.raises(ValueError,match='requires_unreported'):
        Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity())


def test_ten_video_futility_false_stops_at_most_ten_percent_exact_prefixes():
    populations=[[(i+1)/10 for i in range(10)]]
    populations += [[.1]*(10-high)+[1.]*high for high in range(11)]
    for values in populations:
        for king in (.2,.4,.6,.8):
            if sum(values)/10 <= king+.0201:
                continue
            false=sum(futility([{'quality':values[i],'reward':values[i]} for i in ids],king,
                               planned_videos=10) is not None for ids in combinations(range(10),3))
            assert false<=12  # 120 equiprobable prefixes, 90% bound.
    low={'quality':.1,'reward':.1}
    cut=futility([low]*3,.6,planned_videos=10)
    assert cut['planned_videos']==10 and cut['confidence']==.9
    assert futility([low]*4,.6,planned_videos=10) is None
    assert futility([low]*9,.6,planned_videos=10)['confidence']==1.


def test_result_and_report_keep_current_window_policy_during_upgrade(tmp_path):
    import threading
    from witness.benchmark.evaluator import Evaluator
    worker=Evaluator.__new__(Evaluator)
    worker.root=tmp_path; worker.hotkey='validator'; worker.lock=threading.RLock(); worker.outbox=[]
    worker.ledger=Ledger(tmp_path/'chain.sqlite3',activation_block=1,activation_epoch=0,policy=policy_identity())
    entry={'uid':1,'model_id':'a'*64,'hotkey':'miner'}
    scored={'quality':.5,'reward':.4,'per_video':{},'grades':[]}
    for window_id,expected in [(11,FIVE_VIDEO_POLICY),(12,TEN_VIDEO_POLICY),(13,policy_identity())]:
        window={'id':window_id,'start_block':1,'king':None}
        worker._enqueue(entry,window,scored,None,flags=1)
        item=worker.outbox[-1]
        assert Result.parse(item['value']).policy==expected[:24]
        report=json.loads((tmp_path/'reports'/(item['report_hash']+'.json')).read_text())
        assert report['policy_hash']==expected
    worker.ledger.close()
