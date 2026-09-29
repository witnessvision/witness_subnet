"""Media recovery must preserve sampling, miner attempts and finalized history."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from witness.benchmark import pool
from witness.benchmark.contract import InfrastructureError
from witness.benchmark.ledger import Ledger
from witness.benchmark.protocol import LEGACY_POLICY, FIVE_VIDEO_POLICY, Result, policy_identity
from witness.benchmark.submission import Submission, challenge_id
from witness.storage import write_private


def fake_preparation(monkeypatch, *, failures, calls):
    def build(target, *, selected, **kwargs):
        target.mkdir(parents=True, exist_ok=True)
        path = target / 'clips.jsonl'
        old = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        rows = [r for r in old if r['video'] in {v['identifier'] for v in selected}]
        for video in selected:
            name = video['identifier']
            if any(r['video'] == name for r in rows):
                continue
            calls.append(name)
            if name in failures:
                rows.append(dict(video=name, status='failed', error=failures[name]))
            else:
                rows.extend(dict(video=name, status='ok', index=i) for i in range(2))
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def load(target, policy, **kwargs):
        rows = [json.loads(x) for x in (target/'clips.jsonl').read_text().splitlines()]
        good = [r for r in rows if r['status']=='ok']
        if len(good)!=10:
            raise ValueError('evaluation_pool_too_small')
        return list(reversed(good))
    monkeypatch.setattr(pool, 'build_pool', build)
    monkeypatch.setattr(pool, 'load_pool', load)


def initial_draw(tmp_path, monkeypatch):
    calls=[]
    fake_preparation(monkeypatch, failures={}, calls=calls)
    pool.window_batch(tmp_path, 9, 'validator', gpu=None, api=None, policy=None)
    selected=json.loads((tmp_path/'windows/9/draw.json').read_text())['selected']
    (tmp_path/'windows/9/selected.json').unlink()
    (tmp_path/'windows/9/clips.jsonl').unlink()
    return selected


def test_missing_audio_replaced_once_retains_four_valid_and_restart_order(tmp_path, monkeypatch):
    initial=initial_draw(tmp_path, monkeypatch)
    bad=initial[2]['identifier']; calls=[]
    fake_preparation(monkeypatch, failures={bad:'ValueError: missing_audio_or_video'}, calls=calls)
    first=pool.window_batch(tmp_path,9,'validator',gpu=None,api=None,policy=None)
    again=pool.window_batch(tmp_path,9,'validator',gpu=None,api=None,policy=None)
    assert first==again and len(first)==10
    chosen=[r['video'] for r in first[::2]]
    assert chosen[:2]==[v['identifier'] for v in initial[:2]]
    assert chosen[3:]==[v['identifier'] for v in initial[3:]]
    assert bad not in chosen and len(set(chosen))==5
    assert calls.count(bad)==1 and len(calls)==6
    assert json.loads((tmp_path/'windows/9/draw.json').read_text())['selected']==initial
    assert [r['index'] for r in first]==[0,1]*5


def test_transient_download_error_does_not_replace_or_lock(tmp_path, monkeypatch):
    initial=initial_draw(tmp_path, monkeypatch); calls=[]
    fake_preparation(monkeypatch, failures={initial[0]['identifier']:'ReadTimeout: timeout'}, calls=calls)
    with pytest.raises(ValueError,match='evaluation_pool_too_small'):
        pool.window_batch(tmp_path,9,'validator',gpu=None,api=None,policy=None)
    assert not (tmp_path/'media-rejections.json').exists()
    assert not (tmp_path/'windows/9/selected.json').exists()
    assert calls==[v['identifier'] for v in initial]


def test_interrupted_recovery_resumes_same_replacement(tmp_path, monkeypatch):
    initial=initial_draw(tmp_path,monkeypatch); calls=[]
    fake_preparation(monkeypatch,failures={initial[0]['identifier']:'ValueError: missing_audio_or_video'},calls=calls)
    with pytest.raises(InfrastructureError,match='interrupted'):
        pool.window_batch(tmp_path,9,'validator',gpu=SimpleNamespace(cancelled=lambda:True),api=None,policy=None)
    rows=pool.window_batch(tmp_path,9,'validator',gpu=None,api=None,policy=None)
    assert len(rows)==10 and calls.count(initial[0]['identifier'])==1


def test_build_pool_does_not_redownload_permanent_failure(tmp_path,monkeypatch):
    write_private(tmp_path/'unused.json',{})
    video={'identifier':'broken'}
    (tmp_path/'clips.jsonl').write_text(json.dumps(dict(video='broken',status='failed',error='ValueError: missing_audio_or_video'))+'\n')
    monkeypatch.setattr(pool,'_cut',lambda *a,**k:pytest.fail('invalid source was retried'))
    pool.build_pool(tmp_path,gpu=None,api=None,selected=[video],salt='seed',log=lambda x:None)
    assert 'missing_audio_or_video' in (tmp_path/'clips.jsonl').read_text()


def replay(path,policy):
    ledger=Ledger(path,activation_block=1,activation_epoch=0,policy=policy)
    miners=['miner-a','miner-b']; voter='validator'
    keys=miners+[voter]
    for block in range(1,20):
        snapshot={'block':block,'block_hash':str(block),'epoch_index':block-1,
                  'uids':dict(zip(keys,range(3))),'coldkeys':{k:k for k in keys},'validators':{voter:1}}
        rows=[]
        if block==1:
            rows=[{'block':block,'hotkey':m,'value':Submission(str(i+1)*64,'e'*64).commitment}
                  for i,m in enumerate(miners)]
        if block==17:
            rows=[{'block':block,'hotkey':voter,'value':Result(8,i,challenge_id(m,str(i+1)*64),
                     LEGACY_POLICY[:24],'a'*64,60000-i*20000,60000-i*20000,0,0,1).commitment}
                  for i,m in enumerate(miners)]
        ledger.ingest(snapshot,rows)
    return ledger


def test_migration_equals_fresh_replay_preserves_king_closed_history_and_uses(tmp_path):
    path=tmp_path/'old.sqlite3'
    old=replay(path,LEGACY_POLICY)
    king=old.get('king'); uses=old.usage('validator')
    assert king and len(uses)==2
    closed=list(old.db.execute('SELECT id,opening,decision FROM windows WHERE id<9'))
    old.close()
    migrated=Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity())
    fresh=replay(tmp_path/'fresh.sqlite3',policy_identity())
    assert migrated.get('king')==fresh.get('king')==king
    assert migrated.usage('validator')==fresh.usage('validator')==uses
    assert [tuple(r) for r in migrated.db.execute('SELECT id,opening,decision FROM windows WHERE id<9')]==[tuple(r) for r in closed]
    assert migrated.active==fresh.active
    assert migrated.active['policy_hash']==FIVE_VIDEO_POLICY
    assert migrated.get('policy_migration')['first_window']==9
    migrated.close(); fresh.close()


def test_migration_rejects_reported_window_and_arbitrary_policy(tmp_path):
    path=tmp_path/'old.sqlite3'; old=replay(path,LEGACY_POLICY)
    record=Result(9,0,'a'*64,LEGACY_POLICY[:24],'b'*64,0,0,0,0,1)
    old.db.execute('INSERT INTO commitments VALUES (?,?,?)',(19,'validator',record.commitment)); old.close()
    with pytest.raises(ValueError,match='requires_unreported'):
        Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity())
    with pytest.raises(ValueError,match='explicit_migration'):
        Ledger(path,activation_block=1,activation_epoch=0,policy='f'*64)
