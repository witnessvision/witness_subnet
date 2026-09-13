import json
import sqlite3
import httpx
import pytest
from witness.providers import ApiText

@pytest.fixture(autouse=True)
def daily_budget_path(tmp_path, monkeypatch):
    monkeypatch.setenv('WITNESS_BUDGET_DB', str(tmp_path/'daily.sqlite3'))

SCHEMA={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}


def fake_client(monkeypatch, calls, status=200, settlement=140000):
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,url,**kwargs):
            calls.append({'url':url,**kwargs})
            return httpx.Response(status,json={'model':kwargs['json']['model'],'id':'test-response',
                'status':'completed','usage':{'input_tokens':100,'output_tokens':20,'cost_nano_usd':settlement,
                    'input_tokens_details':{'cached_tokens':20,'cache_write_tokens':10}},
                'output':[{'type':'message','content':[{'type':'output_text','text':'{"ok":true}'}]}]})
    monkeypatch.setattr('witness.providers.httpx.Client',Client)
    monkeypatch.setenv('OPENAI_API_KEY','test-only-not-a-credential')


def test_receipt_cache_cost_and_no_key_persistence(tmp_path,monkeypatch):
    calls=[];fake_client(monkeypatch,calls)
    model=ApiText('gpt-5.6-terra',tmp_path,provider='openai')
    for _ in range(2):assert model('system',{'x':1},schema=SCHEMA)=={'ok':True}
    assert [r['cache_hit'] for r in model.calls]==[False,True]
    assert len(calls)==1 and calls[0]['json']['store'] is False
    assert 'tools' not in calls[0]['json']
    for p in tmp_path.glob('*.json'):assert 'test-only-not-a-credential' not in p.read_text()
    with sqlite3.connect(model.ledger) as db:
        cost=db.execute('SELECT cost FROM calls').fetchone()[0]
    assert cost==pytest.approx((70*2+10*2.5+20*.2+20*12)/1e6)


def test_cap_blocks_before_dispatch(tmp_path,monkeypatch):
    calls=[];fake_client(monkeypatch,calls)
    from witness.budget import DailyBudget, BudgetUnavailable
    DailyBudget(tmp_path/'daily.sqlite3').reserve(role='validator',provider='openai',input_hash='prior',upper_usd=9.)
    model=ApiText('gpt-5.6-terra',tmp_path,provider='openai')
    with pytest.raises(BudgetUnavailable,match='daily_budget_exhausted'):model('system',{},schema=SCHEMA)
    assert calls==[]


def test_cache_only_replays_but_never_sends_or_reserves_on_miss(tmp_path,monkeypatch):
    calls=[];fake_client(monkeypatch,calls)
    live=ApiText('gpt-5.6-luna',tmp_path,provider='openai')
    live('system',{'x':1},schema=SCHEMA)
    offline=ApiText('gpt-5.6-luna',tmp_path,cache_only=True,provider='openai')
    assert offline.identity==live.identity
    assert offline('system',{'x':1},schema=SCHEMA)=={'ok':True}
    with pytest.raises(ValueError,match='api_cache_miss'):
        offline('system',{'x':2},schema=SCHEMA)
    with sqlite3.connect(offline.ledger) as db:
        assert db.execute('SELECT count(*) FROM calls').fetchone()[0]==1
    assert len(calls)==1


def test_provider_error_is_bounded_and_never_retried(tmp_path,monkeypatch):
    calls=[];fake_client(monkeypatch,calls,status=429)
    model=ApiText('gpt-5.6-terra',tmp_path,provider='openai')
    with pytest.raises(RuntimeError,match='openai_status_429'):model('system',{},schema=SCHEMA)
    assert len(calls)==1
    with sqlite3.connect(model.ledger) as db:
        reserve,cost,record=db.execute('SELECT reserve,cost,record FROM calls').fetchone()
    assert reserve>0 and cost is None
    assert json.loads(record)['model_requested']=='gpt-5.6-terra'


def test_luna_fast_has_distinct_cache_identity_and_priced_reservation(tmp_path,monkeypatch):
    calls=[];fake_client(monkeypatch,calls)
    standard=ApiText('gpt-5.6-luna',tmp_path,provider='openai',service_tier='default')
    fast=ApiText('gpt-5.6-luna',tmp_path,provider='openai',service_tier='priority')
    assert standard.identity!=fast.identity
    standard('system',{},schema=SCHEMA);fast('system',{},schema=SCHEMA)
    assert [c['json']['service_tier'] for c in calls]==['default','priority']
    with sqlite3.connect(fast.ledger) as db:
        rows=db.execute('select reserve,cost from calls order by rowid').fetchall()
    assert rows[1][0]==pytest.approx(rows[0][0]*2)
    assert rows[1][1]==pytest.approx(rows[0][1]*2)


def test_saygm_routes_only_scoped_key_and_keeps_provider_caches_distinct(tmp_path,monkeypatch):
    calls=[];fake_client(monkeypatch,calls)
    monkeypatch.setenv('GM_API_KEY','saygm-test-only')
    original=ApiText('gpt-5.6-luna',tmp_path,provider='openai')
    alternative=ApiText('gpt-5.6-luna',tmp_path,provider='saygm')
    assert alternative.identity!=original.identity
    original('system',{},schema=SCHEMA)
    image={'type':'input_image','detail':'high','image_url':'data:image/jpeg;base64,test'}
    alternative('system',{},schema=SCHEMA)
    alternative('system',{},schema=SCHEMA)
    alternative('system',{},schema=SCHEMA,images=[image])
    assert len(calls)==3
    assert calls[1]['url']=='https://api.saygm.com/v1/responses'
    assert calls[1]['headers']=={'Authorization':'Bearer saygm-test-only'}
    assert calls[2]['json']['input'][0]['content'][-1]==image
    with sqlite3.connect(alternative.ledger) as db:
        cost,record=db.execute('SELECT cost,record FROM calls ORDER BY rowid DESC LIMIT 1').fetchone()
    assert cost==pytest.approx(.00014)
    assert json.loads(record)['cost_basis']=='provider_settled_nano_usd'
    for path in tmp_path.glob('*.json'):assert 'saygm-test-only' not in path.read_text()


@pytest.mark.parametrize('settlement',[None,-1,True,'123',1.5])
def test_saygm_missing_or_invalid_settlement_retains_reservation(tmp_path,monkeypatch,settlement):
    calls=[];fake_client(monkeypatch,calls,settlement=settlement)
    monkeypatch.setenv('GM_API_KEY','saygm-test-only')
    model=ApiText('gpt-5.6-luna',tmp_path,provider='saygm')
    with pytest.raises(ValueError,match='invalid_provider_settlement'):
        model('system',{},schema=SCHEMA)
    with sqlite3.connect(model.ledger) as db:
        assert db.execute('SELECT cost FROM calls').fetchone()[0] is None
    assert len(calls)==1 and not list(tmp_path.glob('*.json'))


def test_provider_credentials_and_configuration_fail_closed(tmp_path,monkeypatch):
    from witness.providers import load_key
    monkeypatch.delenv('GM_API_KEY',raising=False)
    env=tmp_path/'.env';env.write_text('OPENAI_API_KEY=openai-test\nGM_API_KEY=gm-test\n');env.chmod(0o600)
    monkeypatch.setenv('OPENAI_API_KEY','preserved-test')
    load_key(env,'saygm')
    import os
    assert os.environ['GM_API_KEY']=='gm-test' and os.environ['OPENAI_API_KEY']=='preserved-test'
    env.chmod(0o644)
    with pytest.raises(ValueError,match='unsafe_credential_permissions'):load_key(env,'saygm')
    with pytest.raises(ValueError,match='unsupported_api_provider'):ApiText('gpt-5.6-luna',tmp_path,provider='other')
    with pytest.raises(ValueError,match='unpriced_service_tier'):
        ApiText('gpt-5.6-luna',tmp_path,provider='saygm',service_tier='priority')
    monkeypatch.delenv('GM_API_KEY')
    calls=[];fake_client(monkeypatch,calls)
    model=ApiText('gpt-5.6-luna',tmp_path,provider='saygm')
    with pytest.raises(ValueError,match='api_key_unavailable'):model('system',{},schema=SCHEMA)
    assert calls==[]
