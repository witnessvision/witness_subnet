import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from witness.events import EventsTaskSpec
from witness.events_evaluation import judge_response, PROMPT_HASH
from witness.subnet import production_validator as module

EVALUATOR = {'evaluator_id':'synthetic-evaluator','prompt_hash':PROMPT_HASH}
CALIBRATION = {**EVALUATOR, 'passed':True}
RESPONSE = {'schema_version':'5.0','events':[{'timestamp':20.,'actor':'persona','action':'camina','objects':[],'details':[]}]}


def fixture(tmp_path, monkeypatch, n=6):
    class Chain:
        epoch=10
        netuid=20
        validator_hotkey='validator'
        wallet=SimpleNamespace(hotkey=None)
        registry={uid:'hotkey-'+str(uid) for uid in range(n)}
        def epoch_state(self):return {'epoch_index':self.epoch}
        def miner_endpoints(self):
            return [SimpleNamespace(uid=uid,hotkey=hotkey,axon=SimpleNamespace(ip='1.1.1.1',port=8000+uid)) for uid,hotkey in self.registry.items()]
        def registered_hotkey(self,uid):return self.registry.get(uid)
    chain=Chain(); sent=[]; active={}; peaks=[]
    async def jobs(epoch):
        result=[]
        for i in range(5):
            path=tmp_path/(str(i)+'.mp4');path.write_bytes(str(i).encode())
            result.append({'original':str(i),'clip':str(path),'clip_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'spec':EventsTaskSpec(duration=60.,fps=30.,has_audio=False).model_dump(),
                'reference':{'duration':60.,'events':[{'timestamp':20.,'text':'A person walks.'}]}})
        return result
    async def send(url,clip,spec,key,**kw):
        hotkey=kw['miner_hotkey'];tid=kw['task_id']
        kw['on_dispatch'](SimpleNamespace(model_dump=lambda:{'task_id':tid}))
        sent.append((hotkey,clip.read_bytes(),tid))
        active[hotkey]=active.get(hotkey,0)+1
        assert active[hotkey]==1
        peaks.append(sum(active.values()))
        try:await asyncio.sleep(.005)
        finally:active[hotkey]-=1
        return {'response':RESPONSE,'elapsed_s':6.}
    async def evaluate(job,received):
        return judge_response(job['reference'],received['response'],
            lambda p,v:{'relation':'supported','fields':{k:'supported' for k in v['event_fields']}},
            evaluator_id=EVALUATOR['evaluator_id'],calibration=CALIBRATION)
    monkeypatch.setattr(module,'send_clip',send)
    def create(identity=EVALUATOR):
        return module.ProductionValidator(chain,root=tmp_path/'state',jobs_factory=jobs,evaluate=evaluate,
                                         evaluator_identity=identity,calibration=CALIBRATION)
    return chain,create,sent,peaks


def test_all_endpoints_same_five_tasks_global_four_and_hotkey_ema(tmp_path,monkeypatch):
    chain,create,sent,peaks=fixture(tmp_path,monkeypatch)
    async def run():
        v=create();report=await v.step()
        assert report['complete'] and report['planned']==30
        assert report['winner']['uid']==0
        assert report['winner']['ema']==pytest.approx(.2*.99)
        assert max(peaks)==4
        for hotkey in chain.registry.values():
            assert [clip for h,clip,_ in sent if h==hotkey]==[str(i).encode() for i in range(5)]
        assert len({tid for _,_,tid in sent})==30
        assert await v.step() is None
        v.close();v=create();v.state.recover(chain.epoch)
        assert await v.step() is None
        chain.epoch=15;chain.registry[0]='new-hotkey'
        report=await v.step()
        assert report['winner']['uid']==1
        assert next(m['ema'] for m in report['miners'] if m['uid']==0)==pytest.approx(.198)
        assert len(sent)==60
        v.close()
        with pytest.raises(ValueError,match='new_evaluator'):
            create({**EVALUATOR,'provider':'changed'})
    asyncio.run(run())


def test_provider_failure_retains_response_and_never_penalizes_or_updates_ema(tmp_path,monkeypatch):
    _,create,sent,_=fixture(tmp_path,monkeypatch,n=2)
    async def run():
        v=create()
        async def unavailable(*args):raise RuntimeError('daily_budget_exhausted')
        v.evaluate=unavailable
        report=await v.step()
        assert not report['complete'] and report['winner'] is None
        assert all(m['mean_score'] is None and m['valid']==5 for m in report['miners'])
        assert v.state.db.execute('SELECT COUNT(*) FROM ranking').fetchone()[0]==0
        for task in v.state.tasks(report['round_id']):
            assert task['result']['received']['response']==RESPONSE
            assert task['result']['reward'] is None
        v.close()
    asyncio.run(run())


def test_absent_incompatible_and_invalid_responses_zero(tmp_path,monkeypatch):
    _,create,sent,_=fixture(tmp_path,monkeypatch,n=2)
    original=module.send_clip
    async def send(*args,**kwargs):
        result=await original(*args,**kwargs)
        if kwargs['miner_hotkey']=='hotkey-0':raise TimeoutError()
        return result
    monkeypatch.setattr(module,'send_clip',send)
    async def run():
        v=create();report=await v.step()
        assert report['complete'] and report['winner']['uid']==1
        assert report['miners'][0]['mean_score']==0.
        v.close()
    asyncio.run(run())


def test_restart_never_retries_unknown_dispatch_or_catches_up_old_epoch(tmp_path,monkeypatch):
    chain,create,sent,_=fixture(tmp_path,monkeypatch,n=1)
    original=module.send_clip
    async def crash(*args,**kwargs):
        await original(*args,**kwargs)
        raise asyncio.CancelledError()
    async def run():
        v=create();active=v.state.begin(10,await v.jobs_factory(10),module.announced_endpoints(chain.miner_endpoints()))
        first=v.state.tasks(active['id'])[0]
        monkeypatch.setattr(module,'send_clip',crash)
        with pytest.raises(asyncio.CancelledError):
            await v.dispatch(first,active['jobs'][0],active['endpoints'][0])
        v.close();v=create();chain.epoch=12;v.state.recover(12)
        monkeypatch.setattr(module,'send_clip',original)
        report=await v.step()
        assert not report['complete'] and report['winner'] is None and len(sent)==1
        assert await v.step() is None
        chain.epoch=13;assert (await v.step())['complete']
        assert len(sent)==6 and len(set(tid for _,_,tid in sent))==6
        v.close()
    asyncio.run(run())


def test_encrypted_signer_resolved_once_before_network_loop(tmp_path, monkeypatch):
    chain, create, sent, _ = fixture(tmp_path, monkeypatch, n=2)
    class Wallet:
        accesses = 0
        @property
        def hotkey(self):
            self.accesses += 1
            if self.accesses > 1:
                raise RuntimeError('synchronous_password_kdf_during_dispatch')
            return None
    chain.wallet = Wallet()
    async def run():
        validator = create()
        assert chain.wallet.accesses == 1
        report = await validator.step()
        assert report['complete'] and len(sent) == 10
        chain.epoch += 1
        assert (await validator.step())['complete'] and len(sent) == 20
        assert chain.wallet.accesses == 1
        validator.close()
    asyncio.run(run())
