import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from witness.events import EventsTaskSpec
from witness.subnet import mp4_validator as module


def setup(tmp_path,monkeypatch):
    class Chain:
        netuid=20
        epoch=10
        registered=True
        def epoch_state(self):return {'netuid':20,'epoch_index':self.epoch}
        def miner_endpoints(self):return [SimpleNamespace(hotkey='ours')] if self.registered else []
    chain=Chain();sent=[];prepared=[];key=Ed25519PrivateKey.generate()
    async def factory(epoch):
        prepared.append(epoch)
        jobs=[]
        for i in range(5):
            clip=tmp_path/f'{epoch}-{i}.mp4';clip.write_bytes(f'clip{i}'.encode())
            jobs.append({'original':str(i),'clip':str(clip),'clip_sha256':hashlib.sha256(clip.read_bytes()).hexdigest(),
                         'spec':EventsTaskSpec(duration=60.,fps=30.,has_audio=True).model_dump()})
        return jobs
    async def send(url,clip,spec,key,**kw):
        sent.append(clip.name)
        kw['on_dispatch'](SimpleNamespace(model_dump=lambda:{'task_id':str(len(sent))}))
        return {'elapsed_s':30.,'response':{'schema_version':'5.0','events':[]}}
    async def evaluate(job,received):return {'score':{'f1':1.}}
    monkeypatch.setattr(module,'send_clip',send)
    def create():return module.MP4Validator(chain,root=tmp_path/'state',target_hotkey='ours',url='http://miner',
        signing_key=key,jobs_factory=factory,evaluate=evaluate)
    return chain,create,sent,prepared


def test_five_distinct_tasks_and_no_epoch_catchup_or_restart_duplicates(tmp_path,monkeypatch):
    chain,create,sent,prepared=setup(tmp_path,monkeypatch)
    async def run():
        v=create();report=await v.step()
        assert report['planned']==report['dispatch_attempted']==report['valid']==5
        assert report['reward']==pytest.approx(.95)
        assert len(set(sent))==5 and prepared==[10]
        assert await v.step() is None
        v.close();v=create();v.state.recover()
        assert await v.step() is None
        chain.epoch=14
        await v.step()
        assert len(sent)==10 and prepared==[10,14]
        v.close()
    asyncio.run(run())


def test_interrupted_dispatch_is_never_resent_remaining_tasks_resume(tmp_path,monkeypatch):
    chain,create,sent,_=setup(tmp_path,monkeypatch)
    original=module.send_clip
    async def crash(*a,**kw):
        await original(*a,**kw)
        raise asyncio.CancelledError()
    async def run():
        v=create();monkeypatch.setattr(module,'send_clip',crash)
        with pytest.raises(asyncio.CancelledError):await v.step()
        v.close();v=create();v.state.recover()
        monkeypatch.setattr(module,'send_clip',original)
        report=await v.step()
        assert len(sent)==5 and len(set(sent))==5
        assert report['valid']==4 and report['statuses']['interrupted_unknown']==1
        assert report['dispatch_attempted']==5
        assert report['reward']==pytest.approx(.76)
        v.close()
    asyncio.run(run())


def test_failed_preparation_consumes_epoch_without_resending_or_burst(tmp_path,monkeypatch):
    chain,create,sent,_=setup(tmp_path,monkeypatch);attempts=[]
    async def prepare(epoch):
        attempts.append(epoch);chain.epoch=12
        raise ValueError('no_valid_sources')
    async def run():
        v=create();v.jobs_factory=prepare
        with pytest.raises(ValueError,match='no_valid_sources'):await v.step()
        assert await v.step() is None
        v.close();v=create()
        assert await v.step() is None and attempts==[10] and not sent
        chain.epoch=13;assert (await v.step())['valid']==5
        v.close()
    asyncio.run(run())


def test_crossed_epochs_consumed_and_unregistered_target_never_sent(tmp_path,monkeypatch):
    chain,create,sent,prepared=setup(tmp_path,monkeypatch)
    original=module.send_clip
    async def cross(*a,**kw):
        chain.epoch=12
        return await original(*a,**kw)
    monkeypatch.setattr(module,'send_clip',cross)
    async def run():
        v=create();report=await v.step()
        assert report['epoch']==10 and report['skipped_epochs']==[11,12]
        assert await v.step() is None
        chain.epoch=13;chain.registered=False
        with pytest.raises(ValueError,match='registered_target'):await v.step()
        assert len(sent)==5 and prepared==[10]
        v.close()
    asyncio.run(run())


def test_invalid_source_batch_consumes_epoch_without_preparation_loop(tmp_path,monkeypatch):
    chain,create,sent,_=setup(tmp_path,monkeypatch);prepared=[]
    async def invalid(epoch):
        prepared.append(epoch)
        return [{'original':'same'}]*5
    async def run():
        v=create();v.jobs_factory=invalid
        with pytest.raises(ValueError,match='distinct_originals'):await v.step()
        assert await v.step() is None and prepared==[10] and not sent
        v.close()
    asyncio.run(run())


def test_timeouts_score_zero_and_failed_evaluator_never_counts_success(tmp_path,monkeypatch):
    chain,create,sent,_=setup(tmp_path,monkeypatch)
    original=module.send_clip
    async def fail(*a,**kw):
        await original(*a,**kw)
        raise TimeoutError()
    monkeypatch.setattr(module,'send_clip',fail)
    async def run():
        v=create();report=await v.step()
        assert report['statuses']['expired']==5 and report['reward']==0 and report['scored']==0
        chain.epoch=11;monkeypatch.setattr(module,'send_clip',original)
        async def judge(*a):raise ValueError('failed')
        v.evaluate=judge;report=await v.step()
        assert report['valid']==5 and report['scored']==0 and report['reward']==0
        v.close()
    asyncio.run(run())


def test_three_epochs_over_real_http_with_persistent_restart(tmp_path):
    """Transport/scheduler evidence with a model-independent fixture handler."""
    import socket
    import uvicorn
    from witness.mp4 import create_app
    key=Ed25519PrivateKey.generate();seen=[]
    class Chain:
        netuid=20
        epoch=101
        def epoch_state(self):return {'epoch_index':self.epoch}
        def miner_endpoints(self):return [SimpleNamespace(hotkey='fixture')]
    chain=Chain()
    async def handler(clip,task):
        seen.append((task.task_id,clip.read_bytes()))
        return {'schema_version':'5.0','events':[]}
    async def jobs(epoch):
        result=[]
        for i in range(5):
            path=tmp_path/f'{epoch}-{i}.mp4';path.write_bytes(f'fixture-{epoch}-{i}'.encode())
            result.append({'original':str(i),'clip':str(path),
                'clip_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'spec':EventsTaskSpec(duration=60.,fps=30.,has_audio=False).model_dump()})
        return result
    async def evaluate(*a):return {'score':{'f1':0.,'precision':0.,'recall':0.}}
    async def run():
        app=create_app(handler,validator_key=key.public_key(),state=tmp_path/'miner')
        sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen();sock.setblocking(False)
        port=sock.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(app,log_level='error',access_log=False,lifespan='off'))
        serving=asyncio.create_task(server.serve(sockets=[sock]))
        def create():return module.MP4Validator(chain,root=tmp_path/'validator',target_hotkey='fixture',
            url=f'http://127.0.0.1:{port}',signing_key=key,jobs_factory=jobs,evaluate=evaluate)
        validator=None
        try:
            async with asyncio.timeout(10):
                while not server.started:await asyncio.sleep(.01)
                validator=create()
                for epoch in (101,102,103):
                    chain.epoch=epoch
                    report=await validator.step()
                    assert report['planned']==report['dispatch_attempted']==report['valid']==5
                    assert report['scored']==5 and report['f1']==report['reward']==0
                    validator.close();validator=create();validator.state.recover()
                    assert await validator.step() is None
                assert len(seen)==len({task for task,body in seen})==15
                assert len({body for task,body in seen})==15
                assert not list((tmp_path/'miner').glob('clip-*'))
        finally:
            if validator:validator.close()
            server.should_exit=True
            await serving
            sock.close()
    asyncio.run(run())
