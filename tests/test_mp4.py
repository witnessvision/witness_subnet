import asyncio
import hashlib
import time
import uuid
from pathlib import Path
import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from witness.events import EventsTaskSpec
from witness.mp4 import ClipTask, create_app, signed_headers


def test_direct_adapter_does_not_initialize_chain_modules():
    import subprocess
    import sys
    subprocess.run([sys.executable,'-c',"import sys,witness.mp4; assert 'bittensor' not in sys.modules"],
                   check=True,timeout=10,stdin=subprocess.DEVNULL)


def test_client_rejects_encoded_response_before_decompression(tmp_path,monkeypatch):
    from witness.mp4 import send_clip
    read=[]
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            read.append(True)
            yield b'not accessed'
    def reply(request):return httpx.Response(200,headers={'content-encoding':'gzip'},stream=Body())
    original=httpx.AsyncClient
    monkeypatch.setattr('witness.mp4.httpx.AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(reply),**kwargs))
    clip=tmp_path/'clip.mp4';clip.write_bytes(b'clip')
    with pytest.raises(ValueError,match='encoded_response_not_supported'):
        asyncio.run(send_clip('http://test',clip,task().task_spec,Ed25519PrivateKey.generate()))
    assert read==[]


def test_reward_clock_stops_before_validator_response_validation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from witness import mp4
    from witness.events import canonical_bytes
    ticks=[10.];original_validate=mp4.validate_events
    def validate(value,duration):
        ticks[0]+=5.
        return original_validate(value,duration)
    monkeypatch.setattr(mp4,'time',SimpleNamespace(time=time.time,monotonic=lambda:ticks[0]))
    monkeypatch.setattr(mp4,'validate_events',validate)
    body=canonical_bytes({'schema_version':'5.0','events':[]})
    def reply(request):return httpx.Response(200,content=body,headers={
        'x-witness-response-sha256':hashlib.sha256(body).hexdigest()})
    original=httpx.AsyncClient
    monkeypatch.setattr(mp4.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(reply),**kw))
    clip=tmp_path/'clip.mp4';clip.write_bytes(b'clip')
    result=asyncio.run(mp4.send_clip('http://test',clip,task().task_spec,Ed25519PrivateKey.generate()))
    assert ticks[0]==15. and result['elapsed_s']==0.


def task(raw=b"clip"):
    now=time.time()
    return ClipTask(task_id=uuid.uuid4().hex,issued_at=now,expires_at=now+180,
        clip_sha256=hashlib.sha256(raw).hexdigest(),clip_bytes=len(raw),
        task_spec=EventsTaskSpec(duration=60.,fps=30.,has_audio=True))


def test_signed_body_isolated_cleaned_and_replay_survives_restart(tmp_path):
    async def run():
        key=Ed25519PrivateKey.generate(); seen=[]
        async def handler(clip,t):
            assert clip.read_bytes()==b"clip"
            assert clip.name=="video.mp4" and clip.stat().st_mode&0o077==0
            seen.append(clip)
            return {"schema_version":"5.0","events":[]}
        t=task()
        for expected in [200,409]:
            app=create_app(handler,validator_key=key.public_key(),state=tmp_path)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as c:
                r=await c.post('/events',content=b'clip',headers=signed_headers(t,key))
                assert r.status_code==expected
                if expected==200:assert r.headers['x-witness-response-sha256']==hashlib.sha256(r.content).hexdigest()
        assert len(seen)==1 and not seen[0].exists()
        assert not list(tmp_path.glob('clip-*'))
    asyncio.run(run())


@pytest.mark.parametrize('mutation',['signature','body','length','expired','extra','oversize'])
def test_reject_invalid_requests_before_inference(tmp_path,mutation):
    async def run():
        key=Ed25519PrivateKey.generate(); called=[]
        async def handler(*args):called.append(True);return {}
        t=task();raw=b'clip'
        if mutation=='expired':t=t.model_copy(update={'expires_at':time.time()-1})
        if mutation=='oversize':t=t.model_copy(update={'clip_bytes':129*1024*1024})
        h=signed_headers(t,key)
        if mutation=='signature':h['x-witness-signature']='AAAA'
        if mutation=='body':raw=b'evil'
        if mutation=='length':raw=b'clip plus more'
        if mutation=='extra':
            import base64,json
            from witness.mp4 import DOMAIN
            from witness.events import canonical_bytes
            body=canonical_bytes({**t.model_dump(),'source_url':'private-canary'})
            h['x-witness-task']=base64.b64encode(body).decode()
            h['x-witness-signature']=base64.b64encode(key.sign(DOMAIN+body)).decode()
        app=create_app(handler,validator_key=key.public_key(),state=tmp_path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            r=await c.post('/events',content=raw,headers=h)
        assert r.status_code in [400,422]
        assert not called and 'private-canary' not in r.text
        assert not list(tmp_path.glob('clip-*'))
    asyncio.run(run())


def test_timeout_cancels_worker_and_releases_slot(tmp_path):
    async def run():
        key=Ed25519PrivateKey.generate(); stopped=[]
        async def handler(*args):
            try:await asyncio.sleep(10)
            finally:stopped.append(True)
        app=create_app(handler,validator_key=key.public_key(),state=tmp_path,internal_timeout=.08)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            for _ in range(2):
                r=await c.post('/events',content=b'clip',headers=signed_headers(task(),key))
                assert r.status_code==504
        assert len(stopped)==2 and not list(tmp_path.glob('clip-*'))
    asyncio.run(run())


def test_incomplete_or_oversize_response_never_success(tmp_path):
    async def run():
        key=Ed25519PrivateKey.generate()
        for result in [{}, {'schema_version':'5.0','events':[],'extra':'x'*2097152}]:
            async def handler(*args):return result
            app=create_app(handler,validator_key=key.public_key(),state=tmp_path)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
                r=await c.post('/events',content=b'clip',headers=signed_headers(task(),key))
                assert r.status_code==422
                assert len(r.content)<100
    asyncio.run(run())


def test_disconnect_kills_descendant_and_cleans_clip(tmp_path):
    async def run():
        import os
        import sys
        from witness.subnet.processes import run_process
        key=Ed25519PrivateKey.generate(); marker=tmp_path/'child.pid'
        entered=asyncio.Event()
        async def handler(*args):
            entered.set()
            inner='import os,time,pathlib;pathlib.Path('+repr(str(marker))+').write_text(str(os.getpid()));time.sleep(30)'
            script=('import asyncio,sys;from witness.subnet.processes import run_process;'
                    'asyncio.run(run_process([sys.executable,"-c",'+repr(inner)+'],timeout=20,new_session=False))')
            await run_process([sys.executable,'-c',script],timeout=20)
        app=create_app(handler,validator_key=key.public_key(),state=tmp_path)
        messages=[]; delivered=False
        async def receive():
            nonlocal delivered
            if not delivered:
                delivered=True
                return {'type':'http.request','body':b'clip','more_body':False}
            await entered.wait()
            for _ in range(100):
                if marker.exists():break
                await asyncio.sleep(.01)
            return {'type':'http.disconnect'}
        async def send(message):messages.append(message)
        scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
            'scheme':'http','path':'/events','raw_path':b'/events','query_string':b'',
            'root_path':'','headers':[(k.encode(),v.encode()) for k,v in signed_headers(task(),key).items()],
            'client':('127.0.0.1',1234),'server':('127.0.0.1',80)}
        async with asyncio.timeout(3):await app(scope,receive,send)
        assert marker.exists()
        pid=int(marker.read_text())
        # An orphan awaiting init's reap is stopped too; it executes no work.
        status=Path(f'/proc/{pid}/stat')
        assert not status.exists() or status.read_text().split()[2]=='Z'
        assert not list(tmp_path.glob('clip-*'))
    asyncio.run(run())
