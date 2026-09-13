import asyncio
import io
import json
from pathlib import Path
import subprocess
import time
import wave
import zipfile

import httpx
from PIL import Image
import pytest

from witness.events import content_hash, observation_budget
from witness.tools.native_media import inspect_original, run_decoder
from witness.tools.server import Budget, DEFAULT_FFMPEG, create_app, SessionClosed
from witness.subnet.chain import InMemoryChainAdapter
from witness.subnet.events_validator import EventsValidator
from witness.score_v5_0_0 import score_events


@pytest.fixture
def original(tmp_path):
    return make_original(tmp_path)


def make_original(tmp_path,duration=60):
    directory=tmp_path/'source-private-canary';directory.mkdir()
    subprocess.run([str(DEFAULT_FFMPEG),'-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=10',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t',str(duration),'-c:v','libx264',
        '-preset','ultrafast','-threads','1','-c:a','aac',
        '-metadata','title=PRIVATE_CONTAINER_TITLE_CANARY','-metadata','comment=PRIVATE_CONTAINER_COMMENT_CANARY',
        '-metadata:s:a:0','title=PRIVATE_STREAM_TITLE_CANARY',str(directory/'video.mp4')],check=True,timeout=20)
    meta=inspect_original(directory/'video.mp4',Path(DEFAULT_FFMPEG).with_name('ffprobe'))
    meta['private_canary']='never-return-reference'
    (directory/'scene.json').write_text(json.dumps(meta))
    return directory,meta


def test_private_window_offset_preserves_source_frames_audio_and_relative_times(tmp_path):
    directory,meta=make_original(tmp_path,duration=90)
    window=tmp_path/'window';window.mkdir();(window/'video.mp4').symlink_to(directory/'video.mp4')
    (window/'scene.json').write_text(json.dumps({**meta,'duration':60.,'video_duration':60.,'source_offset_s':10.03}))
    original_app=create_app(directory,log_dir=tmp_path/'original-log',transcript_source='none')
    clip_app=create_app(window,log_dir=tmp_path/'clip-log',transcript_source='none')
    a=original_app.state.store.create(directory.name,Budget(**observation_budget(90)))
    b=clip_app.state.store.create(window.name,Budget(**observation_budget(60)))
    async def check():
        async with (httpx.AsyncClient(transport=httpx.ASGITransport(app=original_app),base_url='http://original') as source,
                    httpx.AsyncClient(transport=httpx.ASGITransport(app=clip_app),base_url='http://clip') as clip):
            prefix=f'/s/{b.session_id}'
            reference=await source.get(f'/s/{a.session_id}/frame',params={'t':11.06})
            observed=await clip.get(prefix+'/frame',params={'t':1.03})
            assert observed.content==reference.content
            assert float(observed.headers['X-Witness-Observation-Time'])==pytest.approx(1.07,abs=.001)
            reference=await source.get(f'/s/{a.session_id}/audio',params={'t0':10.03,'t1':11.03})
            observed=await clip.get(prefix+'/audio',params={'t0':0,'t1':1})
            assert observed.content==reference.content
            terminal=await clip.get(prefix+'/frame',params={'t':59.999})
            assert 59.9<=float(terminal.headers['X-Witness-Observation-Time'])<60
            metadata=await clip.get(prefix+'/meta')
            assert set(metadata.json())=={'duration','cost','has_audio'}
            assert 'source_offset' not in metadata.text and '10.03' not in metadata.text
            assert (await clip.get(prefix+'/frame',params={'t':60.1})).status_code==400
    asyncio.run(check())


def test_native_media_time_audio_privacy_and_closed_sessions(original,tmp_path):
    directory,meta=original
    app=create_app(directory,log_dir=tmp_path/'logs',transcript_source='none',allow_session_creation=False)
    store=app.state.store
    a=store.create(directory.name,Budget(**observation_budget(60)),deadline_at=time.monotonic()+20)
    b=store.create(directory.name,Budget(**observation_budget(60)))
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,raise_app_exceptions=False),base_url='http://test') as client:
            prefix=f'/s/{a.session_id}'
            frame=await client.get(prefix+'/frame',params={'t':1.03})
            assert frame.status_code==200
            assert b'PRIVATE_CONTAINER' not in frame.content
            assert float(frame.headers['X-Witness-Observation-Time'])==pytest.approx(1.1,abs=.001)
            frames=await client.get(prefix+'/frames',params={'t0':58,'t1':60,'fps':2})
            with zipfile.ZipFile(io.BytesIO(frames.content)) as archive:
                manifest=json.loads(archive.read('manifest.json'))
                assert manifest['timestamps']==[58,58.5,59,59.5]
                assert Image.open(io.BytesIO(archive.read('frame_000003.jpg'))).size==(640,360)
                assert 'canary' not in json.dumps(manifest)
                for name in archive.namelist():
                    content=archive.read(name)
                    assert b'PRIVATE_CONTAINER' not in content and b'PRIVATE_STREAM' not in content
            audio=await client.get(prefix+'/audio',params={'t0':0,'t1':1})
            assert b'PRIVATE_CONTAINER' not in audio.content and b'PRIVATE_STREAM' not in audio.content
            with wave.open(io.BytesIO(audio.content)) as stream:
                assert stream.getframerate()==48000 and stream.getnframes()==48000
            for path in ('/meta','/transcript?t0=0&t1=60','/frame?t=never-return-reference'):
                r=await client.get(prefix+path)
                assert 'never-return-reference' not in r.text and 'source-private-canary' not in str(r.headers)
            assert (await client.post('/session',json={'scene_id':directory.name,'budget':a.budget.model_dump()})).status_code==403
            assert b.cost.audio_seconds==0 and b.cost.visual_tokens==0
            store.close(a.session_id)
            assert (await client.get(prefix+'/meta')).status_code==404
    asyncio.run(check())
    with pytest.raises(ValueError):create_app(directory,log_dir=tmp_path/'bad')
    closed=store.create(directory.name,Budget(**observation_budget(60)),deadline_at=time.monotonic()-.1)
    with pytest.raises(SessionClosed):run_decoder(['/bin/sleep','10'],closed)


def test_five_tasks_only_our_registered_miner_epoch_skip_and_reproduction(original,tmp_path):
    directory,meta=original
    calls=[]
    async def respond(task):
        calls.append(task.task_id)
        assert 'canary' not in task.model_dump_json()
        task.reconstruction={'schema_version':'5.0','events':[]}
        task.trace_summary={'status':'ok'}
        return task
    chain=InMemoryChainAdapter({1:respond,2:respond});chain.netuid=20;chain.epoch=10
    chain.epoch_state=lambda:{'epoch_index':chain.epoch,'netuid':20}
    reference={'duration':meta['duration'],'events':[{'timestamp':20.,'text':'private reference canary'}]}
    async def jobs():
        return [{'original':str(i),'video':str(directory/'video.mp4'),'media_sha256':meta['media_sha256'],
                 'reference':reference,'reference_hash':content_hash(reference),'indexed':True} for i in range(5)]
    async def evaluate(ref,pred):
        chain.epoch=12
        return {'score':score_events(ref,pred,[],evaluator_id='test'),'decisions':[]}
    validator=EventsValidator(chain,root=tmp_path/'history',target_hotkey='fake-hotkey-1',jobs_factory=jobs,evaluate=evaluate)
    async def check():
        report=await validator.step()
        assert report['metrics']['planned']==report['metrics']['sent']==report['metrics']['completed']==5
        assert report['skipped_epochs']==[11,12]
        assert report['metrics']['f1']==0 and report['metrics']['provisional'] is True
        assert len(calls)==len(set(calls))==5 and chain.weight_history==[]
        assert all(row['uid']==1 for row in report['rows'])
        assert await validator.step() is None
        assert 'canary' not in json.dumps(report['feedback'])
        for row in report['rows']:
            assert row['evaluation']['score']==score_events(reference,row['response'],[],evaluator_id='test')
    try:asyncio.run(check())
    finally:validator.state.close()
