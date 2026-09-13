import asyncio
import base64
import hashlib
import json
import time
import uuid

import httpx
import pytest
from bittensor_wallet import Keypair

from witness.events import EventsTaskSpec, canonical_bytes
from witness.mp4_v5_2 import ClipTask, create_app, signed_headers, response_message, verify, REQUEST_DOMAIN


def identities():
    return Keypair.create_from_uri('//Alice'), Keypair.create_from_uri('//Bob')


def task(validator, miner):
    return ClipTask(task_id=uuid.uuid4().hex, issued_at=time.time(), expires_at=time.time()+179,
        clip_sha256=hashlib.sha256(b'clip').hexdigest(), clip_bytes=4, netuid=20,
        validator_hotkey=validator.ss58_address, miner_hotkey=miner.ss58_address,
        task_spec=EventsTaskSpec(duration=60., fps=30., has_audio=False))


def test_signed_response_binds_both_hotkeys_task_and_body_and_replay(tmp_path):
    async def run():
        validator, miner = identities()
        t = task(validator, miner)
        async def handler(*args): return {'schema_version':'5.0','events':[]}
        for status in (200,409):
            app = create_app(handler, hotkey=miner, validator_hotkeys={validator.ss58_address}, netuid=20, state=tmp_path)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/events', content=b'clip', headers=signed_headers(t,validator))
                assert response.status_code == status
                if status != 200: continue
                signature = base64.b64decode(response.headers['x-witness-response-signature'])
                verify(miner.ss58_address,response_message(t,response.content),signature)
                for changed in (response_message(t,b'changed'), response_message(task(validator,miner),response.content)):
                    with pytest.raises(ValueError): verify(miner.ss58_address,changed,signature)
                with pytest.raises(ValueError): verify(validator.ss58_address,response_message(t,response.content),signature)
    asyncio.run(run())


@pytest.mark.parametrize('mutation',['miner','validator','netuid','body','size','extra','expired'])
def test_identity_and_integrity_fail_before_handler(tmp_path, mutation):
    async def run():
        validator, miner = identities(); entered=[]; t=task(validator,miner)
        async def handler(*args): entered.append(True); return {}
        data=t.model_dump()
        if mutation in ('miner','validator'): data[mutation+'_hotkey']=Keypair.create_from_uri('//Charlie').ss58_address
        if mutation=='netuid': data['netuid']=21
        if mutation=='size': data['clip_bytes']=129*1024**2
        if mutation=='extra': data['original']='private-canary'
        if mutation=='expired': data['expires_at']=time.time()-1
        body=canonical_bytes(data);headers=signed_headers(t,validator)
        headers.update({'x-witness-task':base64.b64encode(body).decode(),
                        'x-witness-signature':base64.b64encode(validator.sign(REQUEST_DOMAIN+body)).decode()})
        app=create_app(handler,hotkey=miner,validator_hotkeys={validator.ss58_address},netuid=20,state=tmp_path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            r=await client.post('/events',content=b'evil' if mutation=='body' else b'clip',headers=headers)
        assert r.status_code in (400,422) and not entered and 'private-canary' not in r.text
    asyncio.run(run())
