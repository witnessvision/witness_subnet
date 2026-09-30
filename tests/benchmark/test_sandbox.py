import json
import os
from pathlib import Path
import socket
import threading
import time
import pytest
from witness.benchmark.sandbox import Broker,request_for,run_isolated
from witness.benchmark.contract import InfrastructureError
from witness.benchmark.p2p import Auth

MODEL='a'*64

def layout(tmp_path):
    for name in ('models/'+MODEL,'jobs/test/clips','envs/omni','hf','video-SALMONN-2'):
        (tmp_path/name).mkdir(parents=True)
    (tmp_path/'jobs/test/clips/clip.mp4').write_bytes(b'test')
    (tmp_path/'models'/MODEL/'model.safetensors').write_bytes(b'weights')
    data={'arch':'qwen2.5-omni','weights':'/var/lib/witness-gpu/models/'+MODEL,
          'tasks':[{'path':'/var/lib/witness-gpu/jobs/test/clips/clip.mp4'}],
          'cancel_path':'/var/lib/witness-gpu/jobs/test/spec.cancel'}
    (tmp_path/'jobs/test/spec.json').write_text(json.dumps(data))
    request={'job':'test','spec':'spec.json','environment':'omni','script':'pod_runtime.py','seconds':60}
    return Broker(workspace=tmp_path,image='sha256:'+'b'*64,gpu='GPU-test',uid=os.getuid(),gid=os.getgid()),request


def test_sandbox_mounts_only_current_model_no_network_capabilities_or_signer(tmp_path):
    broker,request=layout(tmp_path)
    plan=broker.plan(request,'test-container')
    assert '--privileged' not in plan and '--network' in plan and plan[plan.index('--network')+1]=='none'
    assert plan[plan.index('--cap-drop')+1]=='ALL' and '--read-only' in plan
    mounts=[plan[i+1] for i,x in enumerate(plan) if x=='--mount']
    assert len([m for m in mounts if ',readonly' not in m])==1
    assert not any(x in m for m in mounts for x in ('docker.sock','credstore','validator/mainnet','/etc/witness'))
    assert any('/models/'+MODEL in m and m.endswith(',readonly') for m in mounts)
    assert not any('target=/var/lib/witness-gpu/models,' in m for m in mounts)
    assert 'HF_HUB_OFFLINE=1' in plan and '--pids-limit' in plan and '--memory' in plan


@pytest.mark.parametrize('field,value',[('job','../private'),('spec','../../secret.json'),('environment','evil'),
                                       ('script','/bin/sh'),('seconds',float('nan')),('seconds',999999)])
def test_broker_rejects_arbitrary_execution_and_paths(tmp_path,field,value):
    broker,request=layout(tmp_path);request[field]=value
    with pytest.raises(ValueError):broker.plan(request,'test-container')


def test_broker_rejects_symlink_and_hardlink_output_without_truncation(tmp_path):
    broker,request=layout(tmp_path)
    secret=tmp_path/'private.json';secret.write_bytes(b'keep')
    target=tmp_path/'jobs/test/spec.out.jsonl';target.symlink_to(secret)
    with pytest.raises(ValueError):broker.plan(request,'test-container')
    assert secret.read_bytes()==b'keep'
    target.unlink();target.hardlink_to(secret)
    with pytest.raises(ValueError):broker.plan(request,'test-container')
    assert secret.read_bytes()==b'keep'


def test_command_parser_and_unavailable_broker_fail_closed(tmp_path):
    w='/var/lib/witness-gpu'
    command=['timeout','--kill-after=5','60','env',f'HF_HOME={w}/hf',f'{w}/envs/omni/bin/python',
             f'{w}/pod_runtime.py',f'{w}/jobs/test/spec.json',f'{w}/jobs/test/spec.out.jsonl']
    assert request_for(command,w)['job']=='test'
    with pytest.raises(InfrastructureError):request_for(['sh','-c','echo stolen'],w)
    with pytest.raises(InfrastructureError):
        run_isolated(tmp_path/'missing.sock',request_for(command,w),command,timeout=1,cancelled=lambda:False,remaining_s=lambda:1)


def test_stake_floor_cannot_be_disabled():
    for minimum in (0,99999,-1,False):
        with pytest.raises(ValueError,match='stake_floor'):Auth('receiver',lambda:{},min_stake_alpha=minimum)


def test_local_inference_requires_sandbox_no_fallback(tmp_path):
    from witness.benchmark.gpu import LocalGpu
    gpu=LocalGpu({'workspace':str(tmp_path/'gpu')},tmp_path)
    with pytest.raises(InfrastructureError,match='sandbox_required'):
        gpu.run(['python','/any/pod_runtime.py','spec','output'],timeout=1)


def test_cancellation_waits_for_worker_cleanup_ack(tmp_path):
    address=tmp_path/'test.sock'
    listener=socket.socket(socket.AF_UNIX);listener.bind(str(address));listener.listen()
    cleaned=threading.Event()
    def worker():
        with listener.accept()[0] as connection:
            request=b''
            while b'\n' not in request:request+=connection.recv(1024)
            assert connection.recv(1024)==b'cancel\n'
            time.sleep(.05);cleaned.set()
            connection.sendall(b'{"error":"cancelled"}\n')
        listener.close()
    thread=threading.Thread(target=worker);thread.start()
    started=time.monotonic()
    with pytest.raises(InterruptedError):
        run_isolated(address,{},[],timeout=2,cancelled=lambda:time.monotonic()-started>.1,remaining_s=lambda:2)
    assert cleaned.is_set();thread.join(timeout=2)
