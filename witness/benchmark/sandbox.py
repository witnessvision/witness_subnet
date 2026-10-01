"""Narrow Unix-socket broker for isolated local GPU jobs.

The signer never receives Docker access. The root-owned broker accepts only a
validator-owned runtime/environment and bounded job paths, never shell commands,
images, mounts or environment variables supplied by the client.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import socketserver
import struct
import subprocess
import threading
import time
import uuid

from .contract import InfrastructureError

MAX_MESSAGE = 65536
ENVIRONMENTS = {'salmonn2-pro': 'salmonn', 'qwen2.5-omni': 'omni', 'qwen3-omni': 'omni'}
IDENTIFIER = re.compile(r'[A-Za-z0-9_-]{1,160}')


def request_for(command, workspace):
    """Recognize the exact runner invocation; fail closed on an altered command."""
    if (len(command) != 9 or command[:2] != ['timeout', '--kill-after=5']
            or command[3:5] != ['env', f'HF_HOME={workspace}/hf']):
        raise InfrastructureError('gpu_sandbox_command_denied')
    try:
        environment = str(Path(command[5]).relative_to(Path(workspace)/'envs'))
        if environment not in ('salmonn/bin/python', 'omni/bin/python'):
            raise ValueError()
        script = str(Path(command[6]).relative_to(workspace))
        if script not in ('pod_runtime.py', 'pod_audio.py'):
            raise ValueError()
        spec = Path(command[7]).relative_to(Path(workspace)/'jobs')
        output = Path(command[8]).relative_to(Path(workspace)/'jobs')
        if len(spec.parts) != 2 or output != spec.with_suffix('.out.jsonl'):
            raise ValueError()
        return {'job': spec.parts[0], 'spec': spec.name, 'environment': environment.split('/')[0],
                'script': script, 'seconds': float(command[2])}
    except (TypeError, ValueError):
        raise InfrastructureError('gpu_sandbox_command_denied') from None


def run_isolated(address, request, command, *, timeout, cancelled, remaining_s):
    until = time.monotonic() + min(timeout, remaining_s())
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(2)
            s.connect(str(address))
            s.sendall(json.dumps(request).encode()+b'\n')
            received = bytearray()
            stopping = False
            while b'\n' not in received:
                if not stopping and (cancelled() or time.monotonic() >= until):
                    # Wait for the broker to kill/reap the container before the
                    # caller checkpoints flushed answers or starts another job.
                    s.sendall(b'cancel\n')
                    stopping = True
                    until = time.monotonic() + 30
                elif stopping and time.monotonic() >= until:
                    raise InfrastructureError('gpu_sandbox_cleanup_unconfirmed')
                s.settimeout(min(.2, max(.001, until-time.monotonic())))
                try:
                    chunk = s.recv(8192)
                except socket.timeout:
                    continue
                if not chunk or len(received)+len(chunk) > MAX_MESSAGE:
                    raise InfrastructureError('gpu_sandbox_invalid_response')
                received.extend(chunk)
            response = json.loads(received)
            if stopping:
                raise InterruptedError('evaluation_cancelled_or_budget_expired')
            if response.get('error'):
                raise InfrastructureError('gpu_sandbox_'+response['error'])
            return subprocess.CompletedProcess(command, int(response['returncode']), '', response.get('stderr', ''))
    except InterruptedError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise InfrastructureError('gpu_sandbox_unavailable') from error


def checked_path(root, relative, *, directory=False):
    path = root / relative
    if path.resolve(strict=True) != path.absolute() or not path.is_relative_to(root):
        raise ValueError('sandbox_path_denied')
    if directory != path.is_dir() or (not directory and not path.is_file()):
        raise ValueError('sandbox_path_denied')
    return path


class Broker:
    def __init__(self, *, workspace, image, gpu, uid=999, gid=996, memory='64g', cpus=12):
        self.workspace = Path(workspace).resolve(strict=True)
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('sandbox_requires_pinned_image_id')
        self.image, self.gpu, self.uid, self.gid = image, gpu, uid, gid
        self.memory, self.cpus = memory, cpus
        self.slot = threading.Lock()
        self.owner = "worker.sock"

    def plan(self, request, name):
        if not isinstance(request, dict) or set(request) != {'job','spec','environment','script','seconds'}:
            raise ValueError('sandbox_invalid_request')
        job, spec = request['job'], request['spec']
        if (not isinstance(job, str) or not IDENTIFIER.fullmatch(job)
                or not isinstance(spec, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}\.json', spec)
                or request['script'] not in ('pod_runtime.py','pod_audio.py')
                or request['environment'] not in ('salmonn','omni')
                or type(request['seconds']) not in (int,float) or not math.isfinite(request['seconds'])
                or not 1 <= request['seconds'] <= 2705):
            raise ValueError('sandbox_invalid_request')
        workspace = self.workspace
        folder = checked_path(workspace, f'jobs/{job}', directory=True)
        specification = checked_path(workspace, f'jobs/{job}/{spec}')
        if specification.stat().st_size > MAX_MESSAGE:
            raise ValueError('sandbox_spec_too_large')
        data = json.loads(specification.read_text())
        environment = checked_path(workspace, 'envs/'+request['environment'], directory=True)
        inside = Path('/var/lib/witness-gpu')
        mounts = [(environment, inside/'envs'/request['environment'], True),
                  (folder, inside/'jobs'/job, True)]
        if request['script'] == 'pod_runtime.py':
            if data.get('arch') not in ENVIRONMENTS or ENVIRONMENTS[data['arch']] != request['environment']:
                raise ValueError('sandbox_invalid_architecture')
            weights = Path(data.get('weights',''))
            if weights.parent != inside/'models' or not re.fullmatch(r'[0-9a-f]{64}', weights.name):
                raise ValueError('sandbox_invalid_model')
            mounts.append((checked_path(workspace, 'models/'+weights.name, directory=True), weights, True))
            entries = data.get('tasks')
            expected_cancel = str(inside/'jobs'/job/(Path(spec).stem+'.cancel'))
            if data.get('cancel_path') != expected_cancel:
                raise ValueError('sandbox_invalid_cancel_path')
            if 'pause_after_tasks' in data or 'continue_path' in data:
                expected_continue = str(inside/'jobs'/job/(Path(spec).stem+'.continue'))
                if (type(data.get('pause_after_tasks')) is not int or
                        not 0 < data['pause_after_tasks'] < len(entries) or
                        data.get('continue_path') != expected_continue):
                    raise ValueError('sandbox_invalid_continuation')

        else:
            if request['environment'] != 'omni':
                raise ValueError('sandbox_invalid_architecture')
            entries = data.get('clips')
        if not isinstance(entries, list) or not 1 <= len(entries) <= 100:
            raise ValueError('sandbox_invalid_tasks')
        for entry in entries:
            path = Path(entry.get('path',''))
            if path.parent != inside/'jobs'/job/'clips' or not re.fullmatch(r'[A-Za-z0-9_.-]{1,240}',path.name):
                raise ValueError('sandbox_invalid_clip')
            checked_path(workspace, f'jobs/{job}/clips/{path.name}')
        for extra in ('hf','video-SALMONN-2'):
            mounts.append((checked_path(workspace, extra, directory=True), inside/extra, True))
        output = folder/(Path(spec).stem+'.out.jsonl')
        if output.is_symlink():
            raise ValueError('sandbox_output_symlink')
        fd = os.open(output, os.O_CREAT|os.O_WRONLY|os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(fd).st_nlink != 1:
                raise ValueError('sandbox_output_hardlink')
            os.fchown(fd,self.uid,self.gid)
            os.ftruncate(fd,0)
        finally:
            os.close(fd)
        mounts.append((output, inside/'jobs'/job/output.name, False))
        args = ['docker','run','--detach','--name',name,'--label','witness.sandbox.owner='+self.owner,
                '--pull','never','--network','none',
                '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges=true',
                '--user',f'{self.uid}:{self.gid}','--pids-limit','512','--memory',self.memory,
                '--memory-swap',self.memory,'--cpus',str(self.cpus),
                '--cpuset-cpus',','.join(map(str,sorted(os.sched_getaffinity(0))[:self.cpus])),
                '--shm-size','2g',
                '--ulimit','fsize=16777216:16777216','--ulimit','nofile=4096:4096',
                '--tmpfs','/tmp:rw,nosuid,nodev,size=4g,mode=1777',
                '--log-driver','local','--log-opt','max-size=1m','--log-opt','max-file=2',
                '--gpus',f'"device={self.gpu}"']
        for source,target,readonly in mounts:
            args.extend(['--mount',f'type=bind,source={source},target={target}'+(',readonly' if readonly else '')])
        env = {'HOME':'/tmp','TMPDIR':'/tmp','WITNESS_GPU_WORKSPACE':str(inside),
               'HF_HOME':str(inside/'hf'),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
               'PYTHONPATH':'/opt/witness/source/witness/benchmark/sandbox_env',
               'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false',
               'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
               'RAYON_NUM_THREADS':'4','TRITON_CACHE_DIR':'/tmp/triton',
               'TORCH_EXTENSIONS_DIR':'/tmp/torch_extensions'}
        for k,v in env.items():args.extend(['-e',f'{k}={v}'])
        args.extend(['--entrypoint',str(inside/'envs'/request['environment']/'bin/python'),self.image,
                     '/opt/witness/source/witness/benchmark/'+request['script'],
                     str(inside/'jobs'/job/spec),str(inside/'jobs'/job/output.name)])
        return args

    def execute(self, request, connection):
        if not self.slot.acquire(blocking=False):
            return {'error':'busy'}
        name = 'witness-inference-'+uuid.uuid4().hex
        try:
            args = self.plan(request,name)
            subprocess.run(args,check=True,capture_output=True,timeout=30)
            until=time.monotonic()+request['seconds']
            while True:
                result = subprocess.run(['docker','inspect','--format','{{json .State}}',name],
                                        capture_output=True,text=True,check=True,timeout=10)
                state=json.loads(result.stdout)
                if not state['Running']:
                    # Logs are diagnostics only and never returned unbounded.
                    logs=subprocess.Popen(['docker','logs','--tail','50',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    try:
                        stderr=logs.stdout.read(8192).decode(errors='replace')
                    finally:
                        if logs.poll() is None:logs.kill()
                        logs.wait(timeout=5)
                    return {'returncode':state['ExitCode'],'stderr':stderr[-4000:]}
                if time.monotonic() >= until:
                    return {'returncode':124,'stderr':'sandbox_job_timeout'}
                if select.select([connection],[],[],.2)[0]:
                    if not connection.recv(1,socket.MSG_PEEK):
                        return {'error':'client_disconnected'}
                    if connection.recv(16) == b'cancel\n':
                        return {'error':'cancelled'}
                    return {'error':'unexpected_client_data'}
        except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError):
            return {'error':'job_failed'}
        finally:
            try:
                subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
            finally:
                self.slot.release()


def serve(address, broker):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            _,uid,_=struct.unpack('3i',self.request.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
            if uid != broker.uid:return
            self.request.settimeout(2)
            buffer=bytearray()
            while b'\n' not in buffer:
                chunk=self.request.recv(8192)
                if not chunk or len(buffer)+len(chunk)>MAX_MESSAGE:return
                buffer.extend(chunk)
            try:
                request=json.loads(buffer)
                result=broker.execute(request,self.request)
                self.request.sendall(json.dumps(result).encode()+b'\n')
            except (ValueError,OSError):
                return
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads=True
    address=Path(address)
    broker.owner=address.name
    # A broker crash must not leave an orphan consuming the GPU after restart.
    orphaned=subprocess.check_output(['docker','ps','-aq','--filter','label=witness.sandbox.owner='+broker.owner],text=True,timeout=10).split()
    if orphaned:
        subprocess.run(['docker','rm','-f',*orphaned],check=True,timeout=30,stdout=subprocess.DEVNULL)
    address.unlink(missing_ok=True)
    with Server(str(address),Handler) as server:
        address.chmod(0o600);os.chown(address,broker.uid,broker.gid)
        print('GPU sandbox broker ready',flush=True)
        server.serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--gpu',required=True)
    args=p.parse_args()
    serve(args.socket,Broker(workspace=args.workspace,image=args.image,gpu=args.gpu))

if __name__=='__main__':main()
