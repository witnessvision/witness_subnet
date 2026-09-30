"""Where an evaluator gets its GPU: this machine, an SSH host, or an on-demand RunPod pod.

Every backend offers the small interface ``runner.py`` uses: ``ensure`` (a
reachable GPU), ``run``, ``put``, ``read``, ``touch`` (it was just used),
``stop_if_idle`` and ``discard`` (drop a broken GPU). Choose one in the
evaluator configuration under ``"gpu": {"backend": ...}``:

* ``local`` — the evaluator itself runs on a CUDA machine (``workspace``);
* ``ssh`` — a GPU host the operator keeps running (``host``, ``port``, ``ssh_key``);
* ``runpod`` — a pod started when work arrives and stopped after ``idle_stop_s``
  without jobs, with a UTC-day cost cap (``RUNPOD_API_KEY``). With
  ``network_volume_id`` and ``data_center_id`` pods are placed next to that
  volume first so the workspace survives replacement; when that data centre has
  no free GPU, a pod anywhere with its own disk is used and deleted when idle.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import httpx

from witness.storage import write_private
from .contract import InfrastructureError

# Model/media children never inherit the validator's provider or wallet credentials.
GPU_ENVIRONMENT = frozenset({
    'PATH', 'HOME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TMPDIR', 'LD_LIBRARY_PATH',
    'CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
    'NVIDIA_DRIVER_CAPABILITIES', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
    'CUDA_MODULE_LOADING', 'CUBLAS_WORKSPACE_CONFIG', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
    'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'TOKENIZERS_PARALLELISM',
    'PYTHONUNBUFFERED', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE',
    'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE',
    'TRANSFORMERS_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME',
})


def gpu_environment(workspace):
    return {**{k: v for k, v in os.environ.items() if k in GPU_ENVIRONMENT},
            'WITNESS_GPU_WORKSPACE': workspace}


DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
# 48 GB cards at similar prices, in order of preference.
GPU_TYPES = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"]
# Waiting for capacity or budget is not a failed evaluation attempt.
UNAVAILABLE = ("gpu_no_capacity", "gpu_daily_cost_cap_reached")


class Gpu:
    """A GPU that is always there: nothing to start, stop or replace."""
    workspace: str

    def __init__(self, config: dict, root: Path):
        self.config, self.root = config, root

    def ensure(self) -> None:
        pass

    def touch(self) -> None:
        pass

    def stop_if_idle(self) -> bool:
        return False

    def enforce_budget(self) -> bool:
        return False

    def discard(self) -> None:
        pass

    def lease(self):
        return nullcontext()

    @staticmethod
    def checked_result(command, result):
        # The model supervisor reports generic preprocessing exceptions as
        # invalid answers. Host decoder exhaustion is infrastructure instead.
        if any(str(arg).endswith('/pod_runtime.py') or arg == 'pod_runtime.py' for arg in command):
            stderr = result.stderr or ''
            if ('Resource temporarily unavailable' in stderr
                    and ('video_reader_backend' in stderr or '[swscaler]' in stderr)):
                raise InfrastructureError('gpu_video_decoder_resource_exhausted')
        return result


class LocalGpu(Gpu):
    def __init__(self, config: dict, root: Path):
        super().__init__(config, root)
        self.workspace = str(Path(config.get("workspace", root / "gpu")).resolve())
        Path(self.workspace).mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cancelled = lambda: False
        self.remaining_s = lambda: float('inf')

    @contextmanager
    def lease(self):
        """Other local GPU jobs must share this lock; never kill their processes."""
        with (Path(self.workspace) / 'gpu.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise InfrastructureError('gpu_busy') from error
            probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=10)
            if probe.returncode:
                raise InfrastructureError('gpu_unavailable')
            if probe.stdout.strip():
                raise InfrastructureError('gpu_busy')
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def run(self, command: list[str], *, timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
        if any(str(arg).endswith('/pod_runtime.py') or str(arg).endswith('/pod_audio.py') for arg in command):
            if not self.config.get('sandbox_socket'):
                raise InfrastructureError('gpu_sandbox_required')
            from .sandbox import request_for, run_isolated
            request = request_for(command, self.workspace)
            result = run_isolated(self.config['sandbox_socket'], request, command, timeout=timeout,
                                  cancelled=self.cancelled, remaining_s=self.remaining_s)
            return self.checked_result(command, result)
        from .execution import run_process
        result = run_process(command, input=stdin, text=True, timeout=timeout,
                             cancelled=self.cancelled, remaining_s=self.remaining_s,
                             env=gpu_environment(self.workspace))
        return self.checked_result(command, result)

    def put(self, sources: list[Path], destination: str) -> None:
        for source in sources:
            target = Path(destination) / source.name if Path(destination).is_dir() else Path(destination)
            if source.resolve() == target.resolve():
                continue
            partial = target.with_name(target.name + '.copying')
            try:
                with source.open('rb') as src, partial.open('wb') as dst:
                    while chunk := src.read(1024 * 1024):
                        if self.cancelled() or self.remaining_s() <= 0:
                            raise InfrastructureError('evaluation_cancelled_or_budget_expired')
                        dst.write(chunk)
                partial.chmod(0o600)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)

    def read(self, path: str) -> str:
        return Path(path).read_text() if Path(path).exists() else ""

    def remove_models(self, models: list[str]) -> list[str]:
        from .model_cache import remove_models
        return remove_models(Path(self.workspace) / 'models', models)


class SshGpu(Gpu):
    def __init__(self, config: dict, root: Path):
        super().__init__(config, root)
        self.ssh_key = Path(config["ssh_key"])
        self.workspace = config.get("workspace", "/workspace")
        self.address = (config.get("host"), int(config.get("port", 22)))

    def ensure(self) -> None:
        if self.run(["true"], timeout=30).returncode:
            raise InfrastructureError("gpu_host_unreachable")

    def _options(self) -> list[str]:
        return ["-i", str(self.ssh_key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={self.root / 'gpu_known_hosts'}",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]

    def run(self, command: list[str], *, timeout: float, stdin: str | None = None) -> subprocess.CompletedProcess:
        host, port = self.address
        remote = f"WITNESS_GPU_WORKSPACE={shlex.quote(self.workspace)} " + " ".join(map(shlex.quote, command))
        result = subprocess.run(["ssh", *self._options(), "-p", str(port), f"root@{host}", remote],
                                input=stdin, capture_output=True, text=True, timeout=timeout)
        return self.checked_result(command, result)

    def put(self, sources: list[Path], destination: str) -> None:
        host, port = self.address
        result = subprocess.run(["scp", *self._options(), "-P", str(port), *map(str, sources),
                                 f"root@{host}:{destination}"], capture_output=True, timeout=900)
        if result.returncode:
            raise InfrastructureError("gpu_copy_failed")

    def read(self, path: str) -> str:
        result = self.run(["cat", path], timeout=120)
        return result.stdout if result.returncode == 0 else ""

    def remove_models(self, models: list[str]) -> list[str]:
        # The helper needs only stdlib; do not install packages or start a GPU.
        source = Path(__file__).with_name('model_cache.py').read_text()
        result = self.run(['python3', '-', str(Path(self.workspace) / 'models'), *models],
                          stdin=source, timeout=30)
        if result.returncode:
            raise InfrastructureError('gpu_model_cleanup_failed')
        return json.loads(result.stdout)


class RunPodGpu(SshGpu):
    REST = "https://rest.runpod.io/v1"

    def __init__(self, config: dict, root: Path):
        super().__init__(config, root)
        self.key = os.environ.get("RUNPOD_API_KEY")
        if not self.key:
            raise ValueError("RUNPOD_API_KEY_required")
        self.state_path = root / "gpu.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def _save(self) -> None:
        write_private(self.state_path, self.state)

    def remove_models(self, models: list[str]) -> list[str]:
        if not self.state.get('running_since') or not self.state.get('host'):
            raise InfrastructureError('gpu_model_cleanup_waiting_for_running_host')
        self.address = (self.state['host'], int(self.state['port']))
        return super().remove_models(models)

    def _rest(self, method: str, path: str, body: dict | None = None):
        response = httpx.request(method, self.REST + path, json=body, timeout=60,
                                 headers={"Authorization": "Bearer " + self.key})
        if response.status_code >= 400:
            raise InfrastructureError(f"runpod_{method.lower()}_{response.status_code}")
        return response.json() if response.content else {}

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def spend_today(self) -> float:
        running = self.state.get("running_since")
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        live = (time.time() - max(running, midnight)) / 3600 * self.state.get("cost_per_hr", 0.) if running else 0.
        return self.state.get("spend", {}).get(self._today(), 0.) + live

    def _stop_clock(self) -> None:
        """Settle each UTC day separately, including a run crossing midnight."""
        running = self.state.pop("running_since", None)
        if running:
            spend = self.state.setdefault("spend", {})
            end = time.time()
            while running < end:
                day = datetime.fromtimestamp(running, timezone.utc)
                boundary = day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() + 86400
                until = min(end, boundary)
                key = day.strftime("%Y-%m-%d")
                spend[key] = spend.get(key, 0.) + (until - running) / 3600 * self.state.get("cost_per_hr", 0.)
                running = until

    def _forget(self) -> None:
        for key in ("pod_id", "gpu", "cloud", "network_volume", "host", "port"):
            self.state.pop(key, None)
        self._save()

    def _delete(self) -> None:
        self._rest("DELETE", f"/pods/{self.state['pod_id']}")
        self._stop_clock()
        self._forget()

    def _create(self) -> None:
        """Place a pod next to the network volume first (cached workspace), then anywhere.

        Both models of an evaluation always share the pod, so their timing stays paired.
        """
        gpu_types = self.config.get("gpu_type_ids", GPU_TYPES)
        placements = [("SECURE", gpu, True) for gpu in gpu_types] if self.config.get("network_volume_id") else []
        placements += [(cloud, gpu, False) for cloud in self.config.get("cloud_types", ["SECURE", "COMMUNITY"])
                       for gpu in gpu_types]
        for cloud, gpu, on_volume in placements:
            body = {"name": self.config.get("name", "witness-evaluator"),
                    "imageName": self.config.get("image", DEFAULT_IMAGE), "gpuTypeIds": [gpu], "gpuCount": 1,
                    "cloudType": cloud, "containerDiskInGb": self.config.get("container_disk_gb", 40),
                    "ports": ["22/tcp"], "supportPublicIp": True, "volumeMountPath": "/workspace",
                    "env": {"PUBLIC_KEY": self.ssh_key.with_suffix(".pub").read_text().strip()}}
            if on_volume:
                body.update(networkVolumeId=self.config["network_volume_id"],
                            dataCenterIds=[self.config["data_center_id"]])
            else:
                body["volumeInGb"] = self.config.get("volume_gb", 150)
            try:
                pod = self._rest("POST", "/pods", body)
            except InfrastructureError:
                continue  # no capacity for this placement
            self.state.update(pod_id=pod["id"], gpu=gpu, cloud=cloud, network_volume=on_volume,
                              created_unix=time.time(), started_unix=time.time(), owner="witness-mainnet-2")
            self._save()
            return
        raise InfrastructureError("gpu_no_capacity")

    def ensure(self) -> None:
        if (self.root / 'gpu-watchdog-stop.json').exists():
            raise InfrastructureError('gpu_watchdog_latched_operator_reconciliation_required')
        if self.spend_today() >= self.config.get("daily_cap_usd", 10.):
            self.enforce_budget()
            raise InfrastructureError("gpu_daily_cost_cap_reached")
        pod = None
        if self.state.get("pod_id"):
            try:
                pod = self._rest("GET", f"/pods/{self.state['pod_id']}")
            except InfrastructureError as error:
                if str(error) != "runpod_get_404":
                    raise
                self._forget()  # deleted outside the evaluator
        if pod is None:
            self._create()
        elif pod.get("desiredStatus") != "RUNNING":
            try:
                self._rest("POST", f"/pods/{self.state['pod_id']}/start")
                self.state['started_unix'] = time.time()
                self._save()
            except InfrastructureError:
                self._delete()  # a stopped pod is pinned to its host; if that host is full, replace it
                self._create()
        deadline = time.monotonic() + self.config.get("ready_timeout_s", 900)
        while time.monotonic() < deadline:
            if (self.root / 'gpu-watchdog-stop.json').exists():
                raise InfrastructureError('gpu_watchdog_latched_operator_reconciliation_required')
            pod = self._rest("GET", f"/pods/{self.state['pod_id']}")
            host, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
            if pod.get("desiredStatus") == "RUNNING":
                if "running_since" not in self.state:
                    price = float(pod.get("costPerHr") or 0.)
                    if price <= 0:
                        self._rest("POST", f"/pods/{self.state['pod_id']}/stop")
                        raise InfrastructureError("gpu_price_unknown")
                    self.state.update(running_since=min(time.time(), self.state.get('started_unix', time.time())), cost_per_hr=price)
                    self._save()
                self.enforce_budget()
                if self.spend_today() >= self.config.get('daily_cap_usd', 10.):
                    raise InfrastructureError('gpu_daily_cost_cap_reached')
            if pod.get("desiredStatus") == "RUNNING" and host and port:
                self.address = (host, int(port))
                self.state.update(host=host, port=int(port), last_used=time.time())
                self._save()
                if self.run(["true"], timeout=30).returncode == 0:
                    return
            time.sleep(10)
        raise InfrastructureError("gpu_pod_not_ready")

    def touch(self) -> None:
        self.state["last_used"] = time.time()
        self._save()

    def stop_if_idle(self) -> bool:
        if self.enforce_budget():
            return True
        if (not self.state.get("running_since")
                or time.time() - self.state.get("last_used", 0) <= self.config.get("idle_stop_s", 900)):
            return False
        if self.config.get("network_volume_id") and not self.state.get("network_volume"):
            self._delete()  # a fallback pod outside the volume's data centre: its own disk is not worth paying for
        else:
            self._rest("POST", f"/pods/{self.state['pod_id']}/stop")
            self._stop_clock()
            self._save()
        return True

    def enforce_budget(self) -> bool:
        """Stop owned running compute at its cap even when the queue is nonempty."""
        if (not self.state.get("running_since") or not self.state.get("pod_id")
                or self.spend_today() < self.config.get("daily_cap_usd", 10.)):
            return False
        self._rest("POST", f"/pods/{self.state['pod_id']}/stop")
        self._stop_clock()
        self.state["stopped_reason"] = "daily_cost_cap"
        self._save()
        return True

    def discard(self) -> None:
        """Delete a pod whose GPU is broken; the next attempt gets another host."""
        if self.state.get("pod_id"):
            self._delete()


BACKENDS = {"local": LocalGpu, "ssh": SshGpu, "runpod": RunPodGpu}


def make_gpu(config: dict, root: Path) -> Gpu:
    return BACKENDS[config.get("backend", "runpod")](config, root)
