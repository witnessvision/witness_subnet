"""Jobs an evaluator runs on its GPU: submitted models answering clips, and pool audio evidence.

Only validator files go to the GPU: ``pod_setup.sh`` (pinned environments from
``pod_env/*.txt``), ``pod_runtime.py`` (models) and ``pod_audio.py`` (evidence).
Every job runs under ``timeout`` on the GPU itself, so a slow model can never
keep running into the next one's time.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading

from witness.storage import write_private
from .contract import Execution, InfrastructureError, Policy, Task, prompt
from .gpu import DEFAULT_IMAGE, GPU_TYPES, Gpu
from .submission import verify_directory

HERE = Path(__file__).resolve().parent
ENVIRONMENTS = {"salmonn2-pro": "salmonn", "qwen2.5-omni": "omni", "qwen3-omni": "omni"}
MAX_NEW_TOKENS = 4096
TIMED_OUT = 124  # exit status of coreutils ``timeout``


def runtime_identity(config: dict) -> dict:
    """What decides a model's answers and timing: validator runtime files, image and GPU."""
    return {"runtime": (HERE / "pod_runtime.py").read_text(), "setup": (HERE / "pod_setup.sh").read_text(),
            "audio": (HERE / "pod_audio.py").read_text(),
            "environments": {path.name: path.read_text() for path in sorted((HERE / "pod_env").glob("*.txt"))},
            "backend": config.get("backend", "runpod"), "image": config.get("image", DEFAULT_IMAGE),
            "gpu": config.get("gpu_type_ids", GPU_TYPES)}


def prepare(gpu: Gpu, job: str) -> str:
    """A running GPU with the validator's files and pinned environments; returns the job directory."""
    gpu.ensure()
    workspace = gpu.workspace
    remote = f"{workspace}/jobs/{job}"
    if gpu.run(["mkdir", "-p", remote + "/clips", workspace + "/pod_env"], timeout=60).returncode:
        raise InfrastructureError("gpu_command_failed")
    gpu.put(sorted((HERE / "pod_env").glob("*.txt")), f"{workspace}/pod_env/")
    gpu.put([HERE / "pod_runtime.py", HERE / "pod_audio.py"], f"{workspace}/")
    if gpu.run(["bash", "-s"], timeout=3600, stdin=(HERE / "pod_setup.sh").read_text()).returncode:
        raise InfrastructureError("gpu_setup_failed")
    return remote


def _run_job(gpu: Gpu, script: str, environment: str, spec: dict, local: Path, remote: str,
             seconds: float, *, cancelled=lambda: False) -> tuple[int, str]:
    """Send one job spec, run ``script`` on it for at most ``seconds``; return (exit status, output lines)."""
    cancel_path = f'{remote}/{local.stem}.cancel'
    if script == 'pod_runtime.py':
        spec = {**spec, 'cancel_path': cancel_path, 'job_timeout_s': seconds}
    write_private(local, spec)
    gpu.put([local], f"{remote}/{local.name}")
    workspace = gpu.workspace
    output = f"{remote}/{local.stem}.out.jsonl"
    command = ["timeout", "--kill-after=5", str(max(1, int(seconds) + 2)), "env",
               f"HF_HOME={workspace}/hf", f"{workspace}/envs/{environment}/bin/python", f"{workspace}/{script}",
               f"{remote}/{local.name}", output]
    done = threading.Event()
    def watch():
        while not done.wait(.25):
            if cancelled():
                try:
                    gpu.run(['touch', cancel_path], timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass  # The GPU-side absolute deadline remains the backstop.
                return
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        result = gpu.run(command, timeout=seconds + 10)
        status, stderr = result.returncode, result.stderr
    except subprocess.TimeoutExpired:
        status, stderr = TIMED_OUT, "ssh_timeout"
    finally:
        done.set()
        watcher.join(timeout=6)
    write_private(local.with_suffix(".log.json"), {"returncode": status, "stderr": stderr[-4000:]})
    if "gpu_unhealthy" in stderr:
        gpu.discard()
        raise InfrastructureError("gpu_unhealthy")
    if status == 75 or cancelled():
        raise InfrastructureError('evaluation_cancelled_or_budget_expired')
    gpu.ensure()  # the GPU itself must still be there, or the whole evaluation is retried
    gpu.touch()
    return status, gpu.read(output)


def audio_evidence(gpu: Gpu, clips: list[dict], job: str) -> dict[str, dict]:
    """Speech and sound evidence for pool clips (``{"file", "path", "duration"}``), keyed by file name."""
    remote = prepare(gpu, job)
    gpu.put([Path(clip["path"]) for clip in clips], remote + "/clips/")
    spec = {"clips": [{"file": clip["file"], "path": f"{remote}/clips/{clip['file']}", "duration": clip["duration"]}
                      for clip in clips]}
    status, lines = _run_job(gpu, "pod_audio.py", "omni", spec, gpu.root / "jobs" / job / "audio.json", remote,
                             600 + 30 * len(clips))
    if status:
        raise InfrastructureError("gpu_audio_evidence_failed")
    return {row["file"]: row for row in map(json.loads, lines.splitlines())}


class PodRunner:
    """``runner(models, tasks, paths)`` for ``duel.run_duel``: each model answers every clip on the GPU."""

    def __init__(self, gpu: Gpu, submissions: dict[str, dict], policy: Policy, job: str):
        self.gpu, self.submissions, self.policy, self.job = gpu, submissions, policy, job
        self.hardware_id = None
        self.cancelled = lambda: False
        self.remaining_s = lambda: 900.
        self.batch_index = 0
        self.remote = None
        self.verified = set()

    def __call__(self, models: list[str], tasks: list[Task], paths: list[Path]) -> dict[str, dict[str, Execution]]:
        if self.cancelled() or self.remaining_s() <= 0:
            raise InfrastructureError('evaluation_cancelled_or_budget_expired')
        if self.remote is None:
            self.remote = prepare(self.gpu, self.job)
        remote = self.remote
        self.batch_index += 1
        self.gpu.put(paths, remote + "/clips/")
        results = {}
        for model_id in models:
            submission = self.submissions[model_id]
            manifest, local = submission["manifest"], Path(submission["path"])
            verify_directory(local, manifest)
            weights = f"{self.gpu.workspace}/models/{model_id}"
            for row in ([] if model_id in self.verified else manifest["files"]):
                if self.cancelled():
                    raise InfrastructureError('evaluation_cancelled_or_budget_expired')
                target = f"{weights}/{row['name']}"
                if self.gpu.run(["mkdir", "-p", str(Path(target).parent)], timeout=30).returncode:
                    raise InfrastructureError("gpu_model_directory_failed")
                checked = self.gpu.run(["sha256sum", target], timeout=60)
                if checked.returncode or checked.stdout.split()[0] != row["sha256"]:
                    self.gpu.put([local / row["name"]], target)
                checked = self.gpu.run(["sha256sum", target], timeout=120)
                if checked.returncode or checked.stdout.split()[0] != row["sha256"]:
                    raise InfrastructureError("gpu_model_integrity_failed")
            self.verified.add(model_id)
            spec = {"arch": manifest["arch"], "weights": weights,
                    "max_new_tokens": MAX_NEW_TOKENS, "load_timeout_s": 120,
                    "tasks": [{"task_id": task.id, "path": f"{remote}/clips/{path.name}", "prompt": prompt(task.duration),
                               "deadline_s": self.policy.deadline_s(task.duration)}
                              for task, path in zip(tasks, paths)]}
            # Staged batches permit inference savings without reloading once per
            # video. Reload/warmup are bounded inside the attempt budget.
            seconds = min(self.remaining_s(), 120 + sum(self.policy.deadline_s(task.duration) for task in tasks)
                          + max(self.policy.deadline_s(task.duration) for task in tasks) + 5)
            if seconds <= 0 or self.cancelled():
                raise InfrastructureError('evaluation_cancelled_or_budget_expired')
            status, lines = _run_job(self.gpu, "pod_runtime.py", ENVIRONMENTS[manifest["arch"]], spec,
                                     self.gpu.root / "jobs" / self.job / f"{model_id}-{self.batch_index}.json",
                                     remote, seconds, cancelled=self.cancelled)
            # 0: clips attempted; 2: the weights cannot be loaded; TIMED_OUT: the model ran out of time.
            # Those score as invalid answers. Anything else is the validator's infrastructure: retried, never scored.
            if status not in (0, 2, TIMED_OUT):
                raise InfrastructureError(f"gpu_runtime_exit_{status}")
            rows = {row["task_id"]: row for row in map(json.loads, lines.splitlines())}
            identities = {r["hardware_id"] for r in rows.values() if r.get("hardware_id")}
            if len(identities) > 1:
                raise InfrastructureError("gpu_changed_during_job")
            self.hardware_id = next(iter(identities), None)
            results[model_id] = {task.id: self._execution(model_id, task, rows.get(task.id)) for task in tasks}
        return results

    def _execution(self, model_id: str, task: Task, row: dict | None) -> Execution:
        return Execution(task_id=task.id, model_id=model_id, checkpoint_hash=model_id, clip_sha256=task.clip_sha256,
                         runtime_hash=self.policy.runtime_hash, preprocessing_hash=self.policy.preprocessing_hash,
                         status="timeout" if row is None else row["status"],
                         elapsed_s=row["elapsed_s"] if row else self.policy.deadline_s(task.duration),
                         raw=(row or {}).get("raw", "")[:65536],
                         # The validator's loaders always feed both tracks of the clip.
                         audio_tokens=1, video_tokens=1)
