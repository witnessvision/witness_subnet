"""GPU-side runner: one submitted model answers every clip of a job. Validator code only.

Copied to the pod as a single file together with ``pod_setup.sh``; it imports
nothing from the Witness package or from the miner's package. Verified local
weights are loaded offline from safetensors without ``trust_remote_code``. Time is measured per
clip, from inputs to decoded text; model download and load are excluded.

Usage: python pod_runtime.py JOB.json OUT.jsonl  (under WITNESS_GPU_WORKSPACE, default /workspace)
JOB = {"arch", "weights", "max_new_tokens", "load_timeout_s",
       "tasks": [{"task_id", "path", "prompt", "deadline_s"}]}
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import multiprocessing
import signal

WORKSPACE = Path(os.environ.get("WITNESS_GPU_WORKSPACE", "/workspace"))
MODELS = WORKSPACE / "models"
SALMONN_REPO = WORKSPACE / "video-SALMONN-2/video_SALMONN2_pro"


def salmonn(weights: Path, max_new_tokens: int):
    sys.path[:0] = [str(SALMONN_REPO), str(SALMONN_REPO / "scripts")]
    from transformers import AutoTokenizer
    from qwenvl.data.processing_qwen3_vl import Qwen3VLProcessor
    from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    import inference as reference  # the authors' preprocessing, pinned in pod_setup.sh
    sys.argv = ["inference.py", "--video", "x", "--model", str(weights), "--video-max-frame-pixels", "128000",
                "--max-pixels", "128000", "--video-max-frames", "900"]
    options = reference.parse_args()
    processor = Qwen3VLProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(weights, model_max_length=options.model_max_length,
                                              padding_side="right", use_fast=False,
                                              local_files_only=True, trust_remote_code=False)
    dataset = reference.build_dataset(options, tokenizer, processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(weights, attn_implementation="sdpa",
                                                            torch_dtype=torch.bfloat16,
                                                            use_safetensors=True, local_files_only=True, trust_remote_code=False).cuda().eval()

    def infer(path, prompt):
        inputs = reference.prepare_inputs(dataset, path, prompt, True, torch.device("cuda"))
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return infer


def _audio(path: str):
    """The clip's audio as 16 kHz mono float32, decoded directly (the helper's reader needs old librosa)."""
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def _omni(model, processor, generate):
    from qwen_omni_utils import process_mm_info

    def infer(path, prompt):
        conversation = [{"role": "user", "content": [
            {"type": "video", "video": path, "fps": 2.0, "max_pixels": 640 * 360},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        _, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        # The audio track is interleaved with the frames it belongs to (use_audio_in_video).
        inputs = processor(text=text, audio=[_audio(path)], images=images, videos=videos, return_tensors="pt",
                           padding=True, use_audio_in_video=True).to(model.device).to(model.dtype)
        out = generate(inputs)
        ids = out[0] if isinstance(out, tuple) else out
        ids = getattr(ids, "sequences", ids)
        return processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return infer


def qwen3_omni(weights: Path, max_new_tokens: int):
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    # Text output only: the speech talker is never built.
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        weights, dtype="auto", device_map="cuda", attn_implementation="sdpa", enable_audio_output=False,
        use_safetensors=True, local_files_only=True, trust_remote_code=False).eval()
    return _omni(model, Qwen3OmniMoeProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False), lambda inputs: model.generate(
        **inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=False))


def qwen25_omni(weights: Path, max_new_tokens: int):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    # Only the text "thinker": the talker and its pickled speaker file are never loaded.
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        weights, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", use_safetensors=True, local_files_only=True, trust_remote_code=False).eval()
    return _omni(model, Qwen2_5OmniProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False), lambda inputs: model.generate(
        **inputs, use_audio_in_video=True, max_new_tokens=max_new_tokens, do_sample=False))


LOADERS = {"salmonn2-pro": salmonn, "qwen2.5-omni": qwen25_omni, "qwen3-omni": qwen3_omni}


def _worker(job, pipe):
    global torch
    import torch
    try:
        torch.cuda.init()
        (torch.ones(32, 32, device="cuda") @ torch.ones(32, 32, device="cuda")).sum().item()
        properties = torch.cuda.get_device_properties(0)
        hardware = str(getattr(properties, "uuid", properties.name))
    except Exception:
        pipe.send({"error": "gpu_unhealthy", "exit": 3})
        return
    pipe.send({"loading": True, "hardware_id": hardware})
    weights = Path(job["weights"])
    if not weights.is_dir() or any(weights.rglob("*.py")):
        pipe.send({"error": "invalid_model_directory", "exit": 2})
        return
    try:
        infer = LOADERS[job["arch"]](weights, job["max_new_tokens"])
    except ImportError:
        pipe.send({"error": "runtime_import_failed", "exit": 3})
        return
    except torch.cuda.OutOfMemoryError:
        pipe.send({"error": "model_load_oom", "exit": 2})
        return
    except Exception as error:
        pipe.send({"error": type(error).__name__, "exit": 3 if "CUDA" in str(error) else 2})
        return
    pipe.send({"ready": True, "hardware_id": hardware})
    while True:
        task = pipe.recv()
        if task is None:
            return
        torch.cuda.synchronize()
        start = time.monotonic()
        row = {"task_id": task["task_id"], "hardware_id": hardware}
        try:
            with torch.inference_mode():
                row.update(status="ok", raw=infer(task["path"], task["prompt"])[:65536])
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            row.update(status="resource_limit", raw="")
            torch.cuda.empty_cache()
        except Exception as error:
            if "CUDA" in str(error):
                pipe.send({"error": "gpu_unhealthy", "exit": 3})
                return
            row.update(status="invalid", raw="", error=type(error).__name__)
        row["elapsed_s"] = time.monotonic() - start
        pipe.send(row)


def _entry(worker, job, pipe):
    os.setsid()
    try:
        worker(job, pipe)
    finally:
        pipe.close()


def supervise(job, out, *, worker=_worker, context=None):
    """A killed clip cannot occupy the GPU beyond its deadline. Reload after timeout."""
    context = context or multiprocessing.get_context("spawn")
    process = pipe = None
    hardware = None
    deadline = time.monotonic() + job.get('job_timeout_s', 900.)
    cancel_path = Path(job['cancel_path']) if job.get('cancel_path') else None

    def stop():
        nonlocal process, pipe
        if process is not None:
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
            process.join(timeout=10)
            process.close()
            process = None
        if pipe is not None:
            pipe.close()
            pipe = None

    def receive(seconds):
        until = min(time.monotonic() + seconds, deadline)
        while True:
            if time.monotonic() >= deadline or (cancel_path and cancel_path.exists()):
                raise InterruptedError('job_cancelled_or_budget_expired')
            remaining = until - time.monotonic()
            if remaining <= 0:
                return None
            if pipe.poll(min(.1, remaining)):
                break
        try:
            return pipe.recv()
        except EOFError:
            return {"exit": 3, "error": "worker_died"}

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Jobs are immutable and retries use a new output path; never append stale attempts.
        with out.open("w") as stream:
            for index, task in enumerate(job["tasks"]):
                if index == job.get('pause_after_tasks'):
                    continuation = Path(job['continue_path'])
                    while not continuation.exists():
                        if time.monotonic() >= deadline or (cancel_path and cancel_path.exists()):
                            raise InterruptedError('evaluation_cancelled_or_budget_expired')
                        time.sleep(.05)
                    if json.loads(continuation.read_text()) != {'continue': True}:
                        break

                if process is None:
                    hardware = None
                    pipe, child = context.Pipe()
                    process = context.Process(target=_entry, args=(worker, job, child))
                    process.start()
                    child.close()
                    load_deadline = time.monotonic() + job.get("load_timeout_s", 120)
                    ready = receive(max(0., load_deadline - time.monotonic()))
                    if ready and ready.get("loading"):
                        hardware = ready["hardware_id"]
                        ready = receive(max(0., load_deadline - time.monotonic()))
                    if ready is None or ready.get("exit") == 2:
                        if hardware is None:
                            return 3  # GPU health was not established; do not score the model.
                        for pending in job["tasks"][index:]:
                            stream.write(json.dumps({"task_id": pending["task_id"], "status": "invalid",
                                         "raw": "", "elapsed_s": pending["deadline_s"],
                                         "hardware_id": hardware}) + "\n")
                        stream.flush()
                        return 2
                    if ready.get("exit"):
                        if ready.get("error") == "gpu_unhealthy":
                            print("gpu_unhealthy", file=sys.stderr)
                        return ready["exit"]
                    hardware = ready["hardware_id"]
                    # One bounded warmup per newly loaded model, excluded from timing.
                    pipe.send(task)
                    warm = receive(task["deadline_s"])
                    if warm is None:
                        stop()
                        row = {"task_id": task["task_id"], "status": "timeout", "raw": "",
                               "elapsed_s": task["deadline_s"], "hardware_id": hardware}
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        continue
                    if warm.get("exit"):
                        return warm["exit"]
                pipe.send(task)
                row = receive(task["deadline_s"])
                if row is None:
                    stop()
                    row = {"task_id": task["task_id"], "status": "timeout", "raw": "",
                           "elapsed_s": task["deadline_s"], "hardware_id": hardware}
                elif row.get("exit"):
                    return row["exit"]
                elif row["elapsed_s"] > task["deadline_s"]:
                    row.update(status="timeout", raw="")
                stream.write(json.dumps(row) + "\n")
                stream.flush()
        return 0
    except InterruptedError:
        return 75  # Infrastructure cancellation, not a miner score or a clip timeout.
    finally:
        stop()


def main():
    def terminate(*_):
        raise InterruptedError('supervisor_terminated')
    signal.signal(signal.SIGTERM, terminate)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    job = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(supervise(job, Path(sys.argv[2])))


if __name__ == "__main__":
    main()
