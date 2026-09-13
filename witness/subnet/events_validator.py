"""Five diagnostic tasks per observed epoch, one registered target, no weights."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import time
import uuid

from witness.events import (DEADLINE_S, EVALUATOR_DEADLINE_S, EventsTaskSpec,
                            canonical_bytes, content_hash, observation_budget, response_hash)
from witness.score_v5_0_0 import Reference
from witness.tools.native_media import inspect_original
from witness.tools.server import Budget, DEFAULT_FFMPEG
from .events_state import EventsState
from .events_transport import ResponseRejected, validate_response
from .protocol import EventsFeedback, WitnessFeedback, WitnessTask
from .validator import InProcessToolServer


from witness.storage import write_private


def summarize(rows: list[dict]) -> dict:
    scores = [r["evaluation"]["score"] for r in rows
              if r.get("status") == "ok" and r.get("evaluation") is not None]
    return {"planned": 5,
            "sent": sum(r.get("send_attempted", r.get("send_confirmed")) is True for r in rows),
            "send_unknown": sum(r.get("send_confirmed") is None for r in rows),
            "completed": sum(r.get("status") == "ok" for r in rows),
            "expired": sum(r.get("status") == "deadline_exceeded" for r in rows),
            "rejected": sum(r.get("status") not in {"ok", "deadline_exceeded", "interrupted_unknown"} for r in rows),
            "interrupted": sum(r.get("status") == "interrupted_unknown" for r in rows),
            "scored": len(scores),
            **{key: sum(s[key] for s in scores)/len(scores) if scores else None
               for key in ("f1", "precision", "recall")},
            "provisional": len(scores) != 5 or any(s["provisional"] for s in scores)}


class EventsValidator:
    def __init__(self, chain, *, root: Path, target_hotkey: str, jobs_factory,
                 evaluate, host="127.0.0.1", port=0, public_url=None, start_after_epoch=-1):
        if not target_hotkey:
            raise ValueError("exact_target_hotkey_required")
        self.chain, self.root, self.target_hotkey = chain, root, target_hotkey
        self.jobs_factory, self.evaluate = jobs_factory, evaluate
        self.host, self.port, self.public_url = host, port, public_url
        self.state = EventsState(root, netuid=chain.netuid, target_hotkey=target_hotkey,
                                 start_after_epoch=start_after_epoch)

    def target(self):
        endpoints = [e for e in self.chain.miner_endpoints() if e.hotkey == self.target_hotkey]
        if len(endpoints) != 1:
            raise ValueError("target_not_uniquely_registered")
        return endpoints[0]

    async def task(self, task: dict, endpoint) -> dict:
        job = task["payload"]
        destination = self.root / task["round_id"] / "media" / task["id"]
        # Only native-media measurements enter the serving bundle. Source IDs,
        # licenses, hashes, annotations and paths stay in the private journal.
        metadata = inspect_original(Path(job["video"]), Path(DEFAULT_FFMPEG).with_name("ffprobe"))
        if metadata["media_sha256"] != job["media_sha256"]:
            raise ValueError("media_hash_mismatch")
        if "media_offset_s" in job:
            offset,duration=job["media_offset_s"],job["duration"]
            if (isinstance(offset,bool) or not isinstance(offset,(int,float)) or not math.isfinite(offset)
                or offset<0 or not isinstance(duration,(int,float)) or isinstance(duration,bool)
                or not math.isfinite(duration) or offset+duration>metadata["duration"]+1e-7):
                raise ValueError("invalid_private_media_window")
            metadata.update(source_offset_s=offset,duration=duration,
                            video_duration=min(duration,metadata["video_duration"]-offset))
        spec = EventsTaskSpec(duration=metadata["duration"], fps=metadata["fps"], has_audio=metadata["has_audio"])
        reference = Reference.model_validate(job["reference"]).model_dump()
        if abs(reference["duration"]-spec.duration) > .05:
            raise ValueError("reference_duration_mismatch")
        if content_hash(reference) != job["reference_hash"]:
            raise ValueError("reference_hash_mismatch")
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        (destination / "video.mp4").symlink_to(Path(job["video"]).resolve())
        private_media={k: metadata[k] for k in ("schema_version","duration","fps","video_duration","has_audio","audio")}
        if "source_offset_s" in metadata:private_media["source_offset_s"]=metadata["source_offset_s"]
        write_private(destination / "scene.json",private_media)
        response, status, elapsed, evaluation = None, "error", None, None
        with InProcessToolServer(destination, self.root / task["round_id"] / "observations",
                host=self.host, port=self.port, public_url=self.public_url, transcript_source="none") as server:
            budget = observation_budget(spec.duration)
            start = time.monotonic()
            session = server.store.create(task["id"], Budget(**budget), deadline_at=start+DEADLINE_S)
            request = WitnessTask(task_id=task["id"], tool_base_url=server.advertised_url,
                session_id=session.session_id, scene_id=task["id"], seed_commitment=secrets.token_hex(32),
                budget=budget, task_spec=spec.model_dump(), deadline_s=DEADLINE_S)
            try:
                async with asyncio.timeout(DEADLINE_S):
                    response = await self.chain.query(endpoint, request, timeout=DEADLINE_S)
                    if response is None:
                        raise ResponseRejected("missing")
                    validate_response(request, response)
                    status = "ok" if time.monotonic()-start <= DEADLINE_S else "deadline_exceeded"
            except TimeoutError:
                status = "deadline_exceeded"
            except ResponseRejected as exc:
                status = str(exc)
            except Exception:
                status = "transport_error"
            finally:
                elapsed = time.monotonic()-start
                server.store.close(session.session_id)
            observations = {"cost": session.cost.as_dict(), "failed": session.failed_observations,
                            "calls": sum(1 for line in session.log_path.read_text().splitlines()
                                         if json.loads(line)["endpoint"] != "POST /session")}
        # Invalid responses are not partially scored. Observation infrastructure
        # failures remain distinct from miner failure and never qualify quality.
        reconstruction = response.reconstruction if response is not None and status == "ok" else {}
        trace = response.trace_summary if response is not None and status == "ok" else None
        transport=request._transport_evidence or {"send_attempted": True,
                                                   "send_confirmed": response is not None}
        record = {"task_id": task["id"], "uid": endpoint.uid, "status": status,
                  **transport, "miner_elapsed_s": elapsed, "observations": observations,
                  "response": reconstruction, "trace_summary": trace,
                  "response_hash": response_hash(reconstruction, trace), "evaluation": None,
                  "evaluator_elapsed_s": None, "evaluator_status": "not_run",
                  "indexed": job.get("indexed"), "media_sha256": metadata["media_sha256"]}
        artifact = self.root / task["round_id"] / "responses" / (task["id"] + ".json")
        write_private(artifact, record)
        if status == "ok" and not observations["failed"]:
            started = time.monotonic()
            try:
                async with asyncio.timeout(EVALUATOR_DEADLINE_S):
                    evaluation = await self.evaluate(reference, reconstruction)
                record.update(evaluation=evaluation, evaluator_status="ok")
            except TimeoutError:
                record["evaluator_status"] = "deadline_exceeded"
            except Exception:
                record["evaluator_status"] = "error"
            record["evaluator_elapsed_s"] = time.monotonic()-started
        elif observations["failed"]:
            record["evaluator_status"] = "observation_infrastructure_error"
        write_private(artifact, record)
        return record

    async def step(self):
        snapshot = self.chain.epoch_state()
        active = self.state.active()
        if active is None:
            if not self.state.eligible(snapshot["epoch_index"]):
                return None
            self.target()  # Verify registration before sampling or claiming tasks.
            jobs = await self.jobs_factory()
            # Preparation can cross epochs; claim only the current observed epoch.
            snapshot = self.chain.epoch_state()
            round_id = self.state.begin(snapshot["epoch_index"], jobs)
            if round_id is None:
                return None
            active = self.state.active()
        round_id = active["id"]
        for task in self.state.tasks(round_id):
            if task["status"] != "planned":
                continue
            if not self.state.claim(task["id"]):
                continue
            try:
                result = await self.task(task, self.target())
            except Exception as exc:
                result = {"status": "preflight_rejected", "error_type": type(exc).__name__,
                          "send_confirmed": False, "evaluation": None, "miner_elapsed_s": None}
            result["task_id"] = task["id"]
            self.state.finish_task(task["id"], result)
        finished = self.chain.epoch_state()
        rows = [t["result"] for t in self.state.tasks(round_id)]
        metrics = summarize(rows)
        report = EventsFeedback(round_id=round_id, validator_hotkey=self.chain.validator_hotkey,
            completed_at=datetime.now(timezone.utc).isoformat(),
            **{k: metrics[k] for k in EventsFeedback.model_fields if k in metrics})
        artifact = {"round_id": round_id, "epoch": active["epoch"],
                    "finish_epoch": finished["epoch_index"],
                    "skipped_epochs": list(range(active["epoch"]+1, finished["epoch_index"]+1)),
                    "metrics": metrics, "feedback": report.model_dump(), "rows": rows,
                    "weights_enabled": False, "burn_rate": 1.0, "feedback_status": "not_sent"}
        write_private(self.root / round_id / "round.json", artifact)
        # Finalize BEFORE feedback, which is also attempted at most once.
        self.state.finish_round(round_id, finished["epoch_index"])
        try:
            async with asyncio.timeout(5):
                receipt = await self.chain.send_feedback(self.target(), WitnessFeedback(report=report), timeout=5)
            artifact["feedback_status"] = receipt.get("status", "unknown")
        except Exception:
            artifact["feedback_status"] = "failed"
        write_private(self.root / round_id / "round.json", artifact)
        write_private(self.root / "latest.json", artifact)
        return artifact

    async def run(self, *, poll_s=12.):
        if poll_s <= 0:
            raise ValueError("positive_poll_required")
        with (self.root / ".owner.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state.recover()
            try:
                while True:
                    try:
                        await self.step()
                        status = "waiting_for_epoch"
                    except Exception as exc:
                        status = "error_" + type(exc).__name__
                    write_private(self.root / "heartbeat.json", {"time": time.time(), "status": status})
                    await asyncio.sleep(poll_s)
            finally:
                self.state.close()
