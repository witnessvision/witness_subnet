"""All announced endpoints, five shared clips, no retries and no chain writes."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import ipaddress
from pathlib import Path
import time

from witness.events import EventsTaskSpec, content_hash
from witness.mp4_v5_2 import send_clip
from witness.score_v5_0_0 import score_events
from witness.score_v5_1_0 import latency_reward
from witness.storage import write_private
from witness.subnet.production_state import ProductionState


def announced_endpoints(endpoints):
    result = []
    for e in endpoints:
        axon = e.axon
        ip, port = getattr(axon, "ip", None), getattr(axon, "port", 0)
        if not ip or not port:
            continue
        address = ipaddress.ip_address(ip)
        if address.is_unspecified:
            continue
        host = f"[{address}]" if address.version == 6 else str(address)
        result.append({"uid": e.uid, "hotkey": e.hotkey, "url": f"http://{host}:{int(port)}"})
    return sorted(result, key=lambda e: e["uid"])


class ProductionValidator:
    def __init__(self, chain, *, root: Path, jobs_factory, evaluate, evaluator_identity,
                 calibration, start_after_epoch=-1):
        self.chain, self.root, self.jobs_factory, self.evaluate = chain, root, jobs_factory, evaluate
        # Encrypted wallet access may perform a synchronous password KDF. Resolve
        # it before the network loop so other miners' deadlines include only
        # transport/inference, never repeated validator key decryption.
        self.signing_key = chain.wallet.hotkey
        self.identity = {"transport": "5.2", "netuid": chain.netuid,
                         "validator_hotkey": chain.validator_hotkey, "ema_alpha": .2,
                         "tasks_per_miner": 5, "global_dispatches": 4,
                         "evaluator": evaluator_identity, "calibration_hash": content_hash(calibration)}
        self.calibration = calibration
        self.state = ProductionState(root/"scheduler.sqlite3", self.identity, start_after_epoch=start_after_epoch)
        self.dispatch_slots, self.judge_slots = asyncio.Semaphore(4), asyncio.Semaphore(4)

    def calibrated(self):
        c = self.calibration
        return bool(c and c.get("passed") is True
                    and c.get("prompt_hash") == self.identity["evaluator"]["prompt_hash"]
                    and c.get("evaluator_id") == self.identity["evaluator"]["evaluator_id"])

    async def dispatch(self, task, job, endpoint):
        result = {"status": "incomplete", "dispatch_attempted": False, "response_valid": False,
                  "evaluation_status": "not_started", "reward": None}
        try:
            clip = Path(job["clip"])
            if hashlib.sha256(clip.read_bytes()).hexdigest() != job["clip_sha256"]:
                raise ValueError("prepared_clip_hash_changed")
            async with self.dispatch_slots:
                if not self.state.claim(task["id"]):
                    return
                # Re-read the registry immediately before dispatch; UID reuse cannot inherit a request.
                current = await asyncio.to_thread(self.chain.registered_hotkey, endpoint["uid"])
                if current != endpoint["hotkey"]:
                    result.update(status="invalid_registration", reward=0., f1=0.)
                else:
                    def dispatched(request):
                        result.update(dispatch_attempted=True, task=request.model_dump())
                        self.state.record(task["id"], result, status="dispatching")
                    began = time.monotonic()
                    try:
                        received = await send_clip(endpoint["url"], clip, EventsTaskSpec.model_validate(job["spec"]),
                            self.signing_key, miner_hotkey=endpoint["hotkey"], netuid=self.chain.netuid,
                            task_id=task["id"], on_dispatch=dispatched)
                        result.update(status="valid", response_valid=True, received=received,
                                      miner_elapsed_s=received["elapsed_s"], evaluation_status="pending")
                        self.state.record(task["id"], result, status="evaluating")
                    except Exception as error:
                        result.update(status="invalid_or_absent", error_type=type(error).__name__,
                                      miner_elapsed_s=time.monotonic()-began, reward=0., f1=0.)
            if result["response_valid"]:
                async with self.judge_slots:
                    began = time.monotonic()
                    try:
                        async with asyncio.timeout(300):
                            evaluation = await self.evaluate(job, result["received"])
                        reproduced = score_events(job["reference"], result["received"]["response"], evaluation["decisions"],
                            evaluator_id=self.identity["evaluator"]["evaluator_id"], calibrated=self.calibrated())
                        if (evaluation["score"] != reproduced
                                or evaluation["prompt_hash"] != self.identity["evaluator"]["prompt_hash"]):
                            raise ValueError("stored_decisions_do_not_reproduce_score")
                        result.update(evaluation=evaluation, evaluation_status="complete", f1=reproduced["f1"],
                            reward=latency_reward(reproduced["f1"], result["miner_elapsed_s"])["reward"])
                    except Exception as error:
                        result.update(evaluation_status="pending", evaluator_error=type(error).__name__, reward=None)
                    result["evaluator_elapsed_s"] = time.monotonic()-began
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result.update(status="incomplete", infrastructure_error=type(error).__name__, reward=None)
            # Integrity failures before dispatch also consume their planned slot.
            self.state.claim(task["id"])
        self.state.record(task["id"], result)
        write_private(self.root/"requests"/(task["id"]+".json"), result)

    async def step(self):
        active = self.state.active()
        if active is None:
            epoch = int((await asyncio.to_thread(self.chain.epoch_state))["epoch_index"])
            if not self.state.eligible(epoch):
                return None
            try:
                jobs = await self.jobs_factory(epoch)
                endpoints = announced_endpoints(await asyncio.to_thread(self.chain.miner_endpoints))
                current_epoch = int((await asyncio.to_thread(self.chain.epoch_state))["epoch_index"])
                active = self.state.begin(current_epoch, jobs, endpoints)
            except Exception as error:
                now_epoch = int((await asyncio.to_thread(self.chain.epoch_state))["epoch_index"])
                self.state.skip(now_epoch)
                write_private(self.root/"preparation-errors"/(str(epoch)+".json"), {"error_type": type(error).__name__})
                raise
        if active is None:
            return None
        tasks = self.state.tasks(active["id"])
        async def one_miner(endpoint):
            for task in tasks:
                if task["hotkey"] == endpoint["hotkey"] and task["status"] == "planned":
                    await self.dispatch(task, active["jobs"][task["ordinal"]], endpoint)
        async with asyncio.TaskGroup() as group:
            for endpoint in active["endpoints"]:
                group.create_task(one_miner(endpoint))
        finish_epoch = int((await asyncio.to_thread(self.chain.epoch_state))["epoch_index"])
        report = self.state.finish(active["id"], finish_epoch)
        write_private(self.root/"rounds"/(active["id"]+".json"), report)
        write_private(self.root/"latest.json", report)
        return report

    async def run(self, *, interval=12.):
        with (self.root/".owner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            epoch = int((await asyncio.to_thread(self.chain.epoch_state))["epoch_index"])
            self.state.recover(epoch)
            while True:
                try:
                    await self.step()
                except Exception as error:
                    write_private(self.root/"last-error.json", {"type": type(error).__name__, "unix": time.time()})
                await asyncio.sleep(interval)

    def close(self):
        self.state.close()
