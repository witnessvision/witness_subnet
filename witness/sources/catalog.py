"""Validator-only random native clips; no miner implementation or index."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import random
import re
import secrets
import shutil
import time
import uuid

from witness.events import EventsTaskSpec, content_hash
from witness.mp4 import MAX_CLIP_BYTES
from witness.score_v5_0_0 import Reference
from witness.storage import write_private
from witness.tools.clip_media import crop_mp4
from .activitynet import SourceCache, native_row, sample_window, window_regions

POLICY = "uniform_eligible_original_without_replacement_v1"


def source_order(catalog: dict, seed: int) -> list[str]:
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{11}", key) or row.get("original") != key
           for key, row in catalog["originals"].items()):
        raise ValueError("invalid_catalog_original")
    order = sorted(key for key, row in catalog["originals"].items()
                   if row["eligible_annotation_geometry"])
    random.Random(seed).shuffle(order)
    return order


@contextmanager
def campaign_lock(root: Path):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        (root / ".lock").chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("catalog_campaign_already_running") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def failure_reason(error: Exception) -> str:
    # Never persist raw downloader errors: they can contain playback tokens.
    if isinstance(error, TimeoutError):
        return "preparation_deadline"
    known = {"source_not_public_or_identity_mismatch", "source_duration_changed",
             "source_media_too_large", "insufficient_disk_for_bounded_download",
             "unpinned_video_downloader", "no_unambiguous_native_window",
             "duplicate_source_media", "clip_too_large", "clip_duration_mismatch",
             "clip_changed_native_rate", "clip_outside_source"}
    known.update({'source_absent_from_public_mirror','mirror_source_size_changed'})
    if str(error) in known:
        return str(error)
    message = str(error).lower()
    for fragment, code in (("sign in to confirm", "download_authentication_required"),
                           ("private video", "download_private"),
                           ("video unavailable", "download_unavailable"),
                           ("not available", "download_unavailable")):
        if fragment in message:
            return code
    return type(error).__name__


def jobs_from(state: dict) -> list[dict]:
    return [attempt["job"] for attempt in state["attempts"] if attempt["status"] == "prepared"]


def save_state(root: Path, state: dict):
    state["counts"] = {name: sum(a["status"] == name for a in state["attempts"])
                       for name in ("prepared", "download_failed", "preparation_failed", "interrupted")}
    state["counts"]["drawn"] = len(state["attempts"])
    state["updated_unix"] = time.time()
    # The state is authoritative; jobs.json is a recoverable projection.
    write_private(root / "sampling.json", state)
    write_private(root / "jobs.json", jobs_from(state))


async def prepare_catalog(args, after_prepared=None) -> dict:
    if args.count <= 0 or args.max_attempts <= 0 or args.max_seconds <= 0:
        raise ValueError("positive_campaign_limits_required")
    with campaign_lock(args.output):
        catalog = json.loads(args.catalog.read_text())
        if getattr(args, 'require_available_catalog', False) and not catalog.get('availability'):
            raise ValueError('available_catalog_required')
        if catalog.get('availability') and not getattr(args, 'mirror_root', None):
            raise ValueError('available_catalog_requires_mirror_root')
        path = args.output / "sampling.json"
        state = json.loads(path.read_text()) if path.exists() else None
        seed = state["seed"] if state and args.seed is None else args.seed
        if seed is None:
            seed = secrets.randbits(128)
        order = source_order(catalog, seed)
        excluded={job['original'] for path in getattr(args,'exclude_jobs',[])
                  for job in json.loads(path.read_text())}
        known_originals = set(catalog['originals']) | set(catalog.get('availability', {}).get('excluded', {}))
        if excluded-known_originals:raise ValueError('excluded_source_not_in_catalog')
        order=[original for original in order if original not in excluded]
        identity = {"policy": POLICY, "catalog_hash": content_hash(catalog), "seed": seed,
                    "requested": args.count, "order_hash": content_hash(order)}
        mirror_root=getattr(args,'mirror_root',None)
        if mirror_root:
            from .activitynet_mirror import MirrorSourceCache
            cache=MirrorSourceCache(args.output/'source-cache',mirror_root,max_bytes=args.cache_bytes)
            if catalog.get('availability'):
                from .available_catalog import validate_available_source
                validate_available_source(catalog, cache)
            identity['acquisition_mirror']=cache.identity
        if excluded or (state and 'excluded_originals_hash' in state):
            identity.update(excluded_originals_hash=content_hash(sorted(excluded)),excluded_originals_count=len(excluded))
        if state:
            if any(state.get(key) != value for key, value in identity.items()):
                raise ValueError("catalog_campaign_identity_changed")
            if len(state["attempts"]) > len(order) or any(
                attempt["selection_rank"] != rank or attempt["original"] != order[rank]
                or not re.fullmatch(r"[0-9a-f]{32}\.mp4", attempt["clip_name"])
                for rank, attempt in enumerate(state["attempts"])
            ):
                raise ValueError("catalog_campaign_journal_changed")
        else:
            state = {**identity, "catalog_count": len(catalog["originals"]),
                     "eligible_count": len(order), "created_unix": time.time(),
                     "partition_policy": "all_public_annotation_partitions; diagnostic, not sealed holdout",
                     "availability_policy": "failed sources retained as exclusions; advance without retry",
                     "attempts": [], "episodes": []}
        if not mirror_root:
            cache = SourceCache(args.output / "source-cache", max_bytes=args.cache_bytes)
        for attempt in state["attempts"]:
            if attempt["status"] == "acquiring":
                attempt.update(status="interrupted", reason="previous_preparation_interrupted")
                (args.output / "clips" / attempt["clip_name"]).unlink(missing_ok=True)
                shutil.rmtree(cache.root / attempt["original"], ignore_errors=True)
        save_state(args.output, state)
        if after_prepared and jobs_from(state):
            await after_prepared(args.output / "jobs.json")
        episode = {"started_unix": time.time(), "max_attempts_total": args.max_attempts,
                   "max_seconds_preparation": args.max_seconds, "cache_bytes": args.cache_bytes}
        state["episodes"].append(episode)
        preparation_spent = 0.
        for rank in range(len(state["attempts"]), min(len(order), args.max_attempts)):
            if len(jobs_from(state)) >= args.count or preparation_spent >= args.max_seconds:
                break
            original = order[rank]
            row = catalog["originals"][original]
            attempt = {"original": original, "selection_rank": rank,
                       "status": "acquiring", "started_unix": time.time(),
                       "clip_name": uuid.uuid4().hex + ".mp4"}
            state["attempts"].append(attempt)
            save_state(args.output, state)
            destination = args.output / "clips" / attempt["clip_name"]
            destination.parent.mkdir(mode=0o700, exist_ok=True)
            began = time.monotonic()
            stage = "download"
            try:
                async with asyncio.timeout(args.max_seconds - preparation_spent):
                    source, receipt = await cache.get(row)
                    attempt["acquisition_s"] = time.monotonic() - began
                    attempt["source_receipt"] = receipt
                    stage = "preparation"
                    actual = native_row(row, receipt)
                    if not window_regions(actual):
                        raise ValueError("no_unambiguous_native_window")
                    if any(a.get("job", {}).get("source_sha256") == receipt["media_sha256"]
                           for a in state["attempts"][:-1]):
                        raise ValueError("duplicate_source_media")
                    rng = random.Random(content_hash({"seed": seed, "original": original}))
                    start, duration, reference = sample_window(actual, rng)
                    crop_started = time.monotonic()
                    measured = await crop_mp4(source, destination, start=start, duration=duration)
                    attempt["crop_s"] = time.monotonic() - crop_started
                    if destination.stat().st_size > MAX_CLIP_BYTES:
                        raise ValueError("clip_too_large")
                    reference["duration"] = measured["duration"]
                    Reference.model_validate(reference)
                    spec = EventsTaskSpec(duration=measured["duration"], fps=measured["fps"],
                                          has_audio=measured["has_audio"])
                    attempt["job"] = {"original": original,
                        "partition": receipt["partition"], "clip": str(destination.resolve()),
                        "spec": spec.model_dump(), "clip_sha256": measured["media_sha256"],
                        "source_sha256": receipt["media_sha256"], "source_start_s": start,
                        "reference": reference, "reference_hash": content_hash(reference)}
                    attempt["status"] = "prepared"
            except Exception as error:
                from .activitynet_mirror import AcquisitionPaused
                if isinstance(error,AcquisitionPaused):
                    attempt.update(status='interrupted',reason=str(error))
                    save_state(args.output,state)
                    raise
                attempt.update(status=stage + "_failed", reason=failure_reason(error))
                destination.unlink(missing_ok=True)
                if attempt["reason"] in {"insufficient_disk_for_bounded_download", "unpinned_video_downloader"}:
                    save_state(args.output, state)
                    raise
            finally:
                shutil.rmtree(cache.root / original, ignore_errors=True)
                attempt["preparation_elapsed_s"] = time.monotonic() - began
                preparation_spent += attempt["preparation_elapsed_s"]
            save_state(args.output, state)
            print(json.dumps({**state["counts"], "last_status": attempt["status"],
                              "last_reason": attempt.get("reason"),
                              "last_preparation_s": attempt["preparation_elapsed_s"]}), flush=True)
            if after_prepared and attempt["status"] == "prepared":
                await after_prepared(args.output / "jobs.json")
        state["stop_reason"] = ("complete" if len(jobs_from(state)) >= args.count else
                                "preparation_time_budget" if preparation_spent >= args.max_seconds else
                                "source_attempt_budget")
        episode.update(finished_unix=time.time(), preparation_elapsed_s=preparation_spent)
        save_state(args.output, state)
        return state
