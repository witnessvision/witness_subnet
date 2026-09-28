"""Private audiovisual pools and fresh mainnet windows.

Mainnet window_batch samples five videos from catalogue-v2 and two random clips
per video, sharing that batch across the king and challengers. Retries retain
the private draw; a new window creates a fresh draw and two Luna references per
clip. build_pool processes only the explicitly supplied source selection.
Sources, GPU audio evidence and labeling resume from private hashed artifacts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import quote, urlparse, unquote

import httpx

from witness.budget import BudgetUnavailable
from witness.events import content_hash
from witness.storage import sha256_file, store_immutable, write_private
from .annotate import label_clip
from .archive import request
from .contract import Policy
from .media import PREPROCESSOR_ID, render_clip, probe
from .execution import run_process
from .reward import EVAL, windows
from .runner import audio_evidence

AUDIO_BATCH = 200  # clips per GPU job


def _salt(root: Path) -> str:
    """This evaluator's private window salt; created on the first build, never written to the manifest."""
    path = root / "salt"
    if not path.exists():
        write_private(path, {"salt": secrets.token_hex(32)})
    return json.loads(path.read_text())["salt"]


def source_url(video: dict, source_urls: dict | None = None) -> str:
    """Optional resolved Archive mirror, bound to the same catalogue item/file."""
    url = (source_urls or {}).get(video['identifier'])
    if url is None:
        return f"https://archive.org/download/{quote(video['identifier'], safe='')}/{quote(video['file'])}"
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.archive.org')
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.query or parsed.fragment
            or not unquote(parsed.path).endswith('/items/' + video['identifier'] + '/' + video['file'])):
        raise ValueError('invalid_archive_source_mirror')
    return url


def _cut(root: Path, video: dict, salt: str, *, cancelled=lambda: False, remaining_s=lambda: float('inf'),
         source_urls=None) -> list[dict]:
    """Download one original, render its evaluation windows and delete it."""
    source = root / "src" / (video["identifier"] + ".mp4")
    rows = []
    try:
        run = partial(run_process, cancelled=cancelled, remaining_s=remaining_s)
        with httpx.Client(timeout=max(.1, min(15., remaining_s())), headers={"User-Agent": "WitnessValidator/1.0"}) as client:
            request(client, source_url(video, source_urls),
                    output=source, expected_size=video["size"], cancelled=cancelled)
        length = float(probe(source, run=run)['format']['duration'])
        for index, start, duration in windows(video["identifier"], length, salt):
            name = f"{video['identifier']}__{index:04d}.mp4"
            digest = render_clip(source, root / "clips" / name, start=start, duration=duration, run=run)
            rows.append({"video": video["identifier"], "creator_group": video["creator_group"], "index": index,
                         "start": start, "file": name, "clip_sha256": digest, "duration": duration, "status": "ok"})
        if not rows:
            rows.append({"video": video["identifier"], "status": "too_short"})
    except (OSError, ValueError, httpx.HTTPError) as error:
        rows.append({"video": video["identifier"], "status": "failed", "error": f"{type(error).__name__}: {error}"[:200]})
    finally:
        source.unlink(missing_ok=True)
    return rows


def build_pool(root: Path, *, gpu, api, selected, log=print, salt=None, source_urls=None) -> dict:
    """Build or complete the pool; every stage skips what is already on disk."""
    for name in ("src", "clips", "references", "evidence", "labeling"):
        (root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    videos = selected
    cancelled = getattr(gpu, 'cancelled', lambda: False)
    remaining_s = getattr(gpu, 'remaining_s', lambda: float('inf'))
    run = partial(run_process, cancelled=cancelled, remaining_s=remaining_s)
    rows_path = root / "clips.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    lock = threading.Lock()
    complete = {video["identifier"] for video in videos
                if sum(r["video"] == video["identifier"] and r["status"] == "ok" for r in rows)
                == EVAL.clips_per_video}
    rows = [row for row in rows if row["video"] in complete]
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rows_path.chmod(0o600)
    todo = [video for video in videos if video["identifier"] not in complete]
    salt = salt or _salt(root)

    def cut(video):
        result = _cut(root, video, salt, cancelled=cancelled, remaining_s=remaining_s, source_urls=source_urls)
        with lock, rows_path.open("a") as stream:
            for row in result:
                stream.write(json.dumps(row) + "\n")
            rows.extend(result)
        log(json.dumps({"stage": "clips", "video": video["identifier"], "clips": sum(r["status"] == "ok" for r in result)}))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(cut, todo))
    clips = [row for row in rows if row["status"] == "ok"]
    missing = [row for row in clips if not (root / "evidence" / (row["file"] + ".json")).exists()]
    for first in range(0, len(missing), AUDIO_BATCH):
        batch = missing[first:first + AUDIO_BATCH]
        evidence = audio_evidence(gpu, [{"file": row["file"], "path": str(root / "clips" / row["file"]),
                                    "duration": row["duration"]} for row in batch], f"pool-audio-{first:04d}")
        for row in batch:
            if row["file"] in evidence:
                write_private(root / "evidence" / (row["file"] + ".json"), evidence[row["file"]])
        log(json.dumps({"stage": "audio", "clips": len(evidence)}))
    exhausted = threading.Event()  # today's labeling budget is spent: the rest waits for the next start

    def label(job):
        row, labeling = job
        target = root / "references" / f"{row['clip_sha256']}.{labeling}.json"
        evidence_path = root / "evidence" / (row["file"] + ".json")
        if cancelled() or exhausted.is_set() or target.exists() or not evidence_path.exists():
            return
        try:
            reference, receipt = label_clip(root / "clips" / row["file"], row["duration"],
                                            json.loads(evidence_path.read_text()), api,
                                            root / "labeling" / f"{row['file']}.{labeling}", labeling=labeling, run=run)
        except BudgetUnavailable:
            exhausted.set()
            return
        except (ValueError, KeyError, RuntimeError) as error:
            log(json.dumps({"stage": "label", "file": row["file"], "labeling": labeling,
                            "error": f"{type(error).__name__}: {error}"[:200]}))
            return
        store_immutable(target, reference.model_dump())
        write_private(root / "labeling" / f"{row['file']}.{labeling}.receipt.json", receipt)
        log(json.dumps({"stage": "label", "file": row["file"], "labeling": labeling, "facts": receipt["facts_kept"]}))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(label, [(row, k) for row in clips for k in range(EVAL.references)]))
    labeled = [{key: row[key] for key in ("video", "creator_group", "index", "start", "file", "clip_sha256", "duration")}
               for row in clips if all((root / "references" / f"{row['clip_sha256']}.{k}.json").exists()
                                       for k in range(EVAL.references))]
    manifest = {"schema_version": "witness-eval-pool-4", "slice_hash": content_hash(videos),
                "eval": EVAL.identity, "preprocessing_hash": PREPROCESSOR_ID,
                "videos": len({row["video"] for row in labeled}), "clips": labeled}
    write_private(root / "pool.json", manifest)
    return {"videos": manifest["videos"], "clips": len(labeled), "unlabeled": len(clips) - len(labeled),
            "failed_videos": sum(row["status"] != "ok" for row in rows)}


def load_pool(root: Path, policy: Policy) -> list[dict]:
    """Pool rows whose clip and all ``EVAL.references`` references exist, built with this ``EVAL``."""
    manifest = json.loads((root / "pool.json").read_text())
    if manifest.get("eval") != EVAL.identity:
        raise ValueError("evaluation_pool_built_with_another_eval_spec")
    rows = []
    for row in manifest["clips"]:
        references = [root / "references" / f"{row['clip_sha256']}.{k}.json" for k in range(EVAL.references)]
        media = root / "clips" / row["file"]
        if (media.is_file() and sha256_file(media) == row["clip_sha256"]
                and all(path.is_file() for path in references)):
            rows.append({**row, "media_path": str(media), "reference_paths": [str(path) for path in references]})
    if len({row["video"] for row in rows}) < EVAL.videos:
        raise ValueError("evaluation_pool_too_small")
    return rows


def window_batch(root: Path, window: int, validator: str, *, gpu, api, policy: Policy,
                 catalogue: Path | None = None, log=print, source_urls=None) -> list[dict]:
    """One fresh private draw per evaluator/window, stable across retries/restarts."""
    import random
    catalogue = catalogue or Path(__file__).parent / "data" / "catalogue-v2.json"
    videos = json.loads(catalogue.read_text())["videos"]
    if len(videos) != 1000 or len({v["identifier"] for v in videos}) != 1000:
        raise ValueError("mainnet_catalogue_requires_1000_distinct_videos")
    secret = _salt(root)
    salt = content_hash({"secret": secret, "validator": validator, "window": window,
                         "catalogue": content_hash(videos), "sampling": EVAL.sampling})
    selected = random.Random(salt).sample(videos, EVAL.videos)
    target = root / "windows" / str(window)
    store_immutable(target / "draw.json", {"window": window, "validator": validator,
                                           "selected": selected, "seed_hash": content_hash(salt)})
    build_pool(target, gpu=gpu, api=api, selected=selected, salt=salt, log=log, source_urls=source_urls)
    rows = load_pool(target, policy)
    if len(rows) != EVAL.videos * EVAL.clips_per_video:
        raise RuntimeError("window_batch_incomplete")
    order = {v['identifier']: i for i, v in enumerate(selected)}
    # Download completion order depends on content/length. The statistical look
    # must instead follow the original uniform random draw, grouped by video.
    return sorted(rows, key=lambda row: (order[row['video']], row['index']))
