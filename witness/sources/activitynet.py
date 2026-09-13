"""Validator annotation geometry and bounded native source acquisition."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile

from witness.events import content_hash
from witness.score_v5_0_0 import Reference
from witness.storage import write_private
from witness.subnet.processes import run_process
from witness.storage import sha256_file

CAPTIONS = {
    "url": "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip",
    "sha256": "7ed4f4d613d965683fbdbb9f01baad4fe2a439b99fc39cfd2f6fa6cf9201fd2b",
}
METADATA = {
    "url": "https://raw.githubusercontent.com/activitynet/ActivityNet/master/Evaluation/data/activity_net.v1-3.min.json",
    "sha256": "4c29d5b1561e1cbff9ac69816c159e60d417a18f16db8acc0fc254c377fa9ae3",
}
CATALOG_VERSION = "activitynet-captions-intervals-v1"
ELIGIBILITY_POLICY = "native_boundary_gaps_v2"
MAX_SOURCE_BYTES = 256 * 1024 * 1024
YTDLP_VERSION = "2026.08.19"


def pinned_download(path: Path, pin: dict):
    if path.exists():
        if sha256_file(path) != pin["sha256"]:
            raise ValueError("source_pin_mismatch")
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    try:
        with urllib.request.urlopen(pin["url"], timeout=60) as response, temporary.open("wb") as output:
            size = 0
            while data := response.read(1024*1024):
                size += len(data)
                if size > 32*1024*1024:
                    raise ValueError("source_archive_too_large")
                output.write(data)
        temporary.chmod(0o600)
        if sha256_file(temporary) != pin["sha256"]:
            raise ValueError("source_pin_mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def annotation_gaps(row: dict) -> list[tuple[float, float]]:
    """Closed gaps where clip boundaries do not cut an annotated description."""
    duration = row["duration"]
    cursor, gaps = 0., []
    for label in sorted(row["annotations"], key=lambda a: (a["start"], a["end"])):
        if cursor > duration:
            break
        if label["start"] >= cursor:
            gaps.append((cursor, min(label["start"], duration)))
        cursor = max(cursor, label["end"])
    if cursor <= duration:
        gaps.append((cursor, duration))
    return gaps


def native_row(row: dict, receipt: dict) -> dict:
    # Preserve every original human interval. A rounded human endpoint beyond
    # playable media cannot be silently shortened or supplied with fake padding.
    return {**row,"duration":min(row["duration"],receipt["native_duration"])}


def window_regions(row: dict) -> list[tuple[float, float, float, float]]:
    gaps = annotation_gaps(row)
    regions = []
    for i, (a, b) in enumerate(gaps):
        for c, d in gaps[i+1:]:
            if d-a >= 60 and c-b <= 120 and any(
                label["start"] >= b and label["end"] <= c for label in row["annotations"]
            ):
                regions.append((a,b,c,d))
    return regions


def sample_window(row: dict, rng: random.Random) -> tuple[float, float, dict]:
    """Choose a feasible gap pair, duration, then start; preserve human intervals.

    This is rejection-conditioned sampling, not uniform over all wall time.
    Zero-width gaps allow exact original/annotation boundaries without pretending
    they are random continuous offsets. Distribution is recorded in provenance.
    """
    regions = window_regions(row)
    if not regions:
        raise ValueError("no_unambiguous_60_120s_window")
    a,b,c,d = rng.choice(regions)
    duration = rng.uniform(max(60., c-b), min(120., d-a))
    start = rng.uniform(max(a,c-duration), min(b,d-duration))
    end = start+duration
    labels = [{"start": max(0.,label["start"]-start),
               "end": min(duration,label["end"]-start), "text":label["text"]}
              for label in row["annotations"]
              if label["start"] >= start-1e-7 and label["end"] <= end+1e-7]
    if any(label["start"] < end-1e-7 and label["end"] > start+1e-7
           and not (label["start"] >= start-1e-7 and label["end"] <= end+1e-7)
           for label in row["annotations"]):
        raise ValueError("annotation_cut_by_clip_boundary")
    reference = Reference(duration=duration, events=labels).model_dump()
    return start, duration, reference


def import_catalog(root: Path) -> dict:
    source = root / "source"
    pinned_download(source / "captions.zip", CAPTIONS)
    pinned_download(source / "activitynet-1.3.json", METADATA)
    metadata = json.loads((source / "activitynet-1.3.json").read_text())["database"]
    rows, excluded = {}, {}
    with zipfile.ZipFile(source / "captions.zip") as archive:
        # val_1 and val_2 describe overlapping videos. Do not double-count labels.
        # The alternate set remains byte-for-byte in the pinned archive.
        for split in ("train", "val_1", "val_2"):
            data = json.loads(archive.read(split+".json"))
            for key, value in data.items():
                video_id = key.removeprefix("v_")
                if video_id in rows:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                    raise ValueError("invalid_official_video_id")
                annotations = []
                reason = None
                duration = float(value["duration"])
                if not math.isfinite(duration) or duration <= 0:
                    reason = "invalid_duration"
                if len(value["timestamps"]) != len(value["sentences"]):
                    reason = "label_count_mismatch"
                for (start,end), sentence in zip(value["timestamps"],value["sentences"]):
                    start,end = float(start),float(end)
                    if not (0 <= start < end <= duration) or not sentence.strip():
                        reason = "invalid_human_annotation"
                    annotations.append({"start":start,"end":end,"text":sentence})
                row = {"original":video_id,"duration":duration,"annotation_set":split,
                       "source_url":"https://www.youtube.com/watch?v="+video_id,
                       "annotations":annotations,
                       "activities":sorted({a["label"] for a in metadata.get(video_id,{}).get("annotations",[])})}
                if reason is None and not window_regions(row):
                    reason = "no_unambiguous_60_120s_window"
                row["eligible_annotation_geometry"] = reason is None
                row["exclusion"] = reason
                rows[video_id] = row
                if reason: excluded[reason] = excluded.get(reason,0)+1
    result = {"version":CATALOG_VERSION,"captions":CAPTIONS,"metadata":METADATA,
              "annotation_policy":"train_then_val_1_then_val_2_for_missing_originals_only",
              "sampling":"uniform_original_then_uniform_feasible_gap_pair_duration_start",
              "rights":"Original video rights remain with owners; no commercial or redistribution grant inferred. Public research access only.",
              "participant_ids_available":False,"originals":rows,
              "counts":{"total":len(rows),"annotation_eligible":sum(r["eligible_annotation_geometry"] for r in rows.values()),
                        "exclusions":excluded}}
    write_private(root / "catalog.json", result)
    return result


def partition_for(channel_id: str) -> str:
    if not channel_id or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}",channel_id):
        raise ValueError("verified_channel_required_for_partition")
    bucket = int(hashlib.sha256((CATALOG_VERSION+":"+channel_id).encode()).hexdigest()[:16],16) / 2**64
    return "development" if bucket < .6 else "calibration" if bucket < .8 else "evaluation"


class SourceCache:
    def __init__(self, root: Path, *, max_bytes=2*1024**3, proxy: str | None = None):
        if max_bytes < 2*MAX_SOURCE_BYTES:
            raise ValueError("cache_requires_space_for_one_source_and_remux")
        self.root, self.max_bytes = root, max_bytes
        if proxy is not None:
            from urllib.parse import urlsplit
            parsed=urlsplit(proxy)
            if parsed.scheme!='socks5' or parsed.hostname!='127.0.0.1' or not parsed.port or parsed.username or parsed.password:
                raise ValueError('source_proxy_must_be_local_ssh_tunnel')
        self.proxy=proxy
        root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def prune(self, *, reserve=2*MAX_SOURCE_BYTES):
        entries = sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p:p.stat().st_mtime)
        sizes = {p:sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in entries}
        total = sum(sizes.values())
        for entry in entries:
            if total+reserve <= self.max_bytes:
                break
            shutil.rmtree(entry)
            total -= sizes[entry]
        if shutil.disk_usage(self.root).free < reserve+512*1024**2:
            raise ValueError("insufficient_disk_for_bounded_download")

    async def get(self, row: dict) -> tuple[Path, dict]:
        import yt_dlp.version
        if yt_dlp.version.__version__ != YTDLP_VERSION:
            raise ValueError("unpinned_video_downloader")
        video_id = row["original"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}",video_id):
            raise ValueError("invalid_original")
        destination = self.root / video_id
        receipt = destination / "receipt.json"
        if receipt.exists():
            record = json.loads(receipt.read_text())
            media = destination / "video.mp4"
            if sha256_file(media) != record["media_sha256"]:
                raise ValueError("cached_media_hash_mismatch")
            destination.touch()
            return media,record
        self.prune()
        if destination.exists(): shutil.rmtree(destination)
        destination.mkdir(mode=0o700)
        try:
            await run_process([sys.executable,"-m","yt_dlp","--no-playlist","--no-progress","--no-warnings",
                *(["--proxy",self.proxy] if self.proxy else []),
                "--retries","0","--fragment-retries","0","--extractor-retries","0","--socket-timeout","20",
                "--max-filesize",str(MAX_SOURCE_BYTES//2),"-f","bv*[height<=360]+ba/b[height<=360]",
                "--merge-output-format","mp4","--write-info-json","-o",str(destination/"video.%(ext)s"),
                row["source_url"]], timeout=120, max_output=1024*1024)
            media = destination / "video.mp4"
            info = json.loads((destination/"video.info.json").read_text())
            if info.get("id") != video_id or info.get("availability") != "public":
                raise ValueError("source_not_public_or_identity_mismatch")
            partition = partition_for(info.get("channel_id"))
            probe = json.loads(await run_process(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(media)],timeout=30))
            duration = float(probe["format"]["duration"])
            # A changed upload/duration invalidates historical label alignment.
            if abs(duration-row["duration"]) > .5:
                raise ValueError("source_duration_changed")
            if media.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError("source_media_too_large")
            await run_process(["ffmpeg","-v","error","-xerror","-i",str(media),"-f","null","-"],timeout=120)
            record = {"original":video_id,"media_sha256":sha256_file(media),"bytes":media.stat().st_size,
                      "channel_id":info["channel_id"],"partition":partition,"native_duration":duration,
                      "source_url":row["source_url"],"source_license":info.get("license"),
                      "format_id":info.get("format_id"),"downloader_version":YTDLP_VERSION,
                      "streams":[{k:s[k] for k in ("codec_type","codec_name","sample_rate","channels","r_frame_rate") if k in s}
                                 for s in probe["streams"]]}
            media.chmod(0o600)
            # Source responses can contain expiring playback URLs. Retain only
            # the fixed provenance allowlist, never raw token-bearing metadata.
            (destination/"video.info.json").unlink()
            write_private(receipt,record)
            return media,record
        except BaseException:
            shutil.rmtree(destination,ignore_errors=True)
            raise
