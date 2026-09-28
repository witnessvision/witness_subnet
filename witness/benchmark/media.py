"""Deterministic clip rendering from committed originals."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from witness.events import content_hash
from witness.storage import sha256_file
from .contract import CLIP_MAX_S, CLIP_MIN_S, DURATION_SLACK_S


def _ffmpeg_version() -> str:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
    return result.stdout.splitlines()[0].decode()


PREPROCESSOR_ID = content_hash({"version": 5, "threads": 1, "bitexact": True, "video": "h264-yuv420p-640x360-8fps",
                                "audio": "aac-mono-16000", "ffmpeg": _ffmpeg_version()})


def _run(command: list[str], run=subprocess.run) -> None:
    result = run(command, capture_output=True, check=False, timeout=180)
    if result.returncode:
        raise ValueError("ffmpeg_media_processing_failed")


def probe(path: Path, *, run=subprocess.run) -> dict:
    result = run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                             "-of", "json", str(path)], capture_output=True, check=False, timeout=30)
    if result.returncode:
        raise ValueError("unreadable_media")
    body = json.loads(result.stdout)
    streams = body.get("streams", [])
    if not any(item.get("codec_type") == "video" for item in streams) or not any(
            item.get("codec_type") == "audio" for item in streams):
        raise ValueError("missing_audio_or_video")
    return body


def render_clip(source: Path, destination: Path, *, start: float, duration: float, run=subprocess.run) -> str:
    if start < 0 or not CLIP_MIN_S <= duration <= CLIP_MAX_S or not source.is_file():
        raise ValueError("invalid_clip_bounds")
    source_duration = float(probe(source, run=run)["format"]["duration"])
    if start + duration > source_duration + .05:
        raise ValueError("clip_outside_source")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Input seeking keeps long originals fast; re-encoding makes it frame-accurate.
    _run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{start:.6f}", "-i", str(source),
          "-t", f"{duration:.6f}", "-map", "0:v:0", "-map", "0:a:0",
          "-vf", "fps=8,scale=640:360:flags=lanczos,format=yuv420p",
          # One encoder thread and bitexact muxing: any validator reproduces the same bytes.
          "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-threads", "1",
          "-af", "aresample=16000", "-ac", "1", "-c:a", "aac", "-ar", "16000",
          "-fflags", "+bitexact", "-flags", "+bitexact",
          "-map_metadata", "-1", "-map_chapters", "-1", "-metadata", "creation_time=", "-movflags", "+faststart",
          str(destination)], run=run)
    # A damaged original can yield an undecodable clip; that is a preprocessing
    # failure (replaced from the reserve), never a clip shown to miners.
    check = run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(destination), "-f", "null", "-"],
                           capture_output=True, timeout=180, check=False)
    if check.returncode or check.stderr.strip():
        destination.unlink(missing_ok=True)
        raise ValueError("rendered_clip_not_decodable")
    media = probe(destination, run=run)
    if abs(float(media["format"]["duration"]) - duration) > DURATION_SLACK_S:
        raise ValueError("rendered_clip_duration_mismatch")
    destination.chmod(0o600)
    return sha256_file(destination)
