"""Audio-derived transcript artifacts, independent of benchmark answer labels."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_observed_transcript(directory: Path, duration: float) -> list[dict]:
    """Fail closed on missing/stale/invalid ASR; never fall back to scene truth."""
    artifact = json.loads((directory / "observations/transcript.json").read_text())
    if artifact.get("schema_version") != "1" or artifact.get("source") != "decoded_audio":
        raise ValueError("transcript must be a versioned decoded-audio observation")
    if artifact.get("video_sha256") != sha256(directory / "video.mp4"):
        raise ValueError("transcript media identity does not match")
    if not isinstance(artifact.get("model_sha256"), dict) or not artifact["model_sha256"]:
        raise ValueError("transcript requires model file identities")
    entries = artifact.get("entries")
    if not isinstance(entries, list):
        raise ValueError("transcript entries must be an array")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise ValueError("invalid transcript entry")
        start, end = entry.get("start"), entry.get("end")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (start, end)) or not 0 <= start < end <= duration:
            raise ValueError("invalid transcript interval")
        # Speakers are unknown without independent diarization. Never borrow labels.
        result.append({"start": start, "end": end, "text": entry["text"], "speaker": None})
    return sorted(result, key=lambda e: (e["start"], e["end"]))
