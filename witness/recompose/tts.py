"""Bounded OpenAI TTS client for real-video recomposition."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import wave

import httpx
import numpy as np

from witness.tts import SpeechClip


OPENAI_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_CALL_LOG = Path("data/tts_calls.jsonl")
DEFAULT_CACHE_DIR = Path("data/tts_cache")
MAX_TTS_CALLS = int(os.environ.get("WITNESS_TTS_MAX_CALLS", "500"))
TARGET_SAMPLE_RATE = 24_000


def _read_wav(payload: bytes, *, voice: str) -> SpeechClip:
    with wave.open(io.BytesIO(payload), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if width != 2:
        raise RuntimeError(f"OpenAI TTS returned unsupported {width * 8}-bit WAV")
    samples = np.frombuffer(frames, dtype="<i2").copy()
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
    if sample_rate != TARGET_SAMPLE_RATE:
        target_length = round(len(samples) * TARGET_SAMPLE_RATE / sample_rate)
        positions = np.linspace(0, max(0, len(samples) - 1), target_length)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.int16)
        sample_rate = TARGET_SAMPLE_RATE
    return SpeechClip(samples=samples, sample_rate=sample_rate, engine=DEFAULT_MODEL, voice=voice)


def _cache_key(text: str, voice: str, speed: float, model: str, instructions: str) -> str:
    basis = json.dumps(
        {
            "text": text,
            "voice": voice,
            "speed": speed,
            "model": model,
            "instructions": instructions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(basis).hexdigest()


def _count_calls(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def synthesize_openai(
    text: str,
    *,
    voice: str = "marin",
    speed: float = 1.0,
    model: str = DEFAULT_MODEL,
    instructions: str = "Speak clearly and naturally at a moderate pace.",
    call_log: Path = DEFAULT_CALL_LOG,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> SpeechClip:
    """Synthesize one line, caching it and enforcing the configured persistent call cap."""

    if model not in ("gpt-4o-mini-tts", "tts-1"):
        raise ValueError("model must be gpt-4o-mini-tts or tts-1")
    if not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")
    key = _cache_key(text, voice, speed, model, instructions)
    cache_path = cache_dir / f"{key}.wav"
    if cache_path.is_file():
        return _read_wav(cache_path.read_bytes(), voice=voice)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI TTS")
    call_log.parent.mkdir(parents=True, exist_ok=True)
    count = _count_calls(call_log)
    if count >= MAX_TTS_CALLS:
        raise RuntimeError(f"OpenAI TTS call cap reached ({MAX_TTS_CALLS})")
    record = {
        "call": count + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "voice": voice,
        "speed": speed,
        "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    with call_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()

    payload: dict[str, str | float] = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav",
        "speed": speed,
    }
    if model == "gpt-4o-mini-tts":
        payload["instructions"] = instructions
    try:
        response = httpx.post(
            OPENAI_SPEECH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=120.0,
        )
    finally:
        del api_key
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI TTS failed with HTTP {response.status_code}")
    clip = _read_wav(response.content, voice=voice)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return clip


def logged_call_count(path: Path = DEFAULT_CALL_LOG) -> int:
    """Return how many paid requests have been logged."""

    return _count_calls(path)
