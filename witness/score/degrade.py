"""Deterministic reconstruction perturbations used by scorer tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import random
from typing import Any, Mapping


def degrade_reconstruction(
    reconstruction: Mapping[str, Any],
    *,
    drop_events: int = 0,
    shift_frames: int = 0,
    wrong_answers: int = 0,
    fps: int = 24,
) -> dict[str, Any]:
    """Drop leading events, shift all temporal fields, and corrupt QA answers."""
    degraded = deepcopy(dict(reconstruction))
    degraded["events"] = list(degraded.get("events", []))[max(0, drop_events):]
    if shift_frames:
        for family in ("events", "dialogue", "shots", "on_screen_text", "audio_events", "intentional_errors"):
            for item in degraded.get(family, []):
                for key in ("frame", "start_frame", "end_frame"):
                    if key in item:
                        item[key] += shift_frames
                for key in ("t", "start", "end"):
                    if key in item:
                        item[key] += shift_frames / fps
    answers = degraded.get("qa", {})
    if isinstance(answers, dict):
        for qa_id in sorted(answers)[:max(0, wrong_answers)]:
            answers[qa_id] = "__wrong__"
    return degraded


def degrade_fraction(
    reconstruction: Mapping[str, Any], fraction: float, *, seed: int = 0, fps: int = 24
) -> dict[str, Any]:
    """Apply deterministic, distributed corruption at a requested severity."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    degraded = deepcopy(dict(reconstruction))
    families = ("events", "dialogue", "shots", "on_screen_text", "audio_events", "intentional_errors")
    for family in families:
        items = list(degraded.get(family, []))
        family_seed = int.from_bytes(hashlib.sha256(f"{seed}:{family}".encode()).digest()[:8], "big")
        rng = random.Random(family_seed)
        rng.shuffle(items)
        drop = round(len(items) * fraction)
        degraded[family] = items[drop:]
    answers = degraded.get("qa", {})
    if isinstance(answers, dict):
        keys = sorted(answers)
        rng = random.Random(seed ^ 0x51A7)
        rng.shuffle(keys)
        for key in keys[:round(len(keys) * fraction)]:
            answers[key] = "__wrong__"
    # Timing corruption grows from within the loosest tolerance to clearly outside it.
    shift_frames = round(fraction * fps)
    if shift_frames:
        for family in families:
            for item in degraded.get(family, []):
                for key in ("frame", "start_frame", "end_frame"):
                    if key in item:
                        item[key] += shift_frames
                for key in ("t", "start", "end"):
                    if key in item:
                        item[key] += shift_frames / fps
    return degraded
