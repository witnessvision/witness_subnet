"""Versioned reward shaping and failure-inclusive iteration metrics."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

REWARD_VERSION = "1.9-candidate"
PARTIAL_BAND = 0.15
MIN_PARTIAL_QUALITY = 0.35


def reward_metrics(quality: float, efficiency: float, threshold: float,
                   *, valid: bool = True) -> dict[str, Any]:
    """Retain full-quality thresholds; taper credit only in a narrow band below."""
    for value in (quality, efficiency, threshold):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("reward inputs must be finite and in [0, 1]")
    floor = min(threshold, max(MIN_PARTIAL_QUALITY, threshold - PARTIAL_BAND))
    if not valid:
        factor = 0.0
    elif quality >= threshold:
        factor = 1.0
    elif quality <= floor:
        factor = 0.0
    else:
        factor = (quality - floor) / (threshold - floor)
    continuous = quality * efficiency if valid else 0.0
    return {
        "quality": quality if valid else 0.0,
        "continuous_score": round(continuous, 12),
        "threshold_margin": round(quality - threshold, 12) if valid else None,
        "full_reward_eligible": bool(valid and quality >= threshold),
        "partial_floor": round(floor, 12),
        "reward_factor": round(factor, 12),
        "score": round(continuous * factor, 12),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Scene means include every assigned scene; failed responses count as zero."""
    def summary(rows):
        n = len(rows)
        valid = [r for r in rows if r["responded"]]
        families = sorted({key for r in rows for key in r.get("family_scores", {})})
        mean = lambda values: round(sum(values) / n, 12) if n else 0.0
        return {
            "scenes": n,
            "valid_responses": len(valid),
            "valid_rate": len(valid) / n if n else 0.0,
            "quality": mean(float(r["quality"]) for r in valid),
            "continuous_score": mean(float(r["quality"]) * float(r["efficiency_factor"]) for r in valid),
            "full_reward_rate": mean(float(r["gate"]["passed"]) for r in valid),
            "positive_reward_rate": mean(float(float(r["score"]) > 0) for r in valid),
            "reward": mean(float(r["score"]) for r in valid),
            "family_scores": {key: mean(float(r.get("family_scores", {}).get(key, 0)) for r in valid)
                              for key in families},
        }
    return {**summary(records), "by_tier": {
        str(tier): summary([r for r in records if int(r["tier"]) == tier])
        for tier in sorted({int(r["tier"]) for r in records})}}
