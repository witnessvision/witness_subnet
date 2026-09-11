"""Witness Scorer v1.1.0: fixed inclusive quality gate, frozen v1.8 quality."""
from collections.abc import Mapping
import math
from typing import Any

from witness.score_v18 import LAMBDA, score_reconstruction as score_v18

SCORER_VERSION = "1.1.0"
QUALITY_THRESHOLD = 0.4


def reward_metrics(quality: float, efficiency: float, *, valid: bool = True) -> dict[str, Any]:
    for value in (quality, efficiency):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("reward inputs must be finite and in [0, 1]")
    passed = bool(valid and quality >= QUALITY_THRESHOLD)
    continuous = quality * efficiency if valid else 0.0
    return {
        "quality": quality if valid else 0.0,
        "continuous_score": round(continuous, 12),
        "threshold_margin": round(quality - QUALITY_THRESHOLD, 12) if valid else None,
        "full_reward_eligible": passed,
        "partial_floor": QUALITY_THRESHOLD,
        "reward_factor": float(passed),
        "score": round(continuous if passed else 0.0, 12),
    }


def score_reconstruction(scene: Mapping[str, Any], reconstruction: Mapping[str, Any],
                         cost: Mapping[str, Any] | None = None, *,
                         q_min: Mapping[int, float] | None = None,
                         cost_lambda: float = LAMBDA) -> dict[str, Any]:
    # q_min is accepted for the common scorer interface; this version's threshold
    # is fixed and cannot be changed by a caller or by another miner's quality.
    report = score_v18(scene, reconstruction, cost,
                      q_min={1: QUALITY_THRESHOLD, 2: QUALITY_THRESHOLD, 3: QUALITY_THRESHOLD},
                      cost_lambda=cost_lambda)
    metrics = reward_metrics(report["quality"], report["cost"]["efficiency_factor"])
    report.update(benchmark_version=SCORER_VERSION, score=metrics["score"], metrics=metrics)
    report["gate"].update(partial_floor=QUALITY_THRESHOLD, reward_factor=metrics["reward_factor"])
    return report
