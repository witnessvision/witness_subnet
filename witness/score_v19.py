"""Candidate 1.9: unchanged v1.8 quality with a narrow partial-reward band."""
from collections.abc import Mapping
from typing import Any

from witness.reward import REWARD_VERSION, reward_metrics
from witness.score_v18 import DEFAULT_Q_MIN, LAMBDA, score_reconstruction as score_v18


def score_reconstruction(scene: Mapping[str, Any], reconstruction: Mapping[str, Any],
                         cost: Mapping[str, Any] | None = None, *,
                         q_min: Mapping[int, float] | None = None,
                         cost_lambda: float = LAMBDA) -> dict[str, Any]:
    report = score_v18(scene, reconstruction, cost, q_min=q_min, cost_lambda=cost_lambda)
    metrics = reward_metrics(report["quality"], report["cost"]["efficiency_factor"],
                             report["gate"]["q_min"])
    report.update(benchmark_version=REWARD_VERSION, score=metrics["score"], metrics=metrics)
    report["gate"].update(partial_floor=metrics["partial_floor"], reward_factor=metrics["reward_factor"])
    return report
