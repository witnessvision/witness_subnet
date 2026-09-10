"""Witness Scorer v1.0.0: first production scoring contract.

Promoted from 1.9-candidate without changing quality, metering or reward rules.
Historical quality v1.8 and the shared partial-reward calculation remain frozen
dependencies of this contract; numerical changes require a new scorer version.
"""
from collections.abc import Mapping
from typing import Any

from witness.reward import reward_metrics
from witness.score_v18 import DEFAULT_Q_MIN, LAMBDA, score_reconstruction as score_v18

SCORER_VERSION = "1.0.0"


def score_reconstruction(scene: Mapping[str, Any], reconstruction: Mapping[str, Any],
                         cost: Mapping[str, Any] | None = None, *,
                         q_min: Mapping[int, float] | None = None,
                         cost_lambda: float = LAMBDA) -> dict[str, Any]:
    report = score_v18(scene, reconstruction, cost, q_min=q_min, cost_lambda=cost_lambda)
    metrics = reward_metrics(report["quality"], report["cost"]["efficiency_factor"],
                             report["gate"]["q_min"])
    report.update(benchmark_version=SCORER_VERSION, score=metrics["score"], metrics=metrics)
    report["gate"].update(partial_floor=metrics["partial_floor"], reward_factor=metrics["reward_factor"])
    return report
