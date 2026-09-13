"""Quality-gated latency reward; annotation concordance remains scorer 5.0.0.

Only the validator's complete-response clock is an input. Miner claims about
time or resource use cannot affect this score. Evaluation time is excluded.
"""
from __future__ import annotations
import math
from witness.score_v5_0_0 import score_events as score_quality

SCORER_VERSION = "5.1.0"
DEADLINE_S = 180.0
TIME_WEIGHT = 0.30


def latency_reward(f1: float, elapsed_s: float, *, response_received: bool = True) -> dict:
    for value in (f1, elapsed_s):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("finite_numeric_reward_inputs_required")
    if not 0 <= f1 <= 1 or elapsed_s < 0 or not isinstance(response_received, bool):
        raise ValueError("invalid_reward_inputs")
    on_time = response_received and elapsed_s < DEADLINE_S
    quality = f1 if on_time else 0.
    speed = max(0., 1.-elapsed_s/DEADLINE_S) if on_time else 0.
    quality_component = (1.-TIME_WEIGHT)*quality
    time_component = TIME_WEIGHT*quality*speed
    return {"scorer_version": SCORER_VERSION, "reward": quality_component+time_component,
            "quality_component": quality_component, "time_component": time_component,
            "speed": speed, "time_weight": TIME_WEIGHT, "deadline_s": DEADLINE_S,
            "validator_elapsed_s": elapsed_s, "response_received": response_received,
            "on_time": on_time, "weights_enabled": False}


def score_events(reference: dict, response: dict, decisions: list[dict], *, elapsed_s: float,
                 evaluator_id: str, calibrated: bool = False, response_received: bool = True) -> dict:
    quality = score_quality(reference,response,decisions,evaluator_id=evaluator_id,calibrated=calibrated)
    return {**quality, "quality_scorer_version": quality["scorer_version"],
            **latency_reward(quality["f1"],elapsed_s,response_received=response_received)}
