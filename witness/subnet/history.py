"""Rolling round means followed by an EMA, with an explicit cold start."""
from __future__ import annotations

import math
from collections.abc import Mapping


def rolling_mean_ema(
    round_scores: Mapping[int, float],
    previous_ema: Mapping[str, float],
    previous_rounds: Mapping[str, list[float]],
    *,
    window: int,
    alpha: float,
) -> tuple[dict[int, float], dict[int, float], dict[str, list[float]]]:
    """Include zero rounds; use available samples while the window fills.

    Before the first positive window mean, EMA is zero. The first positive
    mean initializes EMA directly; subsequent means use the configured alpha.
    The window bounds the raw round history, not the EMA's longer memory.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("score window must be a positive integer")
    if not 0 < alpha <= 1:
        raise ValueError("EMA alpha must be in (0, 1]")
    means, smoothed, history = {}, {}, {}
    for uid, current in round_scores.items():
        key = str(uid)
        samples = [*previous_rounds.get(key, []), float(current)][-window:]
        prior = float(previous_ema.get(key, 0.0))
        if any(not math.isfinite(value) or value < 0 for value in [prior, *samples]):
            raise ValueError("reward history must be finite and nonnegative")
        mean = math.fsum(samples) / len(samples)
        means[uid] = mean
        smoothed[uid] = mean if prior == 0 else alpha * mean + (1 - alpha) * prior
        history[key] = samples
    return means, smoothed, history
