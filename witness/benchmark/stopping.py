"""Predeclared futility checks on a random ordering of five complete videos.

For three draws without replacement, missing both largest of five fixed rewards
has probability C(3, 3) / C(5, 3) = 0.1. Otherwise the full mean is at most
(1 + 4 * max(observed)) / 5. This is a one-sided 90% finite-batch bound,
not a posterior probability of general model superiority. Only one statistical
look is allowed. Deterministic best-possible completion bounds are always valid.
"""
from __future__ import annotations

import math

from .protocol import SCALE, quantize

VIDEOS = 5
CONFIDENCE = .9


def upper_units(value: float) -> int:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid_score_bound')
    return min(SCALE, math.ceil(value * SCALE))


def futility(videos: list[dict], king_reward: float) -> dict | None:
    """Return explicit score bounds only when they cannot clear the local margin."""
    n = len(videos)
    if not 0 < n < VIDEOS:
        return None
    for row in videos:
        if not 0 <= row['reward'] <= row['quality'] <= 1:
            raise ValueError('invalid_video_score')
    remaining = VIDEOS - n
    quality = (sum(r['quality'] for r in videos) + remaining) / VIDEOS
    reward = (sum(r['reward'] for r in videos) + remaining) / VIDEOS
    method = 'best_possible_completion'
    if n == 3:
        statistical = (1 + 4 * max(r['reward'] for r in videos)) / VIDEOS
        if statistical < reward:
            reward, method = statistical, 'finite_batch_90'
    # Round upwards, against the exact king value transmitted on chain.
    if (upper_units(reward) - quantize(king_reward)) * 100 > 2 * SCALE:
        return None
    return {'method': method, 'confidence': CONFIDENCE if method == 'finite_batch_90' else 1.,
            'observed_videos': n, 'planned_videos': VIDEOS,
            'quality_upper': quality, 'reward_upper': reward}
