"""One statistical futility look after three randomly ordered complete videos.

The chance of missing the largest k of N fixed rewards is C(N-k,3)/C(N,3).
Choose the smallest k with probability at most 0.1; except on that event,
the full mean is bounded by ((k-1)+(N-k+1)*max(observed))/N.
Deterministic best-possible completion bounds are always valid.
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


def futility(videos: list[dict], king_reward: float, *, planned_videos: int = VIDEOS) -> dict | None:
    """Return explicit score bounds only when they cannot clear the local margin."""
    total = planned_videos
    if type(total) is not int or total < 3:
        raise ValueError("invalid_planned_videos")
    n = len(videos)
    if not 0 < n < total:
        return None
    for row in videos:
        if not 0 <= row['reward'] <= row['quality'] <= 1:
            raise ValueError('invalid_video_score')
    remaining = total - n
    quality = (sum(r['quality'] for r in videos) + remaining) / total
    reward = (sum(r['reward'] for r in videos) + remaining) / total
    method = 'best_possible_completion'
    if n == 3:
        k = next(k for k in range(1, total + 1)
                 if (math.comb(total - k, n) if total - k >= n else 0) * 10 <= math.comb(total, n))
        statistical = ((k - 1) + (total - k + 1) * max(r['reward'] for r in videos)) / total
        if statistical < reward:
            reward, method = statistical, 'finite_batch_90'
    # Round upwards, against the exact king value transmitted on chain.
    if (upper_units(reward) - quantize(king_reward)) * 100 > 2 * SCALE:
        return None
    return {'method': method, 'confidence': CONFIDENCE if method == 'finite_batch_90' else 1.,
            'observed_videos': n, 'planned_videos': total,
            'quality_upper': quality, 'reward_upper': reward}
