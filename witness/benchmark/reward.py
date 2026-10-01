"""The single definition of how Witness evaluates a model.

This module defines how clips are sampled, how one answer is scored and how
clip scores become one number.

Sampling (``EVAL``)
    ``videos`` videos are drawn per evaluation from the pool's videos, and each
    pool video holds ``clips_per_video`` windows. A window has a uniform duration
    on the 1/8 s grid in [10, 30] s and a uniform start inside the video, both
    derived from a hash of (sampling rule, secret salt, video, index). The salt is
    private to each validator, so nobody can compute its windows in advance.

One clip
    events-v2 scores unique supported factual events, salience-weighted recall
    and temporal coverage. Bounded visual review can repair precision only.
    claims-v1 retains the historical claim-count and modality-mean scoring.
    ``clip_reward = quality * (clip_weight + time_weight * time_score)``;
    events-v2 uses 0.9/0.1, while legacy duel grading keeps 0.8/0.2.
    Invalid answers earn zero; time_score is clamped 1 - elapsed / deadline.

Aggregation
    ``video_score`` is the mean over that video's clips and ``eval_score`` the
    mean over videos, so every video weighs the same whatever its length or
    number of clips.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from pydantic import Field

from witness.events import StrictModel, content_hash
from .adjudication import JudgeProvider, assess
from .contract import CLIP_GRID_S, CLIP_MAX_S, CLIP_MIN_S, Policy, Reference, Response
from .scoring import Assessment, score

TAIL_S = .5  # never sampled at the very end of a video


class EvalSpec(StrictModel):
    sampling: str = "witness-window-v2"
    videos: int = Field(default=10, ge=1)
    clips_per_video: int = Field(default=2, ge=1)
    references: int = Field(default=2, ge=1)
    clip_weight: float = Field(default=.9, ge=0, le=1)
    time_weight: float = Field(default=.1, ge=0, le=1)

    @property
    def identity(self) -> str:
        return content_hash(self.model_dump())


EVAL = EvalSpec()


def eval_for_window(window: int) -> EvalSpec:
    from .protocol import TEN_VIDEO_WINDOW
    return EvalSpec(videos=5) if window < TEN_VIDEO_WINDOW else EVAL


def windows(identifier: str, length: float, salt: str, spec: EvalSpec = EVAL) -> list[tuple[int, float, float]]:
    """``(index, start, duration)`` of the video's evaluation windows for one validator's secret ``salt``."""
    if len(salt) < 16:
        raise ValueError("window_salt_too_short")
    last = math.floor((length - TAIL_S) / CLIP_GRID_S)  # grid steps that fit before the tail
    if last < CLIP_MIN_S / CLIP_GRID_S:
        return []
    result = []
    for index in range(spec.clips_per_video):
        digest = int(hashlib.sha256(f"{spec.sampling}:{salt}:{identifier}:{index}".encode()).hexdigest(), 16)
        longest = min(int(CLIP_MAX_S / CLIP_GRID_S), last)
        steps = int(CLIP_MIN_S / CLIP_GRID_S) + digest % (longest - int(CLIP_MIN_S / CLIP_GRID_S) + 1)
        start = (digest // 997) % (last - steps + 1)
        result.append((index, round(start * CLIP_GRID_S, 3), round(steps * CLIP_GRID_S, 3)))
    return result


def time_score(elapsed_s: float, duration: float, policy: Policy) -> float:
    return max(0., min(1., 1. - elapsed_s / policy.deadline_s(duration)))


def clip_reward(clip: float, speed: float, *, valid: bool, spec: EvalSpec = EVAL) -> float:
    return clip * (spec.clip_weight + spec.time_weight * speed) if valid else 0.


def reward(quality: float, elapsed_s: float, duration: float, policy: Policy, *, valid: bool,
           spec: EvalSpec = EVAL) -> dict:
    """``quality``, ``speed`` (time score) and ``reward`` of one answer."""
    if not valid:
        return {"quality": 0., "speed": 0., "reward": 0.}
    speed = time_score(elapsed_s, duration, policy)
    return {"quality": quality, "speed": speed, "reward": clip_reward(quality, speed, valid=True, spec=spec)}


def combine(references: list[Reference], response: Response, assessments: list[Assessment], policy: Policy, *, reviews=None) -> dict:
    """Clip score of one answer against several independent labelings of the same clip."""
    if policy.scoring_version == 'events-v2':
        from .event_scoring import measure
        return measure(references, response, assessments, policy, reviews)
    scores = [score(reference, response, assessment, policy) for reference, assessment in zip(references, assessments)]
    recall = sum(item["recall"] for item in scores) / len(scores)
    if not response.claims:
        return {"quality": 0., "precision": 0., "recall": recall, "per_reference": scores}
    maps = [{decision.prediction_id: decision for decision in assessment.decisions} for assessment in assessments]
    supported = sum(max(m[claim.id].share("supported") for m in maps) for claim in response.claims)
    contradicted = sum(min(m[claim.id].share("contradicted") for m in maps) for claim in response.claims)
    covered = max(len({fact_id for decision in assessment.decisions for part in decision.parts
                       if part.status == "supported" for fact_id in part.fact_ids}) for assessment in assessments)
    credit = min(supported, covered) - policy.contradiction_penalty * contradicted
    precision = max(0., credit) / len(response.claims)
    quality = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"quality": quality, "precision": precision, "recall": recall, "per_reference": scores}


def judge_clip(references: list[Reference], response: Response, *, judge: JudgeProvider, policy: Policy,
               root: Path) -> tuple[dict, list[Assessment]]:
    assessments = [assess(judge, reference, response, policy, root) for reference in references]
    return combine(references, response, assessments, policy), assessments


def video_scores(clips: list[dict]) -> dict[str, dict]:
    """``clips`` rows carry ``video``, ``quality`` and ``reward``; means per video."""
    grouped: dict[str, list[dict]] = {}
    for row in clips:
        grouped.setdefault(row["video"], []).append(row)
    return {video: {"clips": len(rows), "quality": sum(r["quality"] for r in rows) / len(rows),
                    "reward": sum(r["reward"] for r in rows) / len(rows)} for video, rows in sorted(grouped.items())}


def eval_score(videos: dict[str, dict]) -> dict:
    if not videos:
        raise ValueError("no_scored_videos")
    return {"videos": len(videos), "quality": sum(v["quality"] for v in videos.values()) / len(videos),
            "reward": sum(v["reward"] for v in videos.values()) / len(videos)}
