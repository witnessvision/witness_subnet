"""Versioned answer contract, evaluation policy and private evaluation records.

A model sees only the clip and ``prompt(duration)`` and must return a
``Response``. References are validator-owned labels (human, or machine) and
never leave the validator.
"""
from __future__ import annotations

import json
import math
import re
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from witness.events import StrictModel, content_hash

SCHEMA = "witness-native-2"
# Bump whenever a module constant that affects a decision changes (instruction,
# claim cap, clip grid, controls, validity rule, review time limit, bootstrap
# size). It is part of the policy identity, so an old commit cannot match.
RULES_VERSION = "witness-benchmark-rules-5"
MODALITIES = ("visual", "speech", "text", "sound")
Modality = Literal["visual", "speech", "text", "sound"]
CLIP_MIN_S = 10.
CLIP_MAX_S = 30.
CLIP_GRID_S = .125  # one frame at 8 fps
DURATION_SLACK_S = .15  # container rounding after re-encoding
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
ClipSeconds = Annotated[float, Field(ge=CLIP_MIN_S - DURATION_SLACK_S,
                                     le=CLIP_MAX_S + DURATION_SLACK_S, allow_inf_nan=False)]


class InfrastructureError(RuntimeError):
    """A validator-side failure: the evaluation is retried or abandoned, never scored."""


class Claim(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    start: Seconds
    end: Seconds
    subject: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=600)
    modality: Modality

    @field_validator("subject", "description")
    @classmethod
    def substantive(cls, value: str) -> str:
        if not value.strip() or re.fullmatch(r"[\s.…_-]+", value):
            raise ValueError("empty_claim_text")
        return value

    @model_validator(mode="after")
    def interval(self):
        if self.end <= self.start:
            raise ValueError("nonpositive_claim_interval")
        return self


class Response(StrictModel):
    schema_version: Literal["witness-native-2"] = SCHEMA
    claims: list[Claim] = Field(max_length=64)

    @model_validator(mode="after")
    def ordered_unique(self):
        if len({claim.id for claim in self.claims}) != len(self.claims):
            raise ValueError("duplicate_claim_id")
        if any(a.start > b.start for a, b in zip(self.claims, self.claims[1:])):
            raise ValueError("claims_not_ordered")
        return self


def strict_json(raw: str | bytes, *, max_bytes: int) -> object:
    """Parse JSON rejecting oversize input, duplicate keys and NaN/Infinity."""
    if len(raw.encode() if isinstance(raw, str) else raw) > max_bytes:
        raise ValueError("response_too_large")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite_json")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def parse_response(raw: str | bytes, duration: float) -> Response:
    if not math.isfinite(duration) or not CLIP_MIN_S - DURATION_SLACK_S <= duration <= CLIP_MAX_S + DURATION_SLACK_S:
        raise ValueError("invalid_clip_duration")
    payload = strict_json(raw, max_bytes=65536)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "claims"}:
        raise ValueError("invalid_response_envelope")
    response = Response.model_validate(payload)
    if any(claim.end > duration for claim in response.claims):
        raise ValueError("claim_outside_clip")
    return response


def instruction(duration: float) -> str:
    return (
        f"Describe what is seen and heard in this {duration:.3f}-second MP4. "
        "Return English JSON with schema_version='witness-native-2' and up to "
        "64 claims. Each claim has id, start, end, subject, description and "
        "modality (visual, speech, text or sound). State one observable fact "
        "per claim, at the interval where it occurs; include specific actions, "
        "spoken words, readable text and distinctive sounds. Black or frozen "
        "frames, silence, noise and glitches are observable facts too. Do not "
        "infer intent or identity. Order claims by start. No Markdown or "
        "source metadata."
    )


# The exact text a runner gives the model: the instruction plus a format example.
FORMAT_EXAMPLE = ('{"schema_version":"witness-native-2","claims":[{"id":"c1","start":<seconds>,"end":<seconds>,'
                  '"subject":"<who or what>","description":"<one observable fact>",'
                  '"modality":"<visual|speech|text|sound>"}, ...]}')


def prompt(duration: float) -> str:
    return (instruction(duration) + "\nTimes are seconds from the start of the clip. "
            "Output ONLY the JSON object, no other text. Example of the format:\n" + FORMAT_EXAMPLE)


class Task(StrictModel):
    """The only clip description a miner runner receives besides the MP4."""
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    clip_sha256: Digest
    duration: ClipSeconds


class Fact(StrictModel):
    """One atomic, true reference statement. Core facts weigh more than details."""
    claim: Claim
    salience: Literal["core", "detail"]
    origin: Literal["annotator", "review"] = "annotator"


class Reference(StrictModel):
    """Adjudicated ground truth for one rendered clip.

    ``kind`` says who labeled it; a policy accepts only its declared kind, so
    machine pilot labels can never be mistaken for human references.
    """
    schema_version: Literal["witness-reference-3"] = "witness-reference-3"
    kind: Literal["human", "machine"]
    clip_sha256: Digest
    duration: ClipSeconds
    facts: list[Fact] = Field(min_length=1, max_length=256)
    annotators: list[Digest] = Field(min_length=1)  # salted people or labeler-profile hashes

    @model_validator(mode="after")
    def binding(self):
        ids = [fact.claim.id for fact in self.facts]
        if len(ids) != len(set(ids)) or len(self.annotators) != len(set(self.annotators)):
            raise ValueError("duplicate_reference_fact_or_annotator")
        if any(fact.claim.end > self.duration for fact in self.facts):
            raise ValueError("fact_outside_clip")
        if not any(fact.origin == "annotator" for fact in self.facts):
            raise ValueError("reference_without_annotator_fact")
        return self


class Policy(StrictModel):
    """Every threshold that affects a decision; its hash is committed before a draw."""
    schema_version: Literal["witness-benchmark-policy-3"] = "witness-benchmark-policy-3"
    judge_id: str = Field(min_length=1)
    reviewer_id: str | None = None  # optional blind review of claims absent from the reference
    runtime_hash: Digest
    preprocessing_hash: Digest
    clip_min_s: float = Field(default=CLIP_MIN_S, ge=CLIP_MIN_S, le=CLIP_MAX_S)
    clip_max_s: float = Field(default=CLIP_MAX_S, ge=CLIP_MIN_S, le=CLIP_MAX_S)
    reference_kind: Literal["human", "machine"] = "human"
    min_annotators: int = Field(default=2, ge=1)
    deadline_per_clip_s: float = Field(default=10., gt=0, le=20)  # inference seconds per clip second
    clip_timeout_s: float = Field(default=60., gt=0, le=60)
    temporal_tolerance_s: float = Field(default=1., ge=0, le=2)
    min_claim_overlap: float = Field(default=.5, gt=0, le=1)  # share of a claim inside its cited fact
    detail_weight: float = Field(default=.5, gt=0, le=1)
    contradiction_penalty: float = Field(default=1., ge=0, le=2)
    max_facts_per_claim: int = Field(default=3, ge=1, le=8)
    novel_review_limit: int = Field(default=16, ge=1, le=64)
    quality_floor: float = Field(default=.6, ge=0, le=1)
    quality_margin: float = Field(default=.02, ge=0, le=.2)  # non-inferiority on quality
    superiority_margin: float = Field(default=.02, ge=0, le=.2)  # required reward gain
    alpha: float = Field(default=.05, gt=0, le=.1)  # per round, split across challengers
    max_challengers: int = Field(default=8, ge=1, le=32)
    screen_size: int = Field(default=16, ge=1)
    screen_max_invalid: int = Field(default=1, ge=0)
    screen_min_quality: float = Field(default=.3, ge=0, le=1)
    confirmation_size: int = Field(default=64, ge=2)
    control_max_quality: float = Field(default=.05, ge=0, le=1)

    @model_validator(mode="after")
    def consistent(self):
        if self.screen_size >= self.confirmation_size or self.clip_min_s > self.clip_max_s:
            raise ValueError("invalid_policy_sizes")
        if any(not (value / CLIP_GRID_S).is_integer() for value in (self.clip_min_s, self.clip_max_s)):
            raise ValueError("clip_bounds_off_frame_grid")
        if self.reviewer_id is not None and self.reviewer_id == self.judge_id:
            raise ValueError("reviewer_must_differ_from_judge")
        return self

    def deadline_s(self, duration: float) -> float:
        return min(duration * self.deadline_per_clip_s, self.clip_timeout_s)

    @property
    def identity(self) -> str:
        return content_hash({"rules": RULES_VERSION, "policy": self.model_dump()})


class Execution(StrictModel):
    task_id: str
    model_id: str
    checkpoint_hash: Digest
    clip_sha256: Digest
    runtime_hash: Digest
    preprocessing_hash: Digest
    status: Literal["ok", "invalid", "timeout", "resource_limit", "infra_error"]
    elapsed_s: float = Field(ge=0, allow_inf_nan=False)
    raw: str = Field(max_length=65536)
    audio_tokens: int = Field(ge=0)
    video_tokens: int = Field(ge=0)


class Clip(StrictModel):
    """A rendered excerpt bound to its committed source; labeled after preparation."""
    task: Task
    source_group: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_sha256: Digest
    source_path: str = Field(min_length=1)
    source_start: Seconds
    media_path: str = Field(min_length=1)


class Case(Clip):
    reference: Reference

    @model_validator(mode="after")
    def clip_matches(self):
        if (self.task.clip_sha256 != self.reference.clip_sha256
                or self.task.duration != self.reference.duration):
            raise ValueError("case_reference_mismatch")
        return self
