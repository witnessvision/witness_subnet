"""Public Witness synapse. Video bytes and private truth never cross this boundary."""

from __future__ import annotations

import math
import hashlib
import json
from typing import Annotated, Any, ClassVar, Literal
from urllib.parse import urlparse

import bittensor as bt
from pydantic import BaseModel, ConfigDict, Field, field_validator


_BUDGET_FIELDS = {"visual_tokens", "audio_seconds", "transcript_chars"}
_TASK_FIELDS = {"duration", "fps", "tier", "schema_version", "qa"}


class WitnessTask(bt.Synapse):
    """One independently metered reconstruction task and its miner response."""

    required_hash_fields: ClassVar[tuple[str, ...]] = (
        "task_id", "tool_base_url", "session_id", "scene_id", "seed_commitment",
        "budget", "task_spec", "deadline_s",
    )

    @property
    def body_hash(self) -> str:
        # Dict insertion order can differ between client/server validator processes.
        payload = {name: getattr(self, name) for name in self.required_hash_fields}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha3_256(encoded.encode()).hexdigest()

    @classmethod
    def from_headers(cls, headers: dict) -> "WitnessTask":
        # SDK 10.5 headers contain empty placeholders for required body fields.
        # Only transport metadata is validated here. Axon's verify_body_integrity
        # constructs the full class from JSON and runs every payload validator.
        metadata = bt.Synapse.from_headers(headers)
        values = metadata.model_dump()
        values.update(axon=metadata.axon, dendrite=metadata.dendrite)
        return cls.model_construct(
            **values, task_id="", tool_base_url="", session_id="", scene_id="",
            seed_commitment="", budget={}, task_spec={}, deadline_s=0.0,
        )

    task_id: str = Field(min_length=1)
    tool_base_url: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    seed_commitment: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget: dict[str, int | float]
    task_spec: dict[str, Any]
    deadline_s: float = Field(gt=0, allow_inf_nan=False)

    reconstruction: dict[str, Any] = Field(default_factory=dict)
    trace_summary: dict[str, Any] | None = None

    @field_validator("tool_base_url")
    @classmethod
    def validate_tool_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("tool_base_url must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, value: dict[str, int | float]) -> dict[str, int | float]:
        if set(value) != _BUDGET_FIELDS:
            raise ValueError(f"budget fields must be exactly {sorted(_BUDGET_FIELDS)}")
        normalized: dict[str, int | float] = {}
        for key, raw in value.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"budget.{key} must be numeric")
            number = float(raw)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"budget.{key} must be finite and non-negative")
            normalized[key] = raw
        return normalized

    @field_validator("task_spec")
    @classmethod
    def validate_public_task_spec(cls, value: dict[str, Any]) -> dict[str, Any]:
        if set(value) != _TASK_FIELDS:
            raise ValueError(f"task_spec fields must be exactly {sorted(_TASK_FIELDS)}")
        duration = value.get("duration")
        fps = value.get("fps")
        tier = value.get("tier")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError("task_spec.duration must be positive and finite")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("task_spec.fps must be a positive integer")
        if tier not in (1, 2, 3):
            raise ValueError("task_spec.tier must be 1, 2, or 3")
        schema_version = value.get("schema_version")
        if not isinstance(schema_version, str) or schema_version not in {"1.0", "1.1", "1.2", "1.3", "1.4", "1.5"}:
            raise ValueError("task_spec.schema_version must identify a supported scene schema")
        qa = value.get("qa")
        if not isinstance(qa, list):
            raise ValueError("task_spec.qa must be an array")
        public_qa: list[dict[str, str]] = []
        for item in qa:
            if not isinstance(item, dict) or set(item) != {"id", "q"}:
                raise ValueError("each task_spec.qa item must contain only id and q")
            if not all(isinstance(item[key], str) and item[key] for key in ("id", "q")):
                raise ValueError("task_spec.qa id and q must be non-empty strings")
            public_qa.append({"id": item["id"], "q": item["q"]})
        return {
            "duration": float(duration),
            "fps": fps,
            "tier": tier,
            "schema_version": schema_version,
            "qa": public_qa,
        }

    def deserialize(self) -> dict[str, Any]:
        """Dendrite callers consume only the scored response fields."""
        return {
            "reconstruction": self.reconstruction,
            "trace_summary": self.trace_summary,
        }


# Feedback has an explicit numerical schema: no arbitrary miner/validator
# dictionaries, solutions, seed reveals, session credentials or diagnostics.
UnitScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Family = Literal["events", "dialogue", "shots", "on_screen_text", "audio_events", "intentional_errors", "qa"]


class FeedbackModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeedbackCost(FeedbackModel):
    visual_tokens: int = Field(ge=0)
    audio_seconds: Nonnegative
    transcript_chars: int = Field(ge=0)


class FeedbackGate(FeedbackModel):
    threshold: UnitScore
    partial_floor: UnitScore
    reward_factor: UnitScore
    passed: bool


class FeedbackScene(FeedbackModel):
    scene_id: str = Field(min_length=1, max_length=128)
    tier: int = Field(ge=1, le=3)
    responded: bool
    status: Literal["ok", "missing", "busy", "deadline_exceeded", "error", "degraded"]
    quality: UnitScore
    efficiency_factor: UnitScore
    family_scores: dict[Family, UnitScore]
    cost: FeedbackCost
    gate: FeedbackGate
    score_before_duplicates: UnitScore
    duplicate_count: int = Field(ge=0)
    score: UnitScore


class FeedbackMiner(FeedbackModel):
    uid: int = Field(ge=0)
    hotkey: str = Field(min_length=1, max_length=128)
    round_score: UnitScore
    window_score: UnitScore
    window_rounds_observed: int = Field(ge=0)
    ema_score: UnitScore
    weight: UnitScore
    responded: bool
    scenes: list[FeedbackScene] = Field(max_length=64)


class FeedbackAggregation(FeedbackModel):
    version: str
    algorithm: Literal["rolling_mean_ema", "ema"]
    window_rounds: int | None = Field(ge=1)
    alpha: float = Field(gt=0, le=1, allow_inf_nan=False)
    bootstrap: Literal["first_positive_mean", "first_round"]


class RoundFeedback(FeedbackModel):
    schema_version: Literal["1.0"] = "1.0"
    round_id: str = Field(min_length=1, max_length=128)
    validator_hotkey: str = Field(min_length=1, max_length=128)
    completed_at: str = Field(min_length=1, max_length=64)
    scorer_version: str = Field(min_length=1, max_length=64)
    aggregation: FeedbackAggregation
    weight_policy: Literal["winner-takes-all", "proportional"]
    burn_uid: int | None = Field(ge=0)
    burn_rate: UnitScore
    winner_uid: int | None = Field(ge=0)
    submission_status: str = Field(min_length=1, max_length=32)
    weights_applied: None = None
    miners: list[FeedbackMiner] = Field(max_length=4096)


class WitnessFeedback(bt.Synapse):
    """Signed round feedback, delivered after scoring and weight submission."""

    report: RoundFeedback
    accepted: bool = False
    required_hash_fields: ClassVar[tuple[str, ...]] = ("report",)

    @property
    def body_hash(self) -> str:
        encoded = json.dumps(self.report.model_dump(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        return hashlib.sha3_256(encoded.encode()).hexdigest()

    @classmethod
    def from_headers(cls, headers: dict) -> "WitnessFeedback":
        metadata = bt.Synapse.from_headers(headers)
        values = metadata.model_dump()
        values.update(axon=metadata.axon, dendrite=metadata.dendrite)
        return cls.model_construct(**values, report=None, accepted=False)

    def deserialize(self) -> dict[str, Any]:
        return {"accepted": self.accepted}
