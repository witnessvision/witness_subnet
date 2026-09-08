"""Public Witness synapse. Video bytes and private truth never cross this boundary."""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlparse

import bittensor as bt
from pydantic import Field, field_validator


_BUDGET_FIELDS = {"visual_tokens", "audio_seconds", "transcript_chars"}
_TASK_FIELDS = {"duration", "fps", "tier", "schema_version", "qa"}


class WitnessTask(bt.Synapse):
    """One independently metered reconstruction task and its miner response."""

    task_id: str = Field(min_length=1)
    tool_base_url: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    seed_commitment: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget: dict[str, int | float]
    task_spec: dict[str, Any]
    deadline_s: float = Field(gt=0)

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

