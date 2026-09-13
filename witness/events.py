"""Versioned point-event contract; no source identity or reference vocabulary."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from witness.tools.metering import visual_token_cost

SCHEMA_VERSION = "5.0"
SCORER_VERSION = "5.0.0"
DEADLINE_S = 180.0
MINER_DEADLINE_S = 170.0
EVALUATOR_DEADLINE_S = 300.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 2048
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Text = Annotated[str, Field(min_length=1, max_length=4096, pattern=r"\S")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class EventsTaskSpec(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    task_type: Literal["structured_events"] = "structured_events"
    duration: float = Field(ge=60, le=120, allow_inf_nan=False, strict=True)
    fps: float = Field(gt=0, allow_inf_nan=False, strict=True)
    has_audio: bool
    response_language: Literal["es"] = "es"
    max_response_bytes: Literal[2097152] = MAX_RESPONSE_BYTES
    max_events: Literal[2048] = MAX_EVENTS
    miner_deadline_s: Literal[170.0] = MINER_DEADLINE_S


class Event(StrictModel):
    timestamp: Seconds
    actor: Text
    action: Text
    objects: list[Text] = Field(max_length=64)
    details: list[Text] = Field(max_length=64)


class StructuredEvents(StrictModel):
    schema_version: Literal["5.0"]
    events: list[Event] = Field(max_length=MAX_EVENTS)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def response_hash(reconstruction: dict, trace_summary: dict | None = None) -> str:
    """Hash the complete miner-controlled response, including its trace."""
    return content_hash({"reconstruction": reconstruction, "trace_summary": trace_summary})


def validate_events(value: dict, duration: float) -> StructuredEvents:
    if len(canonical_bytes(value)) > MAX_RESPONSE_BYTES:
        raise ValueError("response_too_large")
    result = StructuredEvents.model_validate(value)
    if any(event.timestamp >= duration for event in result.events):
        raise ValueError("event_outside_clip")
    if any(a.timestamp > b.timestamp for a, b in zip(result.events, result.events[1:])):
        raise ValueError("events_not_ordered")
    return result


def observation_budget(duration: float) -> dict[str, int | float]:
    if isinstance(duration, bool) or not math.isfinite(duration) or not 60 <= duration <= 120:
        raise ValueError("clip duration must be 60 to 120 seconds")
    return {"visual_tokens": visual_token_cost(640, 360, math.ceil(2 * duration)),
            "audio_seconds": 2 * duration, "transcript_chars": 0}


def scored_text(event: Event) -> dict[str, str]:
    return {"actor": event.actor, "action": event.action,
            **{f"objects/{i}": s for i, s in enumerate(event.objects)},
            **{f"details/{i}": s for i, s in enumerate(event.details)}}
