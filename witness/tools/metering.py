"""Cost accounting primitives shared by the tool server and clients."""

from __future__ import annotations

import math
from dataclasses import dataclass


PATCH_SIZE = 14


def visual_token_cost(width: int, height: int, frames: int = 1) -> int:
    """Return patch tokens for ``frames`` images using 14x14 pixel patches."""
    if width <= 0 or height <= 0 or frames < 0:
        raise ValueError("width and height must be positive and frames non-negative")
    return math.ceil(width / PATCH_SIZE) * math.ceil(height / PATCH_SIZE) * frames


@dataclass(slots=True)
class Cost:
    visual_tokens: int = 0
    audio_seconds: float = 0.0
    transcript_chars: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            visual_tokens=self.visual_tokens + other.visual_tokens,
            audio_seconds=self.audio_seconds + other.audio_seconds,
            transcript_chars=self.transcript_chars + other.transcript_chars,
        )

    def __sub__(self, other: "Cost") -> "Cost":
        return Cost(
            visual_tokens=self.visual_tokens - other.visual_tokens,
            audio_seconds=self.audio_seconds - other.audio_seconds,
            transcript_chars=self.transcript_chars - other.transcript_chars,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "visual_tokens": self.visual_tokens,
            "audio_seconds": round(self.audio_seconds, 6),
            "transcript_chars": self.transcript_chars,
        }
