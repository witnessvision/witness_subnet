"""Build the lossless miner reconstruction implied by a scene contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def perfect_reconstruction(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Return a reconstruction that receives quality 1.0 for ``scene``."""
    return {
        "schema_version": "1.0",
        "events": deepcopy(list(scene.get("events", []))),
        "dialogue": deepcopy(list(scene.get("dialogue", []))),
        "shots": deepcopy(list(scene.get("shots", []))),
        "on_screen_text": deepcopy(list(scene.get("on_screen_text", []))),
        "audio_events": deepcopy(list(scene.get("audio_events", []))),
        "intentional_errors": deepcopy(list(scene.get("intentional_errors", []))),
        "qa": {item["id"]: item["a"] for item in scene.get("qa", [])},
    }
