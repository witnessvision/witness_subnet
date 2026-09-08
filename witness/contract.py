"""Canonical vocabularies shared by generator, miners and scorer (derived from witness/scene.py)."""

from __future__ import annotations

import re
from typing import Any

ACTIONS_SYNTHETIC = ("enter", "move", "pick_up", "drop", "exit", "change_color", "vanish", "position_jump")
ACTIONS_EDIT = (
    "cut",
    "speed_change",
    "freeze",
    "repeat",
    "insert_foreign",
    "mirror",
    "tint",
    "dialogue",
    "text",
    "audio",
)
ACTIONS = ACTIONS_SYNTHETIC + ACTIONS_EDIT
ERROR_TYPES = (
    "continuity",
    "repeated_shot",
    "tint_change",
    "foreign_shot",
    "audio_visual_contradiction",
    "misleading_subtitle",
)
AUDIO_EVENT_KINDS = ("beep", "door_slam", "alarm")

ACTION_SYNONYMS = {
    "enters": "enter", "entered": "enter", "appear": "enter", "appears": "enter", "walks in": "enter",
    "moves": "move", "moved": "move", "walk": "move", "walks": "move", "walked": "move", "go": "move", "goes": "move",
    "picks up": "pick_up", "picked up": "pick_up", "pick up": "pick_up", "pickup": "pick_up", "grab": "pick_up",
    "grabs": "pick_up", "take": "pick_up", "takes": "pick_up", "lift": "pick_up", "lifts": "pick_up",
    "drops": "drop", "dropped": "drop", "put down": "drop", "puts down": "drop", "place": "drop", "places": "drop",
    "put": "drop", "puts": "drop", "release": "drop", "releases": "drop",
    "exits": "exit", "exited": "exit", "leave": "exit", "leaves": "exit", "left": "exit", "disappear": "vanish",
    "disappears": "vanish", "vanishes": "vanish", "changes color": "change_color", "color change": "change_color",
    "change colour": "change_color", "teleport": "position_jump", "jump": "position_jump", "jumps": "position_jump",
}

RECONSTRUCTION_SCHEMA: dict[str, Any] = {
    "events": [{"t": "seconds (float)", "actor": "the <color> <shape> (omit if none)", "action": "one of " + "|".join(ACTIONS), "object": "the <color> <kind> (omit if none)"}],
    "dialogue": [{"speaker": "speaker label from transcript", "start": "seconds", "end": "seconds", "text": "exact words"}],
    "shots": [{"start": "seconds", "end": "seconds"}],
    "on_screen_text": [{"start": "seconds", "end": "seconds", "text": "exact visible text"}],
    "audio_events": [{"t": "seconds", "kind": "one of " + "|".join(AUDIO_EVENT_KINDS)}],
    "intentional_errors": [{"t": "seconds", "type": "one of " + "|".join(ERROR_TYPES), "object": "the <color> <kind> (only if it applies)", "actor": "the <color> <shape> (only if it applies)"}],
    "qa": {"<question_id>": "shortest possible answer: a number, a time like 10:30, yes/no, a short noun phrase, or an entity description"},
}

_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*s?$")


def normalize_action(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = value.strip().casefold().replace("-", " ").replace("_", " ")
    if key.replace(" ", "_") in ACTIONS:
        return key.replace(" ", "_")
    return ACTION_SYNONYMS.get(key, ACTION_SYNONYMS.get(key.replace("_", " "), value))


def parse_seconds(value: Any) -> float | None:
    """Accept floats, numeric strings, '7.5s' and 'mm:ss'; return None when unparseable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _TIME_RE.match(value)
        if match:
            return int(match.group(1)) * 60 + float(match.group(2))
        try:
            return float(value.strip().rstrip("s"))
        except ValueError:
            return None
    return None
