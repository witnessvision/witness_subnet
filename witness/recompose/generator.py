"""Deterministic real-video recomposition plans and ffmpeg rendering."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping
import wave

import numpy as np

from witness.render import FFMPEG, _event_sound
from witness.scene import seconds
from witness.tts import SpeechClip

from .tts import DEFAULT_MODEL, synthesize_openai


FPS = 24
WIDTH = 640
HEIGHT = 360
AUDIO_RATE = 24_000
FONT_FILES = {
    "DejaVu Sans Bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    "DejaVu Serif Bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    "DejaVu Sans Mono Bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    "Liberation Sans Bold": Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    "Liberation Serif Bold": Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
    "FreeSans Bold": Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
    "Lato Heavy": Path("/usr/share/fonts/truetype/lato/Lato-Heavy.ttf"),
}
TTS_VOICES = (
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx",
    "nova", "sage", "shimmer", "verse", "marin", "cedar",
)
TTS_SPEEDS = (0.92, 0.96, 1.0, 1.04, 1.08)
SOURCE_GAIN_DB = -14.0
SOURCE_GAIN_CHOICES_DB = (-18.0, -16.0, -14.0, -12.0)
DIALOGUE_DUCK_GAIN_DB = -30.0
DIALOGUE_DUCK_PADDING_SECONDS = 0.3
MIN_DIALOGUE_SOURCE_SNR_DB = 12.0
TTS_MIX_GAIN = 0.90
OBSERVATION_FPS = 4
OBSERVATION_RESOLUTION = [320, 180]
MIN_PLANNED_CUT_DIFFERENCE = 8.0
MIN_TEXT_BACKGROUND_CONTRAST = 16.0
TIER_PLAN = {
    1: {
        "segments": (2, 5),
        "duration": (5.25, 8.75),
        "dialogue": (1, 2),
        "edit_probability": 0.25,
        "semantic_error_probability": 0.15,
        "audio_probability": 0.40,
    },
    2: {
        "segments": (4, 9),
        "duration": (4.75, 8.25),
        "dialogue": (1, 4),
        "edit_probability": 0.55,
        "semantic_error_probability": 0.40,
        "audio_probability": 0.60,
    },
    3: {
        "segments": (6, 12),
        "duration": (4.25, 7.75),
        "dialogue": (2, 5),
        "edit_probability": 0.75,
        "semantic_error_probability": 0.65,
        "audio_probability": 0.78,
    },
}
EVENT_DURATIONS = {"beep": 0.35, "door_slam": 0.55, "alarm": 1.2}


def _at(frame: int, **fields: Any) -> dict[str, Any]:
    return {"frame": frame, "t": seconds(frame), **fields}


def _interval(start_frame: int, end_frame: int, **fields: Any) -> dict[str, Any]:
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start": seconds(start_frame),
        "end": seconds(end_frame),
        **fields,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    videos = manifest.get("videos") if isinstance(manifest, dict) else None
    if not isinstance(videos, list) or not videos:
        raise ValueError("pool manifest must contain a non-empty videos array")
    required = {"id", "url", "title", "uploader", "license", "duration", "path"}
    for index, item in enumerate(videos):
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError(f"pool manifest video {index} is missing required fields")
        if float(item["duration"]) <= 0:
            raise ValueError(f"pool manifest video {index} has invalid duration")
    return manifest


def _source_intervals(
    rng: random.Random,
    source_duration: float,
    output_frames: list[int],
    rates: list[float],
) -> list[tuple[float, float]]:
    margin = min(5.0, source_duration * 0.03)
    usable_start, usable_end = margin, source_duration - margin
    bin_width = (usable_end - usable_start) / len(output_frames)
    intervals: list[tuple[float, float]] = []
    for index, (frames, rate) in enumerate(zip(output_frames, rates)):
        length = frames / FPS * rate
        bin_start = usable_start + index * bin_width
        bin_end = usable_start + (index + 1) * bin_width
        if length > bin_width - 0.05:
            raise ValueError(
                f"source is too short for tier plan: segment needs {length:.2f}s in a {bin_width:.2f}s bin"
            )
        start = rng.uniform(bin_start, bin_end - length)
        # Millisecond quantization is stable and more than sufficient for 24 fps extraction.
        start = round(start, 3)
        intervals.append((start, round(start + length, 3)))
    return intervals


def _assign_output_frames(shots: list[dict[str, Any]]) -> None:
    cursor = 0
    for index, shot in enumerate(shots, start=1):
        shot["id"] = f"shot_{index}"
        shot["start_frame"] = cursor
        cursor += int(shot["output_frames"])
        shot["end_frame"] = cursor
        shot["start"] = seconds(shot["start_frame"])
        shot["end"] = seconds(shot["end_frame"])


def _copy_source(item: Mapping[str, Any]) -> dict[str, Any]:
    required = ("id", "url", "title", "uploader", "license", "duration", "path")
    optional = ("query_tag", "query", "format_tag", "format", "tag", "genre", "category")
    return {
        key: deepcopy(item[key])
        for key in (*required, *optional)
        if key in item
    }


def _pool_query_tag(item: Mapping[str, Any]) -> str | None:
    """Return the first explicit per-source pool grouping tag, if present."""

    for key in ("query_tag", "query", "format_tag", "format", "tag", "genre", "category"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return None


def _random_output_frames(rng: random.Random, bounds: tuple[float, float]) -> int:
    """Sample a frame duration without exposing whole-second boundaries."""
    frames = round(rng.uniform(*bounds) * FPS)
    if frames % FPS == 0:
        frames += rng.choice((-1, 1))
    return max(FPS, frames)


def _insert_randomly(rng: random.Random, shots: list[dict[str, Any]], shot: dict[str, Any]) -> None:
    shots.insert(rng.randrange(len(shots) + 1), shot)


def build_edit_plan(
    seed: int,
    tier: int,
    manifest: Mapping[str, Any],
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Return the pure deterministic plan for a seed, tier, and manifest."""

    if tier not in TIER_PLAN:
        raise ValueError("tier must be 1, 2, or 3")
    videos = sorted((dict(item) for item in manifest["videos"]), key=lambda item: str(item["id"]))
    rng = random.Random(seed)
    randomly_selected = videos[rng.randrange(len(videos))]
    source = (
        next(item for item in videos if str(item["id"]) == source_id)
        if source_id is not None
        else randomly_selected
    )
    config = TIER_PLAN[tier]
    segment_bounds = config["segments"]
    count = rng.randint(int(segment_bounds[0]), int(segment_bounds[1]))
    output_frames = [
        _random_output_frames(rng, config["duration"])
        for _ in range(count)
    ]
    edit_probability = float(config["edit_probability"])
    rates = [
        rng.choice((0.82, 0.88, 1.12, 1.18, 1.25))
        if rng.random() < edit_probability / max(2.0, count / 2)
        else 1.0
        for _ in range(count)
    ]
    intervals = _source_intervals(rng, float(source["duration"]), output_frames, rates)

    base: list[dict[str, Any]] = []
    for index, ((start, end), frames, rate) in enumerate(zip(intervals, output_frames, rates), start=1):
        base.append(
            {
                "kind": "source",
                "segment_id": f"segment_{index}",
                "source_id": source["id"],
                "source_start": start,
                "source_end": end,
                "output_frames": frames,
                "playback_rate": rate,
                "mirror": rng.random() < edit_probability / max(2.0, count / 2),
                "tint": "warm" if rng.random() < edit_probability / max(2.0, count / 2) else None,
            }
        )
    rng.shuffle(base)
    shots = [deepcopy(item) for item in base]

    has_repeat = rng.random() < edit_probability
    has_freeze = rng.random() < edit_probability
    foreign_options = [item for item in videos if item["id"] != source["id"]]
    source_query_tag = _pool_query_tag(source)
    same_genre = [
        item for item in foreign_options
        if source_query_tag is not None and _pool_query_tag(item) == source_query_tag
    ]
    if same_genre:
        foreign_options = same_genre
    has_foreign = bool(foreign_options) and rng.random() < edit_probability
    foreign = foreign_options[rng.randrange(len(foreign_options))] if has_foreign else None

    if has_repeat:
        repeated = deepcopy(base[rng.randrange(len(base))])
        repeated["kind"] = "repeat"
        repeated["derived_from"] = repeated["segment_id"]
        original_index = next(
            index
            for index, shot in enumerate(shots)
            if shot["segment_id"] == repeated["segment_id"]
        )
        shots.insert(rng.randint(original_index + 1, len(shots)), repeated)

    if has_freeze:
        frozen_from = base[rng.randrange(len(base))]
        freeze_frames = _random_output_frames(rng, (0.75, 2.35))
        frozen = {
            "kind": "freeze",
            "segment_id": f"freeze_{frozen_from['segment_id']}",
            "source_id": source["id"],
            "source_start": round(
                rng.uniform(float(frozen_from["source_start"]), float(frozen_from["source_end"]) - 1 / FPS),
                6,
            ),
            "output_frames": freeze_frames,
            "playback_rate": 1.0,
            "mirror": False,
            "tint": None,
            "derived_from": frozen_from["segment_id"],
        }
        frozen["source_end"] = round(float(frozen["source_start"]) + 1 / FPS, 6)
        _insert_randomly(rng, shots, frozen)

    if foreign is not None:
        foreign_frames = _random_output_frames(rng, (2.25, 5.75))
        foreign_length = foreign_frames / FPS
        max_start = max(0.0, float(foreign["duration"]) - foreign_length - 1)
        foreign_start = round(rng.uniform(min(5.0, max_start), max_start), 3) if max_start else 0.0
        foreign_shot = {
            "kind": "foreign",
            "segment_id": "foreign_segment",
            "source_id": foreign["id"],
            "source_start": foreign_start,
            "source_end": round(foreign_start + foreign_length, 6),
            "output_frames": foreign_frames,
            "playback_rate": 1.0,
            "mirror": False,
            "tint": None,
        }
        _insert_randomly(rng, shots, foreign_shot)

    semantic_probability = float(config["semantic_error_probability"])
    misleading_subtitle = rng.random() < semantic_probability
    audio_visual_contradiction = rng.random() < semantic_probability
    contradiction_frame = None
    if audio_visual_contradiction:
        estimated_duration = sum(int(shot["output_frames"]) for shot in shots)
        contradiction_frame = rng.randint(FPS, max(FPS, estimated_duration - 2 * FPS))

    _assign_output_frames(shots)
    return {
        "seed": seed,
        "tier": tier,
        "source": _copy_source(source),
        "foreign_source": _copy_source(foreign) if foreign else None,
        "shots": shots,
        "duration_frames": shots[-1]["end_frame"],
        "dialogue_count": rng.randint(*config["dialogue"]),
        "audio_probability": float(config["audio_probability"]),
        "misleading_subtitle": misleading_subtitle,
        "audio_visual_contradiction": audio_visual_contradiction,
        "contradiction_frame": contradiction_frame,
        "source_gain_db": rng.choice(SOURCE_GAIN_CHOICES_DB),
        "foreign_selection": {
            "policy": "same_query_tag_if_available_else_any_source",
            "source_query_tag": source_query_tag,
            "same_genre_match": bool(foreign and source_query_tag and _pool_query_tag(foreign) == source_query_tag),
        },
    }


OVERLAY_STYLES = (
    "lower_third_bar",
    "corner_caption",
    "centered_title",
    "subtitle_strip",
    "watermark",
    "price_tag_ui_chip",
)
NAMES = ("Maya", "Jonah", "Priya", "Owen", "Lena", "Ravi", "Nora", "Felix", "Iris", "Theo", "Zara", "Emil")
OBJECTS = ("mug", "notebook", "parcel", "camera", "key", "basket", "lamp", "ticket", "scarf", "bottle", "map", "plate")
COLORS = ("amber", "blue", "coral", "green", "ivory", "navy", "orange", "purple", "red", "silver", "teal", "yellow")
PLACES = ("kitchen", "station", "studio", "garden", "market", "workshop", "hallway", "terrace", "library", "cafe", "harbor", "office")
WORDS = ("EMBER", "ORBIT", "CEDAR", "LUMEN", "NOVA", "PIXEL", "RIVER", "SUMMIT", "VELVET", "WILLOW", "MINT", "ATLAS")


def _text_style(
    rng: random.Random,
    text: str,
    style: str,
    anchor: tuple[int, int] | None = None,
    *,
    flash: bool = False,
) -> dict[str, Any]:
    if style not in OVERLAY_STYLES:
        raise ValueError(f"unsupported overlay style: {style}")
    font_family = rng.choice(tuple(FONT_FILES))
    if style == "lower_third_bar":
        font_size = rng.randint(23, 30)
        padding = rng.randint(10, 14)
        overlay_color = rng.choice(("#101820", "#152a45", "#401820", "#0b332f"))
        font_color = rng.choice(("#ffffff", "#fff1a8", "#d9f7ff"))
        opacity = rng.choice((0.78, 0.84, 0.90, 0.94))
        background_box = True
    elif style == "centered_title":
        font_size = rng.randint(30, 40)
        padding = rng.randint(8, 12)
        overlay_color = "#101010"
        font_color = rng.choice(("#ffffff", "#fff2a8", "#bcecff"))
        opacity = rng.choice((0.88, 0.94, 1.0))
        background_box = rng.random() < 0.35
    elif style == "subtitle_strip":
        font_size = rng.randint(21, 26)
        padding = rng.randint(8, 11)
        overlay_color = "#080808"
        font_color = rng.choice(("#ffffff", "#fff7d1"))
        opacity = rng.choice((0.76, 0.84, 0.90))
        background_box = True
    elif style == "watermark":
        font_size = rng.randint(18, 22)
        padding = rng.randint(5, 8)
        overlay_color = "#101010"
        font_color = rng.choice(("#ffffff", "#e7f1ff", "#fff2c9"))
        opacity = rng.choice((0.58, 0.66, 0.74))
        background_box = False
    elif style == "price_tag_ui_chip":
        font_size = rng.randint(22, 29)
        padding = rng.randint(8, 11)
        overlay_color = rng.choice(("#ffdd45", "#e8ff66", "#ff7061", "#ffffff", "#2aebbc"))
        font_color = rng.choice(("#111111", "#172033", "#3a1010"))
        opacity = rng.choice((0.88, 0.94, 1.0))
        background_box = True
    else:
        font_size = rng.randint(18, 25)
        padding = rng.randint(7, 10)
        overlay_color = rng.choice(("#101820", "#f3f4f6", "#173f5f", "#fff0b8"))
        font_color = rng.choice(("#ffffff", "#111111", "#fff0a6"))
        opacity = rng.choice((0.72, 0.82, 0.90, 0.96))
        background_box = rng.random() < 0.75
    if flash:
        font_size = max(18, min(font_size, 22))

    width_factor = 0.58 if "Sans" in font_family else 0.62
    width = min(WIDTH - 16, max(64, round(len(text) * font_size * width_factor) + padding * 2))
    height = font_size + padding * 2
    if anchor is None:
        if style == "lower_third_bar":
            width = min(WIDTH - 48, max(width, rng.randint(300, 500)))
            x = rng.randint(24, WIDTH - width - 24)
            y = rng.randint(HEIGHT - height - 58, HEIGHT - height - 20)
        elif style == "centered_title":
            x = (WIDTH - width) // 2
            y = rng.randint(92, max(92, 204 - height))
        elif style == "subtitle_strip":
            width = min(WIDTH - 32, max(width, round(WIDTH * rng.uniform(0.66, 0.92))))
            x = (WIDTH - width) // 2
            y = rng.randint(HEIGHT - height - 34, HEIGHT - height - 14)
        else:
            left = rng.random() < 0.5
            top = rng.random() < 0.5
            x = rng.randint(12, 36) if left else rng.randint(max(12, WIDTH - width - 36), WIDTH - width - 12)
            y = rng.randint(12, 42) if top else rng.randint(max(12, HEIGHT - height - 48), HEIGHT - height - 12)
    else:
        x = min(anchor[0] + rng.randint(-10, 10), WIDTH - width - 8)
        y = min(anchor[1] + rng.randint(-8, 8), HEIGHT - height - 8)
    x, y = max(8, x), max(8, y)
    outline_width = rng.choice((0, 1, 1, 2)) if background_box else rng.choice((1, 1, 2, 2, 3))
    shadow_offset = rng.choice((0, 1, 2, 3))
    return {
        "bbox": [x, y, x + width, y + height],
        "style": style,
        "font_family": font_family,
        "font_file": str(FONT_FILES[font_family]),
        "overlay_color": overlay_color,
        "font_color": font_color,
        "font_size": font_size,
        "opacity": opacity,
        "background_box": background_box,
        "padding": padding,
        "outline_width": outline_width,
        "outline_color": "#000000" if font_color != "#111111" else "#ffffff",
        "shadow_x": shadow_offset,
        "shadow_y": shadow_offset,
        "shadow_color": "#000000",
    }


def _text_observability(start_frame: int, end_frame: int, tier: int, *, flash: bool) -> dict[str, Any]:
    duration_ms = (end_frame - start_frame) / FPS * 1000
    if flash:
        return {
            "guaranteed": False,
            "exception": "tier_3_flash",
            "minimum_duration_ms": 125,
            "duration_ms": round(duration_ms, 3),
        }
    stride = FPS // OBSERVATION_FPS
    sample_frame = ((start_frame + stride - 1) // stride) * stride
    if sample_frame >= end_frame:
        raise RuntimeError("text interval has no 4 fps observation sample")
    return {
        "guaranteed": True,
        "sampling_fps": OBSERVATION_FPS,
        "resolution": OBSERVATION_RESOLUTION.copy(),
        "sample_frame": sample_frame,
        "minimum_text_height_px_at_640x360": 18,
        "duration_ms": round(duration_ms, 3),
    }


def _text_item(
    rng: random.Random,
    start_frame: int,
    end_frame: int,
    *,
    tier: int,
    style: str,
    flash: bool = False,
    anchor: tuple[int, int] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    text = str(fields["text"])
    return _interval(
        start_frame,
        end_frame,
        **fields,
        **_text_style(rng, text, style, anchor, flash=flash),
        observability=_text_observability(start_frame, end_frame, tier, flash=flash),
    )


def _text_items_collide(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if max(int(left["start_frame"]), int(right["start_frame"])) >= min(
        int(left["end_frame"]), int(right["end_frame"])
    ):
        return False
    left_x0, left_y0, left_x1, left_y1 = (int(value) for value in left["bbox"])
    right_x0, right_y0, right_x1, right_y1 = (int(value) for value in right["bbox"])
    return (
        max(left_x0, right_x0) < min(left_x1, right_x1)
        and max(left_y0, right_y0) < min(left_y1, right_y1)
    )


def _resolve_text_collisions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move only colliding overlays, preserving already-valid seeded output."""
    resolved: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: (value["start_frame"], value["id"])):
        if not any(_text_items_collide(item, previous) for previous in resolved):
            resolved.append(item)
            continue

        x0, y0, x1, y1 = (int(value) for value in item["bbox"])
        width, height = x1 - x0, y1 - y0
        maximum_x, maximum_y = WIDTH - width - 8, HEIGHT - height - 8
        preferred = (
            (8, 8),
            ((WIDTH - width) // 2, 8),
            (maximum_x, 8),
            (8, (HEIGHT - height) // 2),
            ((WIDTH - width) // 2, (HEIGHT - height) // 2),
            (maximum_x, (HEIGHT - height) // 2),
            (8, maximum_y),
            ((WIDTH - width) // 2, maximum_y),
            (maximum_x, maximum_y),
        )
        candidates = list(preferred)
        candidates.extend(
            (candidate_x, candidate_y)
            for candidate_y in range(8, maximum_y + 1, 8)
            for candidate_x in range(8, maximum_x + 1, 8)
        )
        seen: set[tuple[int, int]] = set()
        for candidate_x, candidate_y in candidates:
            position = (max(8, candidate_x), max(8, candidate_y))
            if position in seen:
                continue
            seen.add(position)
            candidate = {
                **item,
                "bbox": [
                    position[0],
                    position[1],
                    position[0] + width,
                    position[1] + height,
                ],
            }
            if not any(_text_items_collide(candidate, previous) for previous in resolved):
                item = candidate
                break
        else:
            raise RuntimeError(f"cannot place text overlay without collision: {item['id']}")
        resolved.append(item)
    return resolved


def _random_code(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 6)))


def _random_interval_in_shot(
    rng: random.Random,
    shot: Mapping[str, Any],
    length_seconds: float,
) -> tuple[int, int]:
    length_frames = min(
        max(1, round(length_seconds * FPS)),
        int(shot["end_frame"]) - int(shot["start_frame"]),
    )
    latest = max(int(shot["start_frame"]), int(shot["end_frame"]) - length_frames)
    start = rng.randint(int(shot["start_frame"]), latest)
    return start, start + length_frames


def _build_text(
    plan: Mapping[str, Any],
    rng: random.Random,
    dialogue_plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shots = plan["shots"]
    tier = int(plan["tier"])
    first_content = rng.choice((rng.choice(WORDS), rng.choice(NAMES), f"{rng.choice(WORDS)} TV", f"${rng.randint(3, 89)}.{rng.randint(0, 99):02d}"))
    second_content = rng.choice((rng.choice(WORDS), rng.choice(NAMES), f"{rng.randint(1, 12)}:{rng.randrange(0, 60):02d}", f"{rng.choice(PLACES).title()} Live"))
    if second_content == first_content:
        second_content = f"{second_content} NOW"
    alpha_shot = shots[rng.randrange(len(shots))]
    beta_shot = shots[rng.randrange(len(shots))]
    first_start, first_end = _random_interval_in_shot(rng, alpha_shot, rng.uniform(1.25, 2.85))
    last_start, last_end = _random_interval_in_shot(rng, beta_shot, rng.uniform(1.25, 2.85))
    items = [
        _text_item(rng, first_start, first_end, tier=tier, style="lower_third_bar", id="marker_alpha", role="caption", text=first_content),
        _text_item(rng, last_start, last_end, tier=tier, style="corner_caption", id="marker_beta", role="caption", text=second_content),
    ]
    middle = shots[rng.randrange(len(shots))]
    code_text = _random_code(rng)
    code_start, code_end = _random_interval_in_shot(rng, middle, rng.uniform(1.15, 2.25))
    items.append(
        _text_item(rng, code_start, code_end, tier=tier, style="price_tag_ui_chip", id="code", role="code", text=code_text)
    )
    for index in range(rng.randint(0, int(plan["tier"]) + 1)):
        shot = shots[rng.randrange(len(shots))]
        value = rng.choice(
            (
                rng.choice(WORDS),
                rng.choice(NAMES),
                f"{rng.choice(PLACES).title()} {rng.randint(2, 98)}",
                f"${rng.randint(2, 99)}.{rng.randint(0, 99):02d}",
            )
        )
        start, end = _random_interval_in_shot(rng, shot, rng.uniform(0.9, 2.7))
        style = rng.choice(OVERLAY_STYLES)
        items.append(
            _text_item(
                rng,
                start,
                end,
                tier=tier,
                style=style,
                id=f"marker_extra_{index + 1}",
                role=rng.choice(("caption", "sign")),
                text=value,
            )
        )
    repeated = next((shot for shot in shots if shot["kind"] == "repeat"), None)
    if repeated is not None:
        original = next(
            shot for shot in shots if shot["kind"] == "source" and shot["segment_id"] == repeated["derived_from"]
        )
        label = rng.choice((f"TAKE {_random_code(rng)}", f"{rng.choice(NAMES)} CAM", f"${rng.randint(5, 75)}", f"{rng.choice(WORDS)} {rng.randint(2, 49)}"))
        repeat_style_name = rng.choice(("watermark", "price_tag_ui_chip", "corner_caption"))
        repeat_style = _text_style(rng, label, repeat_style_name)
        for suffix, shot in (("original", original), ("repeat", repeated)):
            start, end = _random_interval_in_shot(rng, shot, rng.uniform(1.25, 2.4))
            items.append(
                _interval(
                    start,
                    end,
                    id=f"repeat_label_{suffix}",
                    role="sign",
                    text=label,
                    **deepcopy(repeat_style),
                    observability=_text_observability(start, end, tier, flash=False),
                )
            )
    if plan.get("misleading_subtitle"):
        target = shots[rng.randrange(len(shots))]
        start, end = _random_interval_in_shot(rng, target, rng.uniform(1.7, 3.2))
        referenced = dialogue_plans[rng.randrange(len(dialogue_plans))]
        term_a, term_b = referenced["reference_terms"]
        subtitle_text = f"Actually, {term_a} was nowhere near {term_b}."
        items.append(
            _text_item(rng, start, end, tier=tier, style="subtitle_strip", id="misleading_subtitle", role="subtitle", text=subtitle_text, misleading=True, references_dialogue_id=referenced["id"], reference_terms=[term_a, term_b])
        )
    if plan.get("audio_visual_contradiction"):
        alarm_frame = int(plan["contradiction_frame"])
        quiet_text = rng.choice(("ALL QUIET", "SILENT ROOM", "NO ALARM", "CALM ZONE"))
        quiet_length = rng.randint(round(1.25 * FPS), round(2.75 * FPS))
        quiet_start = max(0, alarm_frame - rng.randint(0, quiet_length - 1))
        quiet_end = min(int(plan["duration_frames"]), quiet_start + quiet_length)
        items.append(
            _text_item(rng, quiet_start, quiet_end, tier=tier, style="centered_title", id="quiet_status", role="sign", text=quiet_text, contradiction=True)
        )
    if tier == 3 and random.Random(int(plan["seed"])).random() < 0.70:
        tiny_text = rng.choice((rng.choice(NAMES), _random_code(rng), rng.choice(WORDS)))
        tiny_shot = shots[rng.randrange(len(shots))]
        tiny_frames = rng.randint(3, 6)
        tiny_start, _ = _random_interval_in_shot(rng, tiny_shot, tiny_frames / FPS)
        items.append(
            _text_item(rng, tiny_start, tiny_start + tiny_frames, tier=tier, style="corner_caption", flash=True, id="fleeting_text", role="caption", text=tiny_text, ephemeral=True)
        )
    return _resolve_text_collisions(items)


def _build_dialogue(plan: Mapping[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    count = int(plan["dialogue_count"])
    duration = int(plan["duration_frames"])
    entries: list[dict[str, Any]] = []
    starts: list[int] = []
    earliest, latest = FPS, max(FPS, duration - 5 * FPS)
    attempts = 0
    while len(starts) < count and attempts < 500:
        attempts += 1
        candidate = rng.randint(earliest, latest)
        if all(abs(candidate - old) >= round(2.75 * FPS) for old in starts):
            starts.append(candidate)
    if len(starts) < count:
        starts = [round(earliest + (latest - earliest) * (index + 1) / (count + 1)) for index in range(count)]
    for index, start_frame in enumerate(sorted(starts)):
        name = rng.choice(NAMES)
        obj = rng.choice(OBJECTS)
        color = rng.choice(COLORS)
        place = rng.choice(PLACES)
        number = rng.randint(2, 98)
        time = f"{rng.randint(1, 12)}:{rng.randrange(0, 60):02d}"
        lines = (
            (f"{name} left the {color} {obj} beside the {place} entrance.", [name, obj]),
            (f"Did {name} move the {color} {obj} into the {place}?", [name, place]),
            (f"Please deliver {number} {color} parcels to the {place} before noon.", [color, place]),
            (f"I counted {number} people waiting outside the {place} this morning.", [str(number), place]),
            (f"At {time}, {name} plans to inspect the {obj} near the {place}.", [name, obj]),
            (f"Could you ask {name} whether the {color} {obj} belongs here?", [name, obj]),
            (f"Which route takes {name} from the {place} to the studio?", [name, place]),
            (f"The {color} {obj} costs {number} dollars at the {place} today.", [obj, place]),
            (f"Leave the {obj} with {name}, then meet me at the {place}.", [name, obj]),
            (f"Why did {name} carry the {obj} past the {place} twice?", [name, place]),
            (f"Our reservation at the {place} starts promptly at {time} tonight.", [place, time]),
            (f"{name} says the {place} keeps exactly {number} {color} tickets available.", [name, place]),
            (f"I wonder if {name} found the {obj} behind the {place} counter.", [name, obj]),
            (f"Take {number} steps toward the {place} and look for the {color} {obj}.", [place, obj]),
            (f"The message for {name} mentions both the {obj} and the {place}.", [name, obj]),
            (f"Is the {color} {obj} still waiting for {name} near the {place}?", [name, place]),
        )
        line, reference_terms = rng.choice(lines)
        assert 6 <= len(line.rstrip(".?!").split()) <= 14
        entries.append(
            {
                "id": f"dialogue_{index + 1}",
                "speaker": "Narrator",
                "start_frame": start_frame,
                "text": line,
                "voice": rng.choice(TTS_VOICES),
                "speed": rng.choice(TTS_SPEEDS),
                "reference_terms": reference_terms,
            }
        )
    return entries


def _build_audio_events(plan: Mapping[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    probability = float(plan["audio_probability"])
    kinds = [kind for kind in EVENT_DURATIONS if rng.random() < probability]
    if plan.get("audio_visual_contradiction") and "alarm" not in kinds:
        kinds.append("alarm")
    duration = int(plan["duration_frames"])
    entries: list[dict[str, Any]] = []
    used: list[int] = []
    for kind in kinds:
        if kind == "alarm" and plan.get("audio_visual_contradiction"):
            frame = int(plan["contradiction_frame"])
        else:
            candidates = [
                value
                for value in range(FPS, max(FPS + 1, duration - 2 * FPS))
                if all(abs(value - old) >= FPS for old in used)
            ]
            frame = rng.choice(candidates) if candidates else rng.randint(FPS, max(FPS, duration - 2 * FPS))
        used.append(frame)
        length = round(EVENT_DURATIONS[kind] * AUDIO_RATE)
        start_sample = round(frame / FPS * AUDIO_RATE)
        entries.append(
            {
                **_at(frame),
                "start_sample": start_sample,
                "end_sample": start_sample + length,
                "end": round((start_sample + length) / AUDIO_RATE, 6),
                "kind": kind,
                "relation": (
                    "contradicts_image"
                    if kind == "alarm" and plan.get("audio_visual_contradiction")
                    else "overlay"
                ),
            }
        )
    return entries


def _finalize_dialogue(
    plans: list[dict[str, Any]],
    synthesizer: Callable[..., SpeechClip],
) -> tuple[list[dict[str, Any]], list[SpeechClip]]:
    dialogue: list[dict[str, Any]] = []
    clips: list[SpeechClip] = []
    for item in plans:
        clip = synthesizer(item["text"], voice=item["voice"], speed=item["speed"], model=DEFAULT_MODEL)
        if clip.sample_rate != AUDIO_RATE:
            raise RuntimeError(f"TTS sample rate {clip.sample_rate} does not match {AUDIO_RATE}")
        start_sample = round(item["start_frame"] / FPS * AUDIO_RATE)
        end_sample = start_sample + len(clip.samples)
        normalized = clip.samples.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(normalized * normalized))) if normalized.size else 0.0
        rms_dbfs = 20 * np.log10(max(rms, 1e-12))
        dialogue.append(
            {
                "id": item["id"],
                "speaker": item["speaker"],
                "start_frame": item["start_frame"],
                "start": round(start_sample / AUDIO_RATE, 6),
                "end": round(end_sample / AUDIO_RATE, 6),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "text": item["text"],
                "tts": {
                    "engine": DEFAULT_MODEL,
                    "voice": item["voice"],
                    "speed": item["speed"],
                    "rms_dbfs": round(float(rms_dbfs), 3),
                    "mix_gain_db": round(20 * np.log10(TTS_MIX_GAIN), 3),
                },
            }
        )
        clips.append(clip)
    return dialogue, clips


def _fit_dialogue_to_duration(
    plans: list[dict[str, Any]],
    dialogue: list[dict[str, Any]],
    clips: list[SpeechClip],
    duration_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[SpeechClip]]:
    """Shift overflowing lines earlier, or deterministically drop an unfit tail."""
    duration_samples = round(duration_frames / FPS * AUDIO_RATE)
    samples_per_frame = AUDIO_RATE // FPS
    fitted_plans = list(plans)
    fitted_dialogue = list(dialogue)
    fitted_clips = list(clips)

    while fitted_dialogue:
        overflow_index = next(
            (
                index
                for index, item in enumerate(fitted_dialogue)
                if int(item["end_sample"]) > duration_samples
            ),
            None,
        )
        if overflow_index is None:
            return fitted_plans, fitted_dialogue, fitted_clips

        clip_samples = len(fitted_clips[overflow_index].samples)
        latest_start_frame = (duration_samples - clip_samples) // samples_per_frame
        if latest_start_frame >= 0:
            plan_item = {**fitted_plans[overflow_index], "start_frame": latest_start_frame}
            dialogue_item = dict(fitted_dialogue[overflow_index])
            start_sample = latest_start_frame * samples_per_frame
            dialogue_item.update(
                {
                    "start_frame": latest_start_frame,
                    "start": round(start_sample / AUDIO_RATE, 6),
                    "end": round((start_sample + clip_samples) / AUDIO_RATE, 6),
                    "start_sample": start_sample,
                    "end_sample": start_sample + clip_samples,
                }
            )
            fitted_plans[overflow_index] = plan_item
            fitted_dialogue[overflow_index] = dialogue_item
            continue

        # A clip longer than the entire scene cannot be shifted into bounds.
        # Drop from the tail so the choice remains deterministic and truth/audio agree.
        fitted_plans.pop()
        fitted_dialogue.pop()
        fitted_clips.pop()

    raise RuntimeError("all dialogue lines exceed recomposed scene duration")


def _events(
    plan: Mapping[str, Any],
    dialogue: list[dict[str, Any]],
    text: list[dict[str, Any]],
    audio: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, shot in enumerate(plan["shots"]):
        frame = int(shot["start_frame"])
        if index:
            events.append(_at(frame, action="cut", shot=shot["id"]))
        if shot["kind"] == "repeat":
            events.append(_at(frame, action="repeat", segment=shot["derived_from"]))
        elif shot["kind"] == "freeze":
            events.append(_at(frame, action="freeze", segment=shot["derived_from"]))
        elif shot["kind"] == "foreign":
            events.append(_at(frame, action="insert_foreign", source_id=shot["source_id"]))
        if float(shot["playback_rate"]) != 1.0:
            events.append(_at(frame, action="speed_change", rate=shot["playback_rate"]))
        if shot["mirror"]:
            events.append(_at(frame, action="mirror", segment=shot["segment_id"]))
        if shot["tint"]:
            events.append(_at(frame, action="tint", tint=shot["tint"]))
    events.extend(_at(item["start_frame"], action="dialogue", anchor_id=item["id"]) for item in dialogue)
    events.extend(_at(item["start_frame"], action="text", anchor_id=item["id"]) for item in text)
    events.extend(_at(item["frame"], action="audio", anchor_id=item["kind"]) for item in audio)
    return sorted(events, key=lambda item: (item["frame"], item["action"], str(item.get("anchor_id", ""))))


def _intentional_errors(
    plan: Mapping[str, Any], text: list[dict[str, Any]], audio: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for shot in plan["shots"]:
        frame = int(shot["start_frame"])
        if shot["kind"] == "repeat":
            errors.append(_at(frame, type="repeated_shot", segment=shot["derived_from"]))
        elif shot["kind"] == "foreign":
            errors.append(_at(frame, type="foreign_shot", source_id=shot["source_id"]))
        if shot["tint"]:
            errors.append(_at(frame, type="tint_change", segment=shot["segment_id"]))
    subtitle = next((item for item in text if item.get("misleading")), None)
    if subtitle is not None:
        errors.append(
            _at(subtitle["start_frame"], type="misleading_subtitle", text_id=subtitle["id"])
        )
    alarm = next((item for item in audio if item.get("relation") == "contradicts_image"), None)
    quiet = next((item for item in text if item.get("contradiction")), None)
    if alarm is not None and quiet is not None:
        errors.append(
            _at(
                alarm["frame"],
                type="audio_visual_contradiction",
                audio="alarm",
                image=f"{quiet['text']} sign",
            )
        )
    return sorted(errors, key=lambda item: (item["frame"], item["type"]))


def _qa(
    plan: Mapping[str, Any],
    dialogue: list[dict[str, Any]],
    text: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alpha = next(item for item in text if item["id"] == "marker_alpha")
    beta = next(item for item in text if item["id"] == "marker_beta")
    code = next(item for item in text if item["id"] == "code")
    first_dialogue = dialogue[0]
    foreign = any(shot["kind"] == "foreign" for shot in plan["shots"])
    qa = [
        {"id": "cuts", "q": "How many hard cuts are there?", "a": str(len(plan["shots"]) - 1), "type": "count", "evidence_frames": [shot["start_frame"] for shot in plan["shots"][1:]], "references": []},
        {"id": "text_at_time", "q": f"What exact 3-to-6-character code is shown at {seconds((code['start_frame'] + code['end_frame']) // 2):.3f} seconds?", "a": code["text"], "type": "ocr", "evidence_frames": [(code["start_frame"] + code["end_frame"]) // 2], "references": []},
        {"id": "foreign", "q": "Is there a shot from a different source video?", "a": "yes" if foreign else "no", "type": "continuity", "evidence_frames": [shot["start_frame"] for shot in plan["shots"] if shot["kind"] == "foreign"], "references": []},
        {"id": "order", "q": f"Which exact text appears first: {alpha['text']} or {beta['text']}?", "a": min((alpha, beta), key=lambda item: (item["start_frame"], item["id"]))["text"], "type": "temporal", "evidence_frames": [alpha["start_frame"], beta["start_frame"]], "references": []},
        {"id": "said_at_time", "q": f"What exact scripted line begins at {seconds(first_dialogue['start_frame']):.3f} seconds?", "a": first_dialogue["text"], "type": "dialogue", "evidence_frames": [first_dialogue["start_frame"]], "references": []},
    ]
    repeated = next((item for item in text if item["id"] == "repeat_label_repeat"), None)
    if repeated is not None:
        qa.append({"id": "repeated", "q": "Which labeled segment is repeated?", "a": repeated["text"], "type": "continuity", "evidence_frames": [item["start_frame"] for item in text if item["text"] == repeated["text"]], "references": []})
    return qa


def build_scene_from_plan(
    plan: Mapping[str, Any],
    manifest_path: Path,
    *,
    synthesizer: Callable[..., SpeechClip] = synthesize_openai,
) -> tuple[dict[str, Any], list[SpeechClip]]:
    seed = int(plan["seed"])
    dialogue_plans = _build_dialogue(plan, random.Random(seed ^ 0xD1A109))
    dialogue, clips = _finalize_dialogue(dialogue_plans, synthesizer)
    duration_frames = int(plan["duration_frames"])
    dialogue_plans, dialogue, clips = _fit_dialogue_to_duration(
        dialogue_plans, dialogue, clips, duration_frames
    )
    text = _build_text(plan, random.Random(seed ^ 0x7E87), dialogue_plans)
    audio_events = _build_audio_events(plan, random.Random(seed ^ 0xA0D10))
    events = _events(plan, dialogue, text, audio_events)
    errors = _intentional_errors(plan, text, audio_events)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_gain_db = float(plan.get("source_gain_db", SOURCE_GAIN_DB))
    quietest_tts_dbfs = min(
        float(item["tts"]["rms_dbfs"]) + float(item["tts"]["mix_gain_db"])
        for item in dialogue
    )
    duck_gain_db = round(
        min(DIALOGUE_DUCK_GAIN_DB, quietest_tts_dbfs - MIN_DIALOGUE_SOURCE_SNR_DB - 0.5),
        3,
    )
    for item in dialogue:
        mixed_tts_dbfs = float(item["tts"]["rms_dbfs"]) + float(item["tts"]["mix_gain_db"])
        item["tts"]["source_snr_lower_bound_db"] = round(mixed_tts_dbfs - duck_gain_db, 3)
    scene: dict[str, Any] = {
        "schema_version": "1.5",
        "renderer_version": "witness-recompose-0.3.1",
        "seed": int(plan["seed"]),
        "difficulty": int(plan["tier"]),
        "duration_frames": duration_frames,
        "duration": seconds(duration_frames),
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "timebase": "frame timestamps are canonical; intervals are [start_frame, end_frame)",
        "debug_labels": False,
        "summary": "Real Creative Commons footage recomposed with deterministic edits and exact-truth overlays.",
        "source": {**deepcopy(plan["source"]), "manifest_sha256": manifest_sha256},
        "foreign_sources": [deepcopy(plan["foreign_source"])] if plan.get("foreign_source") else [],
        "foreign_selection": deepcopy(plan.get("foreign_selection", {})),
        "actors": [],
        "objects": [],
        "shots": deepcopy(plan["shots"]),
        "events": events,
        "dialogue": dialogue,
        "on_screen_text": text,
        "audio_events": audio_events,
        "intentional_errors": errors,
        "qa": _qa(plan, dialogue, text),
        "audio": {
            "sample_rate": AUDIO_RATE,
            "channels": 1,
            "tts_engine": DEFAULT_MODEL,
            "dialogue_truth": "scripted_overlay_only",
            "source_audio_role": "background",
            "source_mix": {
                "gain_db": source_gain_db,
                "dialogue_duck_gain_db": duck_gain_db,
                "dialogue_duck_padding_seconds": DIALOGUE_DUCK_PADDING_SECONDS,
                "minimum_dialogue_source_snr_db": MIN_DIALOGUE_SOURCE_SNR_DB,
                "snr_method": "conservative_tts_rms_vs_ducked_source_peak_bound",
            },
        },
        "validation_checks": [
            *(
                {"kind": "hard_cut", "frame": shot["start_frame"], "minimum_mean_difference": 2.0}
                for shot in plan["shots"][1:]
            ),
            *(
                {"kind": "text_presence", "text_id": item["id"], "frame": item["observability"].get("sample_frame", (item["start_frame"] + item["end_frame"]) // 2), "bbox": item["bbox"], "expected_color": item["overlay_color"] if item["background_box"] else item["font_color"], "maximum_color_distance": min(220.0, round(442 * (1 - float(item["opacity"])) + 35, 1)) if item["background_box"] else min(190.0, round(442 * (1 - float(item["opacity"])) + 55, 1)), "pixel_statistic": "percentile_25" if item["background_box"] else "nearest"}
                for item in text
            ),
            *(
                {"kind": "short_text", "text_id": item["id"], "start_frame": item["start_frame"], "end_frame": item["end_frame"], "maximum_duration_ms": 250}
                for item in text if item.get("ephemeral")
            ),
            *(
                {
                    "kind": "dialogue_source_snr",
                    "dialogue_id": item["id"],
                    "lower_bound_db": item["tts"]["source_snr_lower_bound_db"],
                    "minimum_db": MIN_DIALOGUE_SOURCE_SNR_DB,
                }
                for item in dialogue
            ),
        ],
        "provenance": {
            "generator": "witness.recompose.generator.build_scene_from_plan",
            "seed": int(plan["seed"]),
            "manifest_sha256": manifest_sha256,
            "sha256_basis": "canonical JSON excluding provenance.scene_sha256",
        },
    }
    canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    scene["provenance"]["scene_sha256"] = hashlib.sha256(canonical).hexdigest()
    return scene, clips


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = AUDIO_RATE) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(samples.astype("<i2", copy=False).tobytes())


def _escape_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")


def _source_paths(plan: Mapping[str, Any], manifest_path: Path) -> dict[str, Path]:
    items = [plan["source"]]
    if plan.get("foreign_source"):
        items.append(plan["foreign_source"])
    paths = {
        str(item["id"]): (manifest_path.parent / str(item["path"])).resolve()
        for item in items
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source video not found: {missing[0]}")
    return paths


def _decode_planned_frame(
    shot: Mapping[str, Any],
    path: Path,
    local_frame: int,
) -> np.ndarray:
    if shot["kind"] == "freeze":
        timestamp = float(shot["source_start"])
    else:
        timestamp = float(shot["source_start"]) + local_frame / FPS * float(shot["playback_rate"])
    filters = [
        "scale=640:360:force_original_aspect_ratio=increase",
        "crop=640:360",
        "setsar=1",
    ]
    if shot.get("mirror"):
        filters.append("hflip")
    if shot.get("tint"):
        filters.append("colorbalance=rs=.18:gs=.04:bs=-.12")
    filters.append("format=rgb24")
    payload = subprocess.check_output(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp:.6f}", "-i", str(path), "-map", "0:v:0",
            "-frames:v", "1", "-vf", ",".join(filters),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
    )
    expected = WIDTH * HEIGHT * 3
    if len(payload) != expected:
        raise RuntimeError(f"could not decode planned frame from {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)


def _planned_cut_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> float:
    before = _decode_planned_frame(
        left,
        paths[str(left["source_id"])],
        max(0, int(left["output_frames"]) - 1),
    ).astype(np.int16)
    after = _decode_planned_frame(right, paths[str(right["source_id"])], 0).astype(np.int16)
    return float(np.mean(np.abs(after - before)))


def _candidate_source_starts(
    plan: Mapping[str, Any],
    shot: Mapping[str, Any],
) -> list[float]:
    length = float(shot["source_end"]) - float(shot["source_start"])
    if shot["kind"] == "foreign":
        source = plan["foreign_source"]
        low = min(5.0, max(0.0, float(source["duration"]) - length - 1.0))
        high = max(low, float(source["duration"]) - length - 1.0)
    elif shot["kind"] == "freeze":
        original = next(
            item for item in plan["shots"]
            if item["kind"] == "source" and item["segment_id"] == shot["derived_from"]
        )
        low = float(original["source_start"])
        high = max(low, float(original["source_end"]) - length)
    else:
        segment_number = int(str(shot["segment_id"]).rsplit("_", 1)[1])
        source = plan["source"]
        base_count = max(
            int(str(item["segment_id"]).rsplit("_", 1)[1])
            for item in plan["shots"] if item["kind"] == "source"
        )
        margin = min(5.0, float(source["duration"]) * 0.03)
        usable_start = margin
        bin_width = (float(source["duration"]) - 2 * margin) / base_count
        low = usable_start + (segment_number - 1) * bin_width
        high = max(low, usable_start + segment_number * bin_width - length)
    if high - low < 0.001:
        return [round(low, 3)]
    candidates = [round(low + (high - low) * index / 31, 3) for index in range(32)]
    current = float(shot["source_start"])
    return [value for value in candidates if value > current] + [value for value in candidates if value <= current]


def _set_shot_source_start(plan: dict[str, Any], shot: Mapping[str, Any], start: float) -> None:
    length = float(shot["source_end"]) - float(shot["source_start"])
    if shot["kind"] in {"source", "repeat"}:
        targets = [
            item for item in plan["shots"]
            if item.get("segment_id") == shot.get("segment_id") and item["kind"] in {"source", "repeat"}
        ]
    else:
        targets = [item for item in plan["shots"] if item is shot]
    for target in targets:
        target["source_start"] = round(start, 3)
        target["source_end"] = round(start + length, 6)


def _ensure_detectable_cut_plan(plan: dict[str, Any], manifest_path: Path) -> None:
    paths = _source_paths(plan, manifest_path)
    shots = plan["shots"]
    for boundary in range(1, len(shots)):
        if _planned_cut_difference(shots[boundary - 1], shots[boundary], paths) >= MIN_PLANNED_CUT_DIFFERENCE:
            continue
        target = shots[boundary]
        original_start = float(target["source_start"])
        for candidate in _candidate_source_starts(plan, target):
            _set_shot_source_start(plan, target, candidate)
            if _planned_cut_difference(shots[boundary - 1], target, paths) < MIN_PLANNED_CUT_DIFFERENCE:
                continue
            if boundary + 1 < len(shots) and _planned_cut_difference(target, shots[boundary + 1], paths) < MIN_PLANNED_CUT_DIFFERENCE:
                continue
            break
        else:
            _set_shot_source_start(plan, target, original_start)
            raise RuntimeError(f"could not construct a detectable cut at shot {boundary + 1}")


def _ensure_text_contrast(scene: dict[str, Any], manifest_path: Path) -> None:
    source_items = [scene["source"], *scene.get("foreign_sources", [])]
    paths = {
        str(item["id"]): (manifest_path.parent / str(item["path"])).resolve()
        for item in source_items
    }
    checks = {
        item["text_id"]: item
        for item in scene["validation_checks"]
        if item.get("kind") == "text_presence"
    }
    for item in scene["on_screen_text"]:
        frame = int(item["observability"].get("sample_frame", item["start_frame"]))
        shot = next(value for value in scene["shots"] if value["start_frame"] <= frame < value["end_frame"])
        background = _decode_planned_frame(
            shot,
            paths[str(shot["source_id"])],
            frame - int(shot["start_frame"]),
        )
        x0, y0, x1, y1 = (int(value) for value in item["bbox"])
        patch = background[y0:y1, x0:x1].reshape(-1, 3)
        local = np.median(patch, axis=0)
        current_key = "overlay_color" if item["background_box"] else "font_color"
        current = np.array(
            [int(str(item[current_key]).lstrip("#")[index:index + 2], 16) for index in (0, 2, 4)],
            dtype=np.float64,
        )
        if float(np.linalg.norm(current - local)) >= MIN_TEXT_BACKGROUND_CONTRAST:
            continue
        black = np.zeros(3, dtype=np.float64)
        white = np.full(3, 255.0, dtype=np.float64)
        contrasting = "#000000" if np.linalg.norm(black - local) >= np.linalg.norm(white - local) else "#ffffff"
        opposite = "#ffffff" if contrasting == "#000000" else "#000000"
        if item["background_box"]:
            item["overlay_color"] = contrasting
            item["font_color"] = opposite
            item["outline_color"] = contrasting
            item["opacity"] = 1.0
        else:
            item["font_color"] = contrasting
            item["outline_color"] = opposite
            item["opacity"] = 1.0
        check = checks[item["id"]]
        check["expected_color"] = item["overlay_color"] if item["background_box"] else item["font_color"]
        check["maximum_color_distance"] = 35.0 if item["background_box"] else 65.0

    scene["provenance"].pop("scene_sha256", None)
    canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    scene["provenance"]["scene_sha256"] = hashlib.sha256(canonical).hexdigest()


def _replan_if_video_stream_is_shorter(
    plan: dict[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    seed: int,
    tier: int,
) -> dict[str, Any]:
    used_ids = {str(item["source_id"]) for item in plan["shots"]}
    effective_manifest = deepcopy(manifest)
    changed = False
    ffprobe = FFMPEG.with_name("ffprobe")
    for item in effective_manifest["videos"]:
        if str(item["id"]) not in used_ids:
            continue
        path = (manifest_path.parent / str(item["path"])).resolve()
        raw = subprocess.check_output(
            [
                str(ffprobe), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", str(path),
            ],
            text=True,
        ).strip()
        video_duration = float(raw)
        declared_duration = float(item["duration"])
        selected_end = max(
            float(shot["source_end"])
            for shot in plan["shots"] if str(shot["source_id"]) == str(item["id"])
        )
        if selected_end > video_duration - 1 / FPS:
            item["duration"] = min(declared_duration, video_duration)
            changed = True
    if not changed:
        return plan
    try:
        return build_edit_plan(seed, tier, effective_manifest)
    except ValueError as first_error:
        videos = sorted(effective_manifest["videos"], key=lambda item: str(item["id"]))
        original_id = str(plan["source"]["id"])
        original_index = next(index for index, item in enumerate(videos) if str(item["id"]) == original_id)
        for offset in range(1, len(videos)):
            candidate = videos[(original_index + offset) % len(videos)]
            candidate_path = (manifest_path.parent / str(candidate["path"])).resolve()
            raw = subprocess.check_output(
                [
                    str(ffprobe), "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1",
                    str(candidate_path),
                ],
                text=True,
            ).strip()
            candidate["duration"] = min(float(candidate["duration"]), float(raw))
            try:
                return build_edit_plan(seed, tier, effective_manifest, source_id=str(candidate["id"]))
            except ValueError:
                continue
        raise first_error


def render_recomposition(
    scene: Mapping[str, Any],
    clips: list[SpeechClip],
    manifest_path: Path,
    target: Path,
) -> None:
    """Render a v1.3+ scene with one ffmpeg filter graph."""

    source_by_id = {str(scene["source"]["id"]): scene["source"]}
    source_by_id.update({str(item["id"]): item for item in scene.get("foreign_sources", [])})
    paths = {
        source_id: (manifest_path.parent / str(item["path"])).resolve()
        for source_id, item in source_by_id.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="witness-recompose-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y"]
        graph: list[str] = []
        concat_labels: list[str] = []
        for index, shot in enumerate(scene["shots"]):
            input_duration = max(1 / FPS, float(shot["source_end"]) - float(shot["source_start"]))
            command.extend(["-ss", f"{float(shot['source_start']):.6f}", "-t", f"{input_duration:.6f}", "-i", str(paths[str(shot["source_id"])])])
            frames = int(shot["output_frames"])
            duration = frames / FPS
            video_filters = ["setpts=PTS-STARTPTS"]
            if shot["kind"] == "freeze":
                video_filters.extend(["trim=end_frame=1", f"tpad=stop_mode=clone:stop_duration={duration:.6f}"])
            elif float(shot["playback_rate"]) != 1.0:
                video_filters.append(f"setpts=PTS/{float(shot['playback_rate']):.6f}")
            video_filters.extend(
                [
                    "fps=24",
                    "scale=640:360:force_original_aspect_ratio=increase",
                    "crop=640:360",
                    "setsar=1",
                ]
            )
            if shot["mirror"]:
                video_filters.append("hflip")
            if shot["tint"]:
                video_filters.append("colorbalance=rs=.18:gs=.04:bs=-.12")
            video_filters.extend([f"trim=end_frame={frames}", "setpts=PTS-STARTPTS", "format=yuv420p"])
            graph.append(f"[{index}:v]{','.join(video_filters)}[v{index}]")
            if shot["kind"] == "freeze":
                graph.append(f"anullsrc=r={AUDIO_RATE}:cl=mono,atrim=duration={duration:.6f},asetpts=PTS-STARTPTS[a{index}]")
            else:
                audio_filters = ["asetpts=PTS-STARTPTS"]
                if float(shot["playback_rate"]) != 1.0:
                    audio_filters.append(f"atempo={float(shot['playback_rate']):.6f}")
                audio_filters.extend([f"aresample={AUDIO_RATE}", "aformat=sample_fmts=fltp:channel_layouts=mono", "apad", f"atrim=duration={duration:.6f}", "asetpts=PTS-STARTPTS"])
                graph.append(f"[{index}:a]{','.join(audio_filters)}[a{index}]")
            concat_labels.append(f"[v{index}][a{index}]")
        graph.append(f"{''.join(concat_labels)}concat=n={len(scene['shots'])}:v=1:a=1[vbase][abase]")

        previous = "vbase"
        for index, item in enumerate(scene["on_screen_text"]):
            current = f"vtext{index}"
            x0, y0, x1, y1 = item["bbox"]
            color = str(item["overlay_color"]).lstrip("#")
            font_color = str(item["font_color"])
            opacity = float(item["opacity"])
            font_size = int(item["font_size"])
            padding = int(item["padding"])
            font_file = Path(str(item["font_file"]))
            if not font_file.is_file():
                raise FileNotFoundError(font_file)
            enable = f"between(n,{item['start_frame']},{item['end_frame'] - 1})"
            box_filter = (
                f"drawbox=x={x0}:y={y0}:w={x1-x0}:h={y1-y0}:color=0x{color}@{opacity:.2f}:t=fill:enable='{enable}',"
                if item["background_box"]
                else ""
            )
            graph.append(
                f"[{previous}]{box_filter}drawtext=fontfile={font_file}:text='{_escape_drawtext(item['text'])}':x={x0+padding}:y={y0+padding}:fontsize={font_size}:fontcolor={font_color}@{opacity:.2f}:borderw={int(item['outline_width'])}:bordercolor={item['outline_color']}@{opacity:.2f}:shadowx={int(item['shadow_x'])}:shadowy={int(item['shadow_y'])}:shadowcolor={item['shadow_color']}@{opacity:.2f}:enable='{enable}'[{current}]"
            )
            previous = current
        graph.append(f"[{previous}]trim=end_frame={scene['duration_frames']},setpts=PTS-STARTPTS[vout]")

        next_input = len(scene["shots"])
        extras: list[tuple[int, int, float]] = []
        for index, (item, clip) in enumerate(zip(scene["dialogue"], clips)):
            path = temp_dir / f"dialogue_{index}.wav"
            _write_wav(path, clip.samples)
            command.extend(["-i", str(path)])
            extras.append((next_input, int(item["start_sample"]), TTS_MIX_GAIN))
            next_input += 1
        for index, item in enumerate(scene["audio_events"]):
            path = temp_dir / f"event_{index}.wav"
            samples = _event_sound(item["kind"], item["end_sample"] - item["start_sample"], AUDIO_RATE)
            _write_wav(path, (samples * 32767).astype(np.int16))
            command.extend(["-i", str(path)])
            extras.append((next_input, int(item["start_sample"]), 0.72))
            next_input += 1

        total_samples = round(int(scene["duration_frames"]) / FPS * AUDIO_RATE)
        source_mix = scene["audio"]["source_mix"]
        source_gain = 10 ** (float(source_mix["gain_db"]) / 20)
        duck_relative_db = float(source_mix["dialogue_duck_gain_db"]) - float(source_mix["gain_db"])
        duck_relative_gain = 10 ** (duck_relative_db / 20)
        padding = float(source_mix["dialogue_duck_padding_seconds"])
        source_filters = [f"volume={source_gain:.9f}"]
        for item in scene["dialogue"]:
            start = max(0.0, float(item["start"]) - padding)
            end = min(float(scene["duration"]), float(item["end"]) + padding)
            source_filters.append(
                f"volume={duck_relative_gain:.9f}:enable='between(t,{start:.6f},{end:.6f})'"
            )
        source_filters.extend(["apad", f"atrim=end_sample={total_samples}"])
        graph.append(f"[abase]{','.join(source_filters)}[abasefull]")
        mix_labels = ["[abasefull]"]
        for index, (input_index, delay_samples, volume) in enumerate(extras):
            label = f"ax{index}"
            graph.append(
                f"[{input_index}:a]aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=mono,volume={volume},adelay={delay_samples}S:all=1,apad,atrim=end_sample={total_samples}[{label}]"
            )
            mix_labels.append(f"[{label}]")
        graph.append(
            f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0,alimiter=limit=0.95[aout]"
        )
        filter_graph = ";".join(graph)
        rendered = temp_dir / "rendered.mp4"
        command.extend(
            [
                "-filter_complex", filter_graph,
                "-map", "[vout]", "-map", "[aout]",
                "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0",
                "-c:a", "aac", "-b:a", "128k", "-ar", str(AUDIO_RATE),
                "-r", str(FPS), "-frames:v", str(scene["duration_frames"]),
                "-movflags", "+faststart", str(rendered),
            ]
        )
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            detail = result.stderr.strip().splitlines()
            raise RuntimeError("\n".join(detail[-12:]) if detail else f"ffmpeg exited {result.returncode}")
        rendered.replace(target)


def generate_recomposition(
    seed: int,
    tier: int,
    manifest_path: Path,
    output: Path,
    *,
    synthesizer: Callable[..., SpeechClip] = synthesize_openai,
) -> tuple[Path, Path]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    plan = build_edit_plan(seed, tier, manifest)
    plan = _replan_if_video_stream_is_shorter(plan, manifest, manifest_path, seed, tier)
    _ensure_detectable_cut_plan(plan, manifest_path)
    scene, clips = build_scene_from_plan(plan, manifest_path, synthesizer=synthesizer)
    _ensure_text_contrast(scene, manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    scene_path = output / "scene.json"
    video_path = output / "video.mp4"
    scene_path.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_recomposition(scene, clips, manifest_path, video_path)
    return scene_path, video_path
