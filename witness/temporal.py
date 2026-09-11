"""Versioned scenes whose answers depend on observed interaction histories.

The layout and auxiliary observations are independent of the story. Reversing
each object's carrier history creates a matched counterfactual with identical
public questions, duration, initial state and final state.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import hmac
import json
import math
from pathlib import Path
import random
from typing import Any

from witness.scene import (AUDIO_RATE, COLORS, FPS, HEIGHT, WIDTH, _add_dialogue,
                           _at, _audio_event, _interval, seconds)
from witness.tts import SpeechClip

VERSION = "2.0"


def _rng(seed: int, domain: str) -> random.Random:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**256:
        raise ValueError("temporal seed must be an unsigned 256-bit integer")
    digest = hmac.new(seed.to_bytes(32, "big"), domain.encode(), hashlib.sha256).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _identity(seed: int, domain: str, index: int) -> str:
    key = seed.to_bytes(32, "big")
    return hmac.new(key, f"id:{domain}:{index}".encode(), hashlib.sha256).hexdigest()[:16]


def build_scene(seed: int, tier: int, *, counterfactual: bool = False,
                include_tts_timing: bool = True,
                speech_clips: list[SpeechClip] | None = None) -> dict[str, Any]:
    if tier not in (1, 2, 3):
        raise ValueError("tier must be 1, 2 or 3")
    layout, story, timing, extras = (_rng(seed, x) for x in ("layout", "story", "timing", "extras"))
    duration = {1: 96, 2: 108, 3: 120}[tier]
    scene: dict[str, Any] = {
        "schema_version": VERSION, "renderer_version": "witness-temporal-2.0",
        "seed": seed, "difficulty": tier, "duration_frames": duration * FPS,
        "duration": float(duration), "fps": FPS, "resolution": [WIDTH, HEIGHT],
        "debug_labels": False, "summary": "Observed object-transfer histories.",
        "actors": [], "objects": [], "events": [], "shots": [], "dialogue": [],
        "on_screen_text": [], "audio_events": [], "intentional_errors": [], "qa": [],
        "validation_checks": [],
        "audio": {"sample_rate": AUDIO_RATE, "channels": 1, "tts_engine": "espeak-ng"},
    }
    actor_count = 3 if tier == 1 else 4
    actor_colors = layout.sample(list(COLORS), actor_count)
    shapes = [layout.choice(("circle", "square")) for _ in range(actor_count)]
    homes = layout.sample([[80, 204], [235, 202], [400, 202], [555, 204]], actor_count)
    for i, (color, shape, home) in enumerate(zip(actor_colors, shapes, homes)):
        actor = {"id": _identity(seed, "actor", i), "name": None,
                 "name_grounded_by": None, "visual_description": f"the {color} {shape}",
                 "color": color, "shape": shape, "initial_position": home}
        scene["actors"].append(actor)
        scene["events"].append(_at(0, action="enter", actor=actor["id"], position=home))
    object_colors = layout.sample(list(COLORS), 3)
    kinds = layout.sample(["mug", "key", "book"], 3)
    positions = layout.sample([[170, 296], [325, 298], [480, 296]], 3)
    for i, (color, kind, position) in enumerate(zip(object_colors, kinds, positions)):
        scene["objects"].append({
            "id": _identity(seed, "object", i), "name": None, "name_grounded_by": None,
            "visual_description": f"the {color} {kind}", "kind": kind, "color": color,
            "initial_position": position, "initial_visible": True,
        })

    histories = []
    for _ in scene["objects"]:
        count = story.randint(6 + tier, 7 + tier)
        history = [story.randrange(actor_count) for _ in range(count)]
        def distinguishable_orders(sequence):
            denominator = math.prod(math.factorial(n) for n in Counter(sequence).values())
            return math.factorial(len(sequence)) // denominator
        # Short, unbalanced bags were guessable even with no temporal evidence.
        # Require at least 200 distinct orders even if the complete bag leaks.
        while (len(set(history)) != actor_count or history == history[::-1]
               or distinguishable_orders(history) < 200):
            history = [story.randrange(actor_count) for _ in range(count)]
        histories.append(history[::-1] if counterfactual else history)
    order = [i for i, history in enumerate(histories) for _ in history]
    story.shuffle(order)
    seen: Counter[int] = Counter()
    object_positions = [list(o["initial_position"]) for o in scene["objects"]]
    slot_frames = (scene["duration_frames"] - 7 * FPS) // len(order)
    for step, object_index in enumerate(order):
        object_def = scene["objects"][object_index]
        actor_index = histories[object_index][seen[object_index]]
        seen[object_index] += 1
        actor = scene["actors"][actor_index]
        start = 2 * FPS + step * slot_frames + timing.randrange(0, 6)
        approach = timing.randint(11, 17)
        hold = timing.randint(7, 12)
        travel = timing.randint(20, 28)
        return_time = timing.randint(10, 14)
        pick, move = start + approach, start + approach + hold
        arrive = move + travel
        drop = arrive + 4
        current = object_positions[object_index]
        last = seen[object_index] == len(histories[object_index])
        if last:
            destination = list(object_def["initial_position"])
        else:
            # Separate objects spatially; a label may never rely on an invisible
            # overlap. Destinations and actor choices use different RNG streams.
            choices = [[x, y] for x in range(100, 551, 10) for y in (270, 302)
                       if abs(x - current[0]) >= 65
                       and all(abs(x - p[0]) >= 50 for j, p in enumerate(object_positions)
                               if j != object_index)]
            destination = list(timing.choice(choices))
        # Lifting changes the object's image immediately at pick_up. Without
        # this vertical offset, "picked up" and "not yet picked up" look equal
        # until motion begins, making the annotated pickup time unknowable.
        origin_actor = [current[0] - 28, current[1] - 2]
        target_actor = [destination[0] - 28, destination[1] - 2]
        scene["events"].extend([
            _at(start, action="move", actor=actor["id"], from_position=homes[actor_index],
                to_position=origin_actor, end_frame=pick, end=seconds(pick)),
            _at(pick, action="pick_up", actor=actor["id"], object=object_def["id"]),
            _at(move, action="move", actor=actor["id"], from_position=origin_actor,
                to_position=target_actor, end_frame=arrive, end=seconds(arrive)),
            _at(drop, action="drop", actor=actor["id"], object=object_def["id"], position=destination),
            _at(drop + 5, action="move", actor=actor["id"], from_position=target_actor,
                to_position=homes[actor_index], end_frame=drop + 5 + return_time,
                end=seconds(drop + 5 + return_time)),
        ])
        object_positions[object_index] = destination
    for index, (obj, history) in enumerate(zip(scene["objects"], histories)):
        scene["qa"].append({
            "id": f"history_{index + 1}", "type": "temporal_sequence",
            "q": (f"Which actors picked up {obj['visual_description']} during the video, in chronological order? "
                  "Include every pickup, including repeats. Separate the observed actor descriptions with ' > '."),
            "a": " > ".join(scene["actors"][i]["visual_description"] for i in history),
            "object": obj["id"], "actor_sequence": [scene["actors"][i]["id"] for i in history],
            "evidence_frames": [e["frame"] for e in scene["events"]
                                if e["action"] == "pick_up" and e.get("object") == obj["id"]],
            "references": [{"entity_type": "object", "id": obj["id"],
                            "surface": obj["visual_description"], "grounded_by": "visual_description"}],
        })

    # Auxiliary channels carry independent information; no fixed dialogue or
    # sound can be recovered by knowing the action generator or a scene's tier.
    boundaries = [0, extras.randrange(8 * FPS, round(duration * .45) * FPS),
                  extras.randrange(round(duration * .55) * FPS, (duration - 8) * FPS), duration * FPS]
    crops = [[0, 0, 640, 360], [25, 15, 615, 345], [0, 0, 600, 338]]
    extras.shuffle(crops)
    for i, (begin, end, crop) in enumerate(zip(boundaries, boundaries[1:], crops)):
        scene["shots"].append(_interval(begin, end, id=f"shot{i}", camera="wide", crop=crop))
    words = ["river", "garden", "copper", "window", "forest", "silver", "harbor", "yellow"]
    text_boxes = extras.sample([[18, 18, 160, 48], [450, 18, 620, 48], [235, 25, 405, 55]], 2)
    for i, begin in enumerate(extras.sample(range(4 * FPS, (duration - 6) * FPS), 2)):
        text = f"{extras.choice(words).upper()} {extras.randrange(100, 1000)}"
        scene["on_screen_text"].append(_interval(begin, begin + extras.randint(40, 70),
            id=f"text{i}", role="sign", text=text,
            bbox=text_boxes[i]))
    dialogue = []
    for i in range(2):
        begin = round((duration * (0.20 + 0.5 * i) + extras.uniform(-2, 2)) * FPS)
        actor = extras.choice(scene["actors"])
        first, second = extras.sample(words, 2)
        text = f"The message is {first} {extras.choice(['before', 'after'])} {second}."
        dialogue.append((actor["id"], begin, text))
        if extras.random() < 0.7:
            misleading = extras.random() < 0.5
            caption = text
            if misleading:
                replacement = extras.choice([w for w in words if w not in (first, second)])
                caption = text.replace(first, replacement)
                scene["intentional_errors"].append(_at(begin, type="misleading_subtitle",
                    text_id=f"subtitle{i}", claim=caption, truth=text))
            scene["on_screen_text"].append(_interval(begin, begin + 4 * FPS,
                id=f"subtitle{i}", role="subtitle", text=caption,
                bbox=[28, 64, 612, 102], misleading=misleading))
    _add_dialogue(scene, dialogue, include_tts_timing, speech_clips)
    sound_frames: list[int] = []
    for _ in range(extras.randint(1, 3)):
        choices = [f for f in range(3 * FPS, (duration - 3) * FPS)
                   if all(abs(f - line[1]) > 4 * FPS for line in dialogue)
                   and all(abs(f - prior) > 2 * FPS for prior in sound_frames)]
        frame = extras.choice(choices)
        sound_frames.append(frame)
        scene["audio_events"].append(_audio_event(frame, extras.choice(("beep", "door_slam", "alarm")), "off_screen"))
    scene["events"].sort(key=lambda e: e["frame"])
    scene["provenance"] = {"generator": "witness.temporal.build_scene", "version": VERSION,
                           "counterfactual": counterfactual}
    return scene


def generate(seed: int, tier: int, output: Path, *, counterfactual: bool = False) -> tuple[Path, Path]:
    from witness.render import render_video
    output.mkdir(parents=True, exist_ok=True)
    clips: list[SpeechClip] = []
    scene = build_scene(seed, tier, counterfactual=counterfactual, speech_clips=clips)
    scene_path, video_path = output / "scene.json", output / "video.mp4"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    render_video(scene, video_path, speech_clips=clips)
    return scene_path, video_path
