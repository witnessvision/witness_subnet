"""Seeded scene construction and frame-accurate state evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .tts import synthesize

FPS = 24
WIDTH = 640
HEIGHT = 360
AUDIO_RATE = 22_050

COLORS = {
    "red": "#dc3f45",
    "blue": "#3577d4",
    "green": "#3aa76d",
    "yellow": "#e0ad32",
    "purple": "#8856c4",
    "orange": "#e67832",
}


def seconds(frame: int) -> float:
    return round(frame / FPS, 6)


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


def _qa_ref(entity_type: str, entity_id: str, surface: str) -> dict[str, str]:
    return {
        "entity_type": entity_type,
        "id": entity_id,
        "surface": surface,
        "grounded_by": "visual_description",
    }


def _base(seed: int, tier: int) -> tuple[dict[str, Any], random.Random]:
    if tier not in (1, 2, 3):
        raise ValueError("tier must be 1, 2, or 3")
    rng = random.Random(seed)
    frames = {1: 576, 2: 720, 3: 864}[tier] + rng.randrange(0, 4) * 24
    scene: dict[str, Any] = {
        "schema_version": "1.2",
        "renderer_version": "witness-0.1.0",
        "seed": seed,
        "difficulty": tier,
        "duration_frames": frames,
        "duration": seconds(frames),
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "timebase": "frame timestamps are canonical; intervals are [start_frame, end_frame)",
        "summary": "",
        "actors": [],
        "objects": [],
        "shots": [],
        "events": [],
        "dialogue": [],
        "on_screen_text": [],
        "audio_events": [],
        "intentional_errors": [],
        "qa": [],
        "validation_checks": [],
        "audio": {"sample_rate": AUDIO_RATE, "channels": 1, "tts_engine": "espeak-ng"},
    }
    return scene, rng


def build_scene(
    seed: int,
    tier: int,
    *,
    include_tts_timing: bool = True,
    debug_labels: bool = False,
) -> dict[str, Any]:
    """Build a deterministic JSON-compatible scene description."""

    if debug_labels and tier in (2, 3):
        raise ValueError("debug labels are forbidden at tiers 2 and 3")
    scene, rng = _base(seed, tier)
    scene["debug_labels"] = debug_labels
    palette = rng.sample(list(COLORS), 5)
    kinds = rng.sample(["mug", "key", "book"], 3)
    actor_names = rng.sample(["Ari", "Bea", "Cato", "Dina", "Eli"], 2)
    actor_a_description = f"the {palette[0]} circle"
    actor_b_description = f"the {palette[1]} square"
    object_1_description = f"the {palette[2]} {kinds[0]}"
    object_2_description = f"the {palette[3]} {kinds[0]}"

    scene["actors"].append(
        {
            "id": "A",
            "name": actor_names[0],
            "visual_description": actor_a_description,
            "name_grounded_by": "label" if debug_labels else None,
            "color": palette[0],
            "shape": "circle",
            "initial_position": [-60, 252],
        }
    )
    scene["objects"].append(
        {
            "id": "obj1",
            "name": None,
            "visual_description": object_1_description,
            "name_grounded_by": None,
            "kind": kinds[0],
            "color": palette[2],
            "initial_position": [330, 270],
            "initial_visible": True,
        }
    )

    if tier == 1:
        scene["summary"] = f"{actor_a_description.capitalize()} enters, picks up, carries, and drops {object_1_description}."
        scene["shots"] = [_interval(0, scene["duration_frames"], id="shot1", camera="wide", crop=[0, 0, 640, 360])]
        scene["events"] = [
            _at(24, actor="A", action="enter", position=[40, 252]),
            _at(72, actor="A", action="move", from_position=[40, 252], to_position=[300, 252], end_frame=168, end=seconds(168)),
            _at(180, actor="A", action="pick_up", object="obj1"),
            _at(240, actor="A", action="move", from_position=[300, 252], to_position=[500, 252], end_frame=336, end=seconds(336)),
            _at(348, actor="A", action="drop", object="obj1", position=[520, 270]),
        ]
        dialogue = [("A", 384, f"I left the {kinds[0]} by the right wall.")]
        scene["on_screen_text"] = [_interval(96, 168, id="clock", role="clock", text=f"{8 + rng.randrange(4)}:{rng.choice(['15', '30', '45'])}", bbox=[536, 18, 624, 48])]
        scene["audio_events"] = [_audio_event(456, "beep", "on_screen_timer")]
        scene["qa"] = [
            {
                "id": "temporal_1",
                "q": f"What did {actor_a_description} do after picking up {object_1_description}?",
                "a": "moved to the right wall",
                "type": "temporal",
                "evidence_frames": [180, 336],
                "references": [
                    _qa_ref("actor", "A", actor_a_description),
                    _qa_ref("object", "obj1", object_1_description),
                ],
            },
            {"id": "count_1", "q": "How many actors appear?", "a": "1", "type": "count", "evidence_frames": [72], "references": []},
            {"id": "ocr_1", "q": "What time is shown on the clock?", "a": scene["on_screen_text"][0]["text"], "type": "ocr", "evidence_frames": [120], "references": []},
            {
                "id": "causal_1",
                "q": f"Why does {object_1_description} end by the right wall?",
                "a": f"because {actor_a_description} carried and dropped it there",
                "type": "causal",
                "evidence_frames": [180, 348],
                "references": [
                    _qa_ref("object", "obj1", object_1_description),
                    _qa_ref("actor", "A", actor_a_description),
                ],
            },
        ]
        scene["validation_checks"] = [
            {"kind": "object_color", "frame": 360, "object": "obj1", "expected": palette[2], "sample_world": [520, 270]},
        ]
    else:
        scene["actors"].append(
            {
                "id": "B",
                "name": actor_names[1],
                "visual_description": actor_b_description,
                "name_grounded_by": None,
                "color": palette[1],
                "shape": "square",
                "initial_position": [690, 252],
            }
        )
        scene["objects"].append(
            {
                "id": "obj2",
                "name": None,
                "visual_description": object_2_description,
                "name_grounded_by": None,
                "kind": kinds[0],
                "color": palette[3],
                "initial_position": [420, 270],
                "initial_visible": True,
            }
        )
        cut1, cut2 = (240, 456) if tier == 2 else (216, 432)
        scene["shots"] = [
            _interval(0, cut1, id="shot1", camera="wide", crop=[0, 0, 640, 360]),
            _interval(cut1, cut2, id="shot2", camera="close", crop=[210, 100, 470, 330]),
            _interval(cut2, scene["duration_frames"], id="shot3", camera="pan", crop_start=[0, 60, 500, 341], crop_end=[140, 60, 640, 341]),
        ]
        scene["events"] = [
            _at(24, actor="A", action="enter", position=[40, 252]),
            _at(36, actor="B", action="enter", position=[600, 252]),
            _at(72, actor="A", action="move", from_position=[40, 252], to_position=[300, 252], end_frame=168, end=seconds(168)),
            _at(96 if tier == 2 else 72, actor="B", action="move", from_position=[600, 252], to_position=[440, 252], end_frame=168, end=seconds(168)),
            _at(180, actor="A", action="pick_up", object="obj1"),
            _at(204 if tier == 2 else 180, actor="B", action="pick_up", object="obj2"),
            _at(264, actor="A", action="move", from_position=[300, 252], to_position=[500, 252], end_frame=360, end=seconds(360)),
            _at(372, actor="A", action="drop", object="obj1", position=[520, 270]),
        ]
        scene["summary"] = f"{actor_a_description.capitalize()} and {actor_b_description} handle two similar {kinds[0]}s across three shots."
        dialogue = [("B", 480, f"I picked up the {palette[3]} {kinds[0]} from the table.")]
        scene["on_screen_text"] = [_interval(120, 192, id="sign", role="sign", text=rng.choice(["NORTH EXIT", "ROOM 204", "KEEP LEFT"]), bbox=[18, 18, 160, 48])]
        scene["audio_events"] = [_audio_event(408, "door_slam", "off_screen")]
        scene["events"].append(_at(300, action="change_color", object="obj1", from_color=palette[2], to_color=palette[4]))
        object_1_description = f"the {palette[4]} {kinds[0]}"
        scene["objects"][0]["visual_description"] = object_1_description
        scene["intentional_errors"].append(
            _at(300, type="continuity", object="obj1", before=palette[2], after=palette[4], description="carried object changes color across the close shot")
        )
        scene["validation_checks"] = [
            {"kind": "object_color", "frame": 600, "object": "obj1", "expected": palette[4], "sample_world": [520, 270]},
            {"kind": "hard_cut", "frame": cut1, "minimum_mean_difference": 8.0},
            {"kind": "hard_cut", "frame": cut2, "minimum_mean_difference": 8.0},
        ]
        scene["qa"] = [
            {
                "id": "temporal_1",
                "q": "Which actor picked up an object first?",
                "a": actor_a_description if tier == 2 else f"{actor_a_description} and {actor_b_description} picked them up simultaneously",
                "type": "temporal",
                "evidence_frames": [180, 204 if tier == 2 else 180],
                "references": [
                    _qa_ref("actor", "A", actor_a_description),
                    *([_qa_ref("actor", "B", actor_b_description)] if tier == 3 else []),
                ],
            },
            {"id": "count_1", "q": f"How many {kinds[0]}s are present initially?", "a": "2", "type": "count", "evidence_frames": [120], "references": []},
            {"id": "ocr_1", "q": "What does the wall sign say?", "a": scene["on_screen_text"][0]["text"], "type": "ocr", "evidence_frames": [144], "references": []},
            {
                "id": "causal_1",
                "q": f"Why is {object_1_description} by the right wall?",
                "a": f"because {actor_a_description} carried and dropped it there",
                "type": "causal",
                "evidence_frames": [300, 372],
                "references": [
                    _qa_ref("object", "obj1", object_1_description),
                    _qa_ref("actor", "A", actor_a_description),
                ],
            },
        ]

        if tier == 3:
            flash_start = 510
            scene["on_screen_text"].extend(
                [
                    _interval(flash_start, flash_start + 4, id="flash", role="code", text=str(rng.randrange(1000, 9999)), bbox=[286, 42, 354, 66]),
                    _interval(570, 642, id="subtitle", role="subtitle", text=f"SUBTITLE: The {palette[2]} {kinds[0]} vanished.", bbox=[120, 315, 520, 344], misleading=True),
                ]
            )
            scene["audio_events"].append(_audio_event(540, "alarm", "contradicts_visible_calm_scene", contradicts_image=True))
            scene["events"].extend(
                [
                    _at(520, action="vanish", object="obj2"),
                    _at(660, actor="B", action="position_jump", from_position=[440, 252], to_position=[100, 252]),
                ]
            )
            scene["intentional_errors"].extend(
                [
                    _at(520, type="continuity", object="obj2", before="visible", after="vanished", description="second object disappears without an action"),
                    _at(540, type="audio_visual_contradiction", audio="alarm", image="calm room", description="alarm sounds without a visible alarm or reaction"),
                    _at(570, type="misleading_subtitle", text_id="subtitle", claim=f"{palette[2]} {kinds[0]} vanished", truth=f"{palette[3]} {kinds[0]} vanished"),
                    _at(660, type="continuity", actor="B", before=[440, 252], after=[100, 252], description="actor jumps position")
                ]
            )
            scene["qa"][2] = {"id": "ocr_1", "q": "What four-digit code flashes briefly?", "a": scene["on_screen_text"][1]["text"], "type": "ocr", "evidence_frames": [511], "references": []}
            scene["qa"].append({"id": "causal_2", "q": "Did a visible event cause the alarm?", "a": "no", "type": "causal", "evidence_frames": [540], "references": []})
            scene["validation_checks"].append({"kind": "short_text", "start_frame": flash_start, "end_frame": flash_start + 4, "maximum_duration_ms": 200})
            dialogue = [("B", 570, f"The {palette[3]} {kinds[0]} is missing, not the {palette[2]} one.")]

    _add_dialogue(scene, dialogue, include_tts_timing)
    scene["provenance"] = {
        "generator": "witness.scene.build_scene",
        "seed": seed,
        "sha256_basis": "canonical JSON excluding provenance.scene_sha256",
    }
    canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    scene["provenance"]["scene_sha256"] = hashlib.sha256(canonical).hexdigest()
    return scene


def _add_dialogue(scene: dict[str, Any], plans: list[tuple[str, int, str]], include_timing: bool) -> None:
    for speaker, start_frame, text in plans:
        if include_timing:
            actor = next(actor for actor in scene["actors"] if actor["id"] == speaker)
            rate = 150 if actor["shape"] == "square" else 160
            clip = synthesize(text, voice="en-us", rate=rate)
            if clip.sample_rate != AUDIO_RATE:
                raise RuntimeError(f"unexpected eSpeak sample rate {clip.sample_rate}; expected {AUDIO_RATE}")
            start_sample = round(start_frame / FPS * AUDIO_RATE)
            end_sample = start_sample + len(clip.samples)
        else:
            start_sample = round(start_frame / FPS * AUDIO_RATE)
            end_sample = start_sample
            rate = 150
        scene["dialogue"].append(
            {
                "speaker": speaker,
                "start_frame": start_frame,
                "start": round(start_sample / AUDIO_RATE, 6),
                "end": round(end_sample / AUDIO_RATE, 6),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "text": text,
                "tts": {"engine": "espeak-ng", "voice": "en-us", "rate": rate},
            }
        )
        if end_sample > round(scene["duration"] * AUDIO_RATE):
            raise RuntimeError("dialogue exceeds scene duration")


def _audio_event(frame: int, kind: str, relation: str, **extra: Any) -> dict[str, Any]:
    durations = {"beep": 0.35, "door_slam": 0.55, "alarm": 1.2}
    start_sample = round(frame / FPS * AUDIO_RATE)
    length = round(durations[kind] * AUDIO_RATE)
    return {
        "frame": frame,
        "t": seconds(frame),
        "start_sample": start_sample,
        "end_sample": start_sample + length,
        "end": round((start_sample + length) / AUDIO_RATE, 6),
        "kind": kind,
        "relation": relation,
        **extra,
    }


def state_at(scene: dict[str, Any], frame: int) -> dict[str, Any]:
    """Resolve actor/object world state at the beginning of a frame."""

    actors = {
        actor["id"]: {"position": list(actor["initial_position"]), "visible": False, "carrying": None}
        for actor in scene["actors"]
    }
    objects = {
        obj["id"]: {
            "position": list(obj["initial_position"]),
            "visible": obj["initial_visible"],
            "color": obj["color"],
            "carrier": None,
        }
        for obj in scene["objects"]
    }
    for event in sorted(scene["events"], key=lambda item: item["frame"]):
        if event["frame"] > frame:
            break
        action = event["action"]
        actor_id = event.get("actor")
        object_id = event.get("object")
        if action == "enter":
            actors[actor_id]["visible"] = True
            actors[actor_id]["position"] = list(event["position"])
        elif action == "move":
            span = event["end_frame"] - event["frame"]
            ratio = min(1.0, max(0.0, (frame - event["frame"]) / span))
            start = event["from_position"]
            end = event["to_position"]
            actors[actor_id]["position"] = [start[i] + (end[i] - start[i]) * ratio for i in (0, 1)]
        elif action == "pick_up":
            old_carrier = objects[object_id]["carrier"]
            if old_carrier:
                actors[old_carrier]["carrying"] = None
            objects[object_id]["carrier"] = actor_id
            actors[actor_id]["carrying"] = object_id
        elif action == "drop":
            objects[object_id]["carrier"] = None
            objects[object_id]["position"] = list(event["position"])
            actors[actor_id]["carrying"] = None
        elif action == "change_color":
            objects[object_id]["color"] = event["to_color"]
        elif action == "vanish":
            objects[object_id]["visible"] = False
            carrier = objects[object_id]["carrier"]
            if carrier:
                actors[carrier]["carrying"] = None
            objects[object_id]["carrier"] = None
        elif action == "position_jump":
            actors[actor_id]["position"] = list(event["to_position"])
    for obj in objects.values():
        if obj["carrier"]:
            x, y = actors[obj["carrier"]]["position"]
            obj["position"] = [x + 28, y - 8]
    return {"actors": actors, "objects": objects}
