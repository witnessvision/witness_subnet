"""Fresh action/result worlds for the local grounded-video 3.0 contract.

Motion plans are private renderer state. Scored events describe visible contact,
object state changes and destinations, rather than the generator's move commands.
"""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from witness.scene import AUDIO_RATE, COLORS, FPS, _add_dialogue
from witness.temporal import _identity, _rng
from witness.tts import SpeechClip

VERSION = "3.0"
ACTIONS = ("carry", "push", "lift", "touch", "pass")
ZONES = {"left": 135, "center": 320, "right": 505}
KINDS = ("mug", "key", "book")


def event_phrase(event: dict[str, Any]) -> str:
    recipient = f" to {event['recipient']}" if event.get("recipient") else ""
    return f"{event['actor']} {event['action']}{recipient} {event['target']}"


def _track(scene, entity, begin, end, origin, destination):
    scene["motion"].append({"entity": entity, "start_frame": begin, "end_frame": end,
                            "from": list(origin), "to": list(destination)})


def build_scene(seed: int, tier: int, *, counterfactual: bool = False,
                include_tts_timing: bool = True,
                speech_clips: list[SpeechClip] | None = None) -> dict[str, Any]:
    if tier not in (1, 2, 3):
        raise ValueError("tier must be 1, 2 or 3")
    layout, story, timing, sounds = (_rng(seed, "grounded:" + x)
                                    for x in ("layout", "story", "timing", "audio"))
    base_count = {1: 4, 2: 5, 3: 6}[tier]
    counts = {kind: story.randint(base_count-1, base_count+1) for kind in KINDS}
    scene = {
        "schema_version": VERSION, "renderer_version": "witness-grounded-3.0",
        "seed": seed, "difficulty": tier, "duration_frames": 0,
        "duration": 0, "fps": FPS, "resolution": [640, 360],
        "debug_labels": False, "content_kind": "interaction_world",
        "actors": [], "objects": [], "events": [], "motion": [], "episodes": [],
        "dialogue": [], "shots": [], "on_screen_text": [], "audio_events": [],
        "intentional_errors": [], "qa": [], "validation_checks": [],
        "audio": {"sample_rate": AUDIO_RATE, "channels": 1, "tts_engine": "espeak-ng"},
        "evaluation": {"required_qa_groups": ["history", "audio_grounding"],
                       "event_fields": ["action", "actor", "object", "target", "recipient"],
                       "interval_tolerance": .35},
    }
    palette = layout.sample(list(COLORS), 3)
    homes = layout.sample([[65, 74], [320, 74], [575, 74]], 3)
    for i, name in enumerate(("A", "B", "C")):
        scene["actors"].append({"id": name, "name": None,
            "visual_description": f"actor {name}", "color": palette[i],
            "shape": layout.choice(("circle", "square")), "initial_position": homes[i]})
    object_colors = layout.sample(list(COLORS), 3)
    object_rows = layout.sample([154, 226, 298], 3)
    starts = layout.sample(list(ZONES), 3)
    for i, kind in enumerate(KINDS):
        scene["objects"].append({"id": kind, "name": None,
            "visual_description": f"the {kind}", "kind": kind, "color": object_colors[i],
            "initial_position": [ZONES[starts[i]], object_rows[i]], "initial_visible": True})
    # Each object has repeated operations; operation/outcome choices are independent
    # of appearance, speech selection, initial arrangement and camera parameters.
    plan = []
    for obj in scene["objects"]:
        count = counts[obj['id']]
        operations = [story.choice(ACTIONS) for _ in range(count)]
        while len(set(operations)) < 3:
            operations = [story.choice(ACTIONS) for _ in range(count)]
        for action in operations:
            actor = story.choice([a["id"] for a in scene["actors"]])
            recipient = story.choice([a["id"] for a in scene["actors"] if a["id"] != actor])
            plan.append({"object": obj["id"], "action": action, "actor": actor,
                         "recipient": recipient if action == "pass" else None,
                         "zone_choice": story.randrange(2)})
    story.shuffle(plan)
    if counterfactual:
        # Same initial public scene, speech schedule and timing. Change the visible
        # operation, retaining actors/objects: actor-motion or transcript templates
        # cannot determine what happened to the object.
        replacement = {"carry": "push", "push": "carry", "lift": "touch",
                       "touch": "lift", "pass": "carry"}
        for p in plan:
            p["action"] = replacement[p["action"]]
            if p["action"] != "pass":
                p["recipient"] = None
    positions = {o["id"]: list(o["initial_position"]) for o in scene["objects"]}
    actor_homes = {a["id"]: a["initial_position"] for a in scene["actors"]}
    cursor = 2 * FPS + timing.randrange(FPS)
    for index, item in enumerate(plan):
        actor, kind, action = item["actor"], item["object"], item["action"]
        begin = cursor
        contact = begin + 20
        span = timing.randint(30, 50)
        end = contact + span
        origin = positions[kind]
        origin_zone = min(ZONES, key=lambda z: abs(ZONES[z] - origin[0]))
        alternatives = [z for z in ZONES if z != origin_zone]
        target = alternatives[item["zone_choice"]] if action in {"carry", "push", "pass"} else origin_zone
        destination = [ZONES[target], origin[1]]
        actor_origin = [origin[0] - 29, origin[1] - 27]
        actor_end = [destination[0] - 29, destination[1] - 27]
        event = {"id": _identity(seed, "grounded-event", index), "start_frame": contact,
                 "end_frame": end, "start": contact / FPS, "end": end / FPS,
                 "action": action, "actor": actor, "object": kind, "target": target,
                 "recipient": item["recipient"]}
        scene["events"].append(event)
        episode = {**deepcopy(event), "approach_frame": begin,
                   "origin": list(origin), "destination": list(destination),
                   "before_zone": origin_zone, "after_zone": target}
        scene["episodes"].append(episode)
        _track(scene, actor, begin, contact, actor_homes[actor], actor_origin)
        if action in {"carry", "push"}:
            _track(scene, actor, contact + 5, end - 5, actor_origin, actor_end)
            _track(scene, kind, contact + 5, end - 5, origin, destination)
        elif action == "pass":
            receiver = item["recipient"]
            halfway = (contact + end) // 2
            # Receiver approaches from the other side. Both contacts are visible;
            # the object changes its supporting hand before the receiver moves.
            receiver_origin = [origin[0] + 30, origin[1] - 27]
            receiver_end = [destination[0] + 30, destination[1] - 27]
            _track(scene, receiver, begin, contact, actor_homes[receiver], receiver_origin)
            _track(scene, receiver, halfway + 2, end - 4, receiver_origin, receiver_end)
            _track(scene, kind, halfway + 2, end - 4, origin, destination)
            _track(scene, receiver, end + 3, end + 20, receiver_end, actor_homes[receiver])
            actor_end = actor_origin
        _track(scene, actor, end + 3, end + 20, actor_end, actor_homes[actor])
        positions[kind] = destination
        # A non-interacting actor makes an approach and retreats. Movement toward
        # an object is not itself evidence that an interaction occurred.
        idle = next(a["id"] for a in scene["actors"]
                    if a["id"] not in {actor, item["recipient"]})
        fake = [timing.randint(180, 460), timing.choice((108, 190, 263))]
        _track(scene, idle, begin + 4, contact + 6, actor_homes[idle], fake)
        _track(scene, idle, end - 4, end + 17, fake, actor_homes[idle])
        cursor = end + 20 + timing.randint(8, 30)
    duration_frames = scene['episodes'][-1]['end_frame'] + 2 * FPS + timing.randrange(13)
    scene.update(duration_frames=duration_frames, duration=duration_frames/FPS)
    for obj in scene["objects"]:
        events = [e for e in scene["events"] if e["object"] == obj["id"]]
        scene["qa"].append({"id": f"history_{obj['id']}", "type": "grounded_sequence",
            "group": "history", "q": f"List every interaction with the {obj['kind']} in chronological order. "
            "For each give 'actor action destination' (for pass: 'actor pass to recipient destination'), "
            "separated by ' > '. Actors are the visible letter badges. Actions: carry, push, lift, touch, pass. "
            "Destination is the marked left, center or right zone after the action. Ignore approaches without contact.",
            "a": " > ".join(event_phrase(e) for e in events),
            "event_ids": [e['id'] for e in events],
            "sequence": [event_phrase(e) for e in events]})
    # The audio schedule is independent of action choice. Three observations from
    # nine or more events cannot be recovered just from a fixed beat or script.
    selected = sorted(sounds.sample(range(len(plan)), 3))
    dialogue = [("A", scene["episodes"][i]["approach_frame"] - 24,
                 f"Record the {scene['episodes'][i]['object']} now.") for i in selected]
    generated_clips = speech_clips if speech_clips is not None else []
    _add_dialogue(scene, dialogue, include_tts_timing, generated_clips)
    # This is private validation evidence, never part of public_task_spec or an
    # observation response. eSpeak re-synthesis is not sample-deterministic.
    scene["audio"]["validation_pcm"] = []
    for clip in generated_clips:
        pcm = clip.samples.astype("<i2").tobytes()
        scene["audio"]["validation_pcm"].append({
            "sha256": hashlib.sha256(pcm).hexdigest(),
            "zlib_base64": base64.b64encode(zlib.compress(pcm)).decode("ascii"),
        })
    scene["qa"].append({"id": "spoken_records", "type": "grounded_sequence", "group": "audio_grounding",
        "q": "For each spoken instruction 'Record the ... now', in speech order, report the immediately following "
             "interaction with the named object as 'object actor action destination' (for pass: "
             "'object actor pass to recipient destination'). Separate instructions by ' > '. "
             "Report what visibly happened, not a guessed action from the instruction.",
        "a": " > ".join(f"{scene['events'][i]['object']} {event_phrase(scene['events'][i])}" for i in selected),
        "event_ids": [scene['events'][i]['id'] for i in selected],
        "sequence": [f"{scene['events'][i]['object']} {event_phrase(scene['events'][i])}" for i in selected]})
    # Camera and appearance parameters are nuisance factors independent of labels.
    scene["appearance"] = {"background": layout.choice(("#e6e0d6", "#d6e3e7", "#dddce8")),
        "zoom": layout.uniform(.97, 1.015), "pan": layout.uniform(-7, 7),
        "actor_palette": [palette, palette[1:] + palette[:1]],
        "palette_change_frame": duration_frames // 2 + layout.randint(-FPS, FPS),
        "texture_phase": layout.randrange(7)}
    scene["provenance"] = {"generator": "witness.grounded.build_scene", "version": VERSION,
                           "counterfactual": counterfactual}
    return scene


def state_at(scene: dict[str, Any], frame: int) -> dict[str, Any]:
    positions = {e["id"]: list(e["initial_position"])
                 for e in [*scene["actors"], *scene["objects"]]}
    for track in sorted(scene["motion"], key=lambda t: t["start_frame"]):
        if frame < track["start_frame"]:
            continue
        fraction = min(1, (frame - track["start_frame"]) /
                       max(1, track["end_frame"] - track["start_frame"]))
        positions[track["entity"]] = [a + (b - a) * fraction
                                      for a, b in zip(track["from"], track["to"])]
    active = next((e for e in scene["episodes"] if e["start_frame"] <= frame < e["end_frame"]), None)
    hands = []
    if active:
        kind, actor, action = active["object"], active["actor"], active["action"]
        if action in {"carry", "lift", "pass"}:
            positions[kind][1] -= 18
        if action == "pass" and frame >= (active["start_frame"] + active["end_frame"]) // 2:
            actor = active["recipient"]
        hands.append((actor, kind))
    return {"positions": positions, "hands": hands, "active": active}


def render_frame(scene: dict[str, Any], frame: int) -> Image.Image:
    from witness.render import _draw_object
    width, height = scene["resolution"]
    appearance = scene["appearance"]
    image = Image.new("RGB", (width, height), appearance["background"])
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    for name, x in ZONES.items():
        draw.text((x - 25, 8), name, fill="#293443", font=font)
    for x in (225, 415):
        draw.line((x, 34, x, height), fill="#a3a8aa", width=2)
    phase = appearance["texture_phase"]
    for y in range(40 + phase, height, 22):
        draw.line((0, y, width, y), fill="#c1c7c5", width=1)
    state = state_at(scene, frame)
    palette = appearance["actor_palette"][int(frame >= appearance["palette_change_frame"])]
    for i, actor in enumerate(scene["actors"]):
        x, y = state["positions"][actor["id"]]
        color = COLORS[palette[i]]
        box = (x - 21, y - 21, x + 21, y + 21)
        if actor["shape"] == "circle":
            draw.ellipse(box, fill=color, outline="#202936", width=3)
        else:
            draw.rectangle(box, fill=color, outline="#202936", width=3)
        draw.rectangle((x - 10, y - 13, x + 11, y + 13), fill="#ffffff", outline="#202936")
        draw.text((x - 7, y - 12), actor["id"], fill="#101620", font=font)
    for actor_id, object_id in state["hands"]:
        a, b = state["positions"][actor_id], state["positions"][object_id]
        # The supporting hand must not paint over the actor's identity badge.
        draw.line((a[0], a[1] + 16, b[0], b[1]), fill="#17212b", width=4)
    for obj in scene["objects"]:
        position = state["positions"][obj["id"]]
        # Shadows mark the ground; lift/carry separate the object from its shadow,
        # whereas a push keeps it at ground height throughout the displacement.
        active = state["active"]
        lift = 18 if active and active["object"] == obj["id"] and active["action"] in {"carry", "lift", "pass"} else 0
        x, y = position
        draw.ellipse((x - 13, y + lift + 8, x + 16, y + lift + 14), fill="#89918e")
        _draw_object(draw, obj["kind"], position, COLORS[obj["color"]], None, font)
    zoom = appearance["zoom"]
    drift = appearance["pan"] * np.sin(frame / scene["fps"] / 8)
    return image.transform(image.size, Image.Transform.AFFINE,
        (1 / zoom, 0, width / 2 * (1 - 1 / zoom) + drift,
         0, 1 / zoom, height / 2 * (1 - 1 / zoom)),
        resample=Image.Resampling.BILINEAR, fillcolor=appearance["background"])


def generate(seed: int, tier: int, output: Path, *, counterfactual: bool = False):
    from witness.render import render_video
    output = Path(output)
    if any((output / name).exists() for name in ('scene.json', 'video.mp4')):
        raise FileExistsError('grounded generation requires a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    clips: list[SpeechClip] = []
    scene = build_scene(seed, tier, counterfactual=counterfactual, speech_clips=clips)
    scene_path, video_path = output / "scene.json", output / "video.mp4"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    render_video(scene, video_path, speech_clips=clips)
    return scene_path, video_path


def generate_pair(seed: int, tier: int, output: Path):
    """Matched controls with identical speech samples, questions and endpoints."""
    from witness.render import render_video
    if any((Path(output)/str(index)/name).exists() for index in (0,1) for name in ('scene.json','video.mp4')):
        raise FileExistsError('grounded pair generation requires fresh output directories')
    clips: list[SpeechClip] = []
    first = build_scene(seed, tier, speech_clips=clips)
    second = build_scene(seed, tier, counterfactual=True, include_tts_timing=False)
    second['audio'] = deepcopy(first['audio'])
    second['dialogue'] = deepcopy(first['dialogue'])
    paths = []
    for index, scene in enumerate((first, second)):
        destination = Path(output) / str(index)
        destination.mkdir(parents=True, exist_ok=True)
        labels, video = destination / 'scene.json', destination / 'video.mp4'
        labels.write_text(json.dumps(scene, indent=2) + '\n')
        render_video(scene, video, speech_clips=clips)
        paths.append((labels, video))
    return paths
