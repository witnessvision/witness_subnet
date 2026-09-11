from __future__ import annotations

import json
import re

import pytest

from witness.gen import generate
from witness.scene import build_scene
from witness.validate import validate


def _canonical(scene: dict) -> str:
    return json.dumps(scene, sort_keys=True, separators=(",", ":"))


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)", text.casefold()) is not None


def test_scene_json_is_deterministic_for_seed_and_tier() -> None:
    for tier in (1, 2, 3):
        first = build_scene(7123, tier)
        second = build_scene(7123, tier)
        assert _canonical(first) == _canonical(second)
        assert _canonical(first) != _canonical(build_scene(7124, tier))


def test_tier_contracts() -> None:
    scenes = {tier: build_scene(80 + tier, tier) for tier in (1, 2, 3)}
    assert len(scenes[1]["actors"]) == len(scenes[1]["shots"]) == 1
    assert len(scenes[1]["events"]) <= 5
    assert len(scenes[2]["actors"]) == 2 and len(scenes[2]["shots"]) == 3
    assert any(error["type"] == "continuity" for error in scenes[2]["intentional_errors"])
    assert any(event["relation"] == "off_screen" for event in scenes[2]["audio_events"])
    assert any(error["type"] == "audio_visual_contradiction" for error in scenes[3]["intentional_errors"])
    assert any(error["type"] == "misleading_subtitle" for error in scenes[3]["intentional_errors"])
    flashes = [item for item in scenes[3]["on_screen_text"] if item["role"] == "code"]
    assert (flashes[0]["end_frame"] - flashes[0]["start_frame"]) / scenes[3]["fps"] <= 0.2
    assert {entry["type"] for entry in scenes[3]["qa"]} == {"temporal", "count", "ocr", "causal"}
    assert all(scene["debug_labels"] is False for scene in scenes.values())
    assert not any(
        re.search(r"\bobj\d+\b", entry[field], re.IGNORECASE)
        for scene in scenes.values()
        for entry in scene["qa"]
        for field in ("q", "a")
    )


def test_debug_labels_are_tier_one_only() -> None:
    debug_scene = build_scene(1, 1, debug_labels=True)
    assert debug_scene["debug_labels"] is True
    assert debug_scene["actors"][0]["name_grounded_by"] == "label"
    for tier in (2, 3):
        with pytest.raises(ValueError, match="forbidden"):
            build_scene(1, tier, debug_labels=True)


def test_qa_entity_references_are_observable() -> None:
    for tier in (1, 2, 3):
        scene = build_scene(900 + tier, tier)
        entities = {
            (entity_type, entity["id"]): entity
            for entity_type, group in (("actor", scene["actors"]), ("object", scene["objects"]))
            for entity in group
        }
        for entity in entities.values():
            assert set(("name", "visual_description", "name_grounded_by")) <= entity.keys()
            assert entity["name_grounded_by"] in (None, "dialogue", "label")
        for entry in scene["qa"]:
            qa_text = f'{entry["q"]} {entry["a"]}'.casefold()
            for reference in entry["references"]:
                entity = entities[(reference["entity_type"], reference["id"])]
                assert _contains_phrase(qa_text, reference["surface"])
                if reference["grounded_by"] == "visual_description":
                    assert reference["surface"] == entity["visual_description"]
                else:
                    assert reference["surface"] == entity["name"]
                    assert entity["name_grounded_by"] in ("dialogue", "label")
            for entity in entities.values():
                if entity["name"] and entity["name_grounded_by"] is None:
                    assert not _contains_phrase(qa_text, entity["name"])


def test_generated_media_passes_validation(tmp_path) -> None:
    scene_path, video_path = generate(41, 1, tmp_path / "scene_41")
    assert validate(scene_path, video_path) == []


def test_generation_renders_original_speech_without_a_second_synthesis(tmp_path, monkeypatch):
    from witness import render
    monkeypatch.setattr(render, "synthesize", lambda *a, **kw: pytest.fail("synthesized speech twice"))
    scene_path, video_path = generate(41, 1, tmp_path / "scene")
    assert validate(scene_path, video_path) == []


def test_reused_speech_still_enforces_the_timing_contract():
    from witness.render import compose_audio
    from witness.tts import SpeechClip
    clips = []
    scene = build_scene(41, 1, speech_clips=clips)
    compose_audio(scene, speech_clips=clips)
    first = clips[0]
    clips[0] = SpeechClip(first.samples[:-1], first.sample_rate, first.engine, first.voice)
    with pytest.raises(RuntimeError, match="length changed"):
        compose_audio(scene, speech_clips=clips)
