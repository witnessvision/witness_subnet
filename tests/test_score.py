from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

from witness.score.degrade import degrade_reconstruction
from witness.score.oracle import perfect_reconstruction
from witness.score.scorer import FALSE_POSITIVE_CAPS, score_reconstruction
from witness.tools.metering import visual_token_cost

ROOT = Path(__file__).parents[1]
SCENES = [ROOT / "data/scenes/synthetic" / name / "scene.json" for name in ("scene_101", "scene_202", "scene_303")]
EMPTY = {
    "events": [],
    "dialogue": [],
    "shots": [],
    "on_screen_text": [],
    "audio_events": [],
    "intentional_errors": [],
    "qa": {},
}


def load_scene(path: Path) -> dict:
    from witness.scene import build_scene
    seed = int(path.parent.name.split("_")[-1])
    return build_scene(seed, {101: 1, 202: 2, 303: 3}[seed])


@pytest.mark.parametrize("path", SCENES)
def test_oracle_quality_is_one(path: Path) -> None:
    scene = load_scene(path)
    report = score_reconstruction(scene, perfect_reconstruction(scene), {})
    assert report["quality"] == 1.0
    assert report["score"] == 1.0
    assert set(report["family_scores"].values()) == {1.0}


@pytest.mark.parametrize("path", SCENES)
def test_empty_reconstruction_scores_zero(path: Path) -> None:
    scene = load_scene(path)
    report = score_reconstruction(scene, EMPTY, {})
    assert report["score"] == 0.0
    assert report["gate"]["passed"] is False


def test_degradation_is_monotonic() -> None:
    scene = load_scene(SCENES[2])
    oracle = perfect_reconstruction(scene)
    mild = degrade_reconstruction(oracle, drop_events=1, shift_frames=3, wrong_answers=1)
    severe = degrade_reconstruction(oracle, drop_events=4, shift_frames=30, wrong_answers=3)
    qualities = [score_reconstruction(scene, item, {})["quality"] for item in (oracle, mild, severe)]
    assert qualities[0] > qualities[1] > qualities[2]


def test_text_and_dialogue_intervals_accept_iou_when_endpoints_exceed_tolerance() -> None:
    scene = load_scene(SCENES[1])
    reconstruction = perfect_reconstruction(scene)
    shift_frames = 12  # tier-2 endpoint tolerance is nine frames
    text = reconstruction["on_screen_text"][0]
    text["start_frame"] += shift_frames
    text["end_frame"] += shift_frames
    if "start" in text:
        text["start"] += shift_frames / scene["fps"]
        text["end"] += shift_frames / scene["fps"]
    dialogue = reconstruction["dialogue"][0]
    dialogue["start"] += shift_frames / scene["fps"]
    dialogue["end"] += shift_frames / scene["fps"]
    report = score_reconstruction(scene, reconstruction, {})
    assert report["family_scores"]["on_screen_text"] == 1.0
    assert report["family_scores"]["dialogue"] == 1.0


def test_false_positive_penalty_is_capped_within_each_family() -> None:
    scene = load_scene(SCENES[1])
    capped = perfect_reconstruction(scene)
    spam = deepcopy(capped["on_screen_text"][0])
    spam["text"] = "unmatched spam"
    capped["on_screen_text"].extend(
        deepcopy(spam) for _ in range(FALSE_POSITIVE_CAPS["on_screen_text"])
    )
    flooded = deepcopy(capped)
    flooded["on_screen_text"].extend(deepcopy(spam) for _ in range(100))
    capped_report = score_reconstruction(scene, capped, {})
    flooded_report = score_reconstruction(scene, flooded, {})
    assert flooded_report["family_scores"]["on_screen_text"] > 0.0
    assert flooded_report["family_scores"]["on_screen_text"] == capped_report["family_scores"]["on_screen_text"]
    assert flooded_report["false_positive_caps"] == FALSE_POSITIVE_CAPS


def test_scoring_is_deterministic() -> None:
    scene = load_scene(SCENES[1])
    reconstruction = degrade_reconstruction(perfect_reconstruction(scene), drop_events=2, shift_frames=4, wrong_answers=1)
    cost = {"visual_tokens": 9000, "audio_seconds": 4.25, "transcript_chars": 81}
    first = score_reconstruction(scene, reconstruction, cost)
    second = score_reconstruction(scene, reconstruction, cost)
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(second, sort_keys=True, separators=(",", ":"))


def test_full_read_cost_applies_lambda() -> None:
    scene = load_scene(SCENES[1])
    oracle = perfect_reconstruction(scene)
    reference = 33 * visual_token_cost(640, 360)
    report = score_reconstruction(scene, oracle, {"visual_tokens": reference})
    assert report["cost"]["ratio"] == 1.0
    assert report["score"] == pytest.approx(0.7)


def test_tier_tolerances_and_shot_boundary_window() -> None:
    tier_one = load_scene(SCENES[0])
    oracle_one = perfect_reconstruction(tier_one)
    at_limit = degrade_reconstruction(oracle_one, shift_frames=12)
    outside_limit = degrade_reconstruction(oracle_one, shift_frames=13)
    assert score_reconstruction(tier_one, at_limit, {})["family_scores"]["events"] == 1.0
    assert score_reconstruction(tier_one, outside_limit, {})["family_scores"]["events"] == 0.0

    tier_two = load_scene(SCENES[1])
    oracle_two = perfect_reconstruction(tier_two)
    within_two_frames = degrade_reconstruction(oracle_two, shift_frames=2)
    outside_two_frames = degrade_reconstruction(oracle_two, shift_frames=3)
    assert score_reconstruction(tier_two, within_two_frames, {})["family_scores"]["shots"] == 1.0
    assert score_reconstruction(tier_two, outside_two_frames, {})["family_scores"]["shots"] == 0.0


def test_errors_are_precision_heavy_and_qa_is_normalized() -> None:
    scene = load_scene(SCENES[1])
    reconstruction = perfect_reconstruction(scene)
    reconstruction["intentional_errors"].append(
        {"frame": 700, "type": "continuity", "object": "invented"}
    )
    expected_answer = next(item["a"] for item in scene["qa"] if item["id"] == "temporal_1")
    reconstruction["qa"]["temporal_1"] = f"  THE {expected_answer}!!! "
    report = score_reconstruction(scene, reconstruction, {})
    assert report["family_scores"]["intentional_errors"] == pytest.approx(5 / 9)
    assert report["family_scores"]["qa"] == 1.0


def _rewrite_entities_with_descriptions(scene: dict, reconstruction: dict) -> dict:
    rewritten = deepcopy(reconstruction)
    actor_descriptions = {entity["id"]: entity["visual_description"] for entity in scene["actors"]}
    object_descriptions = {entity["id"]: entity["visual_description"] for entity in scene["objects"]}
    for family in ("events", "intentional_errors"):
        for item in rewritten[family]:
            if item.get("actor") in actor_descriptions:
                item["actor"] = actor_descriptions[item["actor"]]
            if item.get("object") in object_descriptions:
                item["object"] = object_descriptions[item["object"]]
    return rewritten


@pytest.mark.parametrize("path", SCENES)
def test_visual_descriptions_resolve_like_internal_ids(path: Path) -> None:
    scene = load_scene(path)
    reconstruction = _rewrite_entities_with_descriptions(scene, perfect_reconstruction(scene))
    report = score_reconstruction(scene, reconstruction, {})
    assert report["quality"] == 1.0
    assert report["family_scores"]["events"] == 1.0
    assert report["family_scores"]["intentional_errors"] == 1.0


def test_ambiguous_partial_entity_does_not_resolve() -> None:
    scene = load_scene(SCENES[1])
    assert len({entity["kind"] for entity in scene["objects"]}) == 1
    reconstruction = perfect_reconstruction(scene)
    ambiguous_kind = scene["objects"][0]["kind"]
    for event in reconstruction["events"]:
        if event.get("object") is not None:
            event["object"] = ambiguous_kind
    for error in reconstruction["intentional_errors"]:
        if error.get("object") is not None:
            error["object"] = ambiguous_kind
    report = score_reconstruction(scene, reconstruction, {})
    assert report["family_scores"]["events"] < 1.0
    assert report["family_scores"]["intentional_errors"] == 0.0


def test_unambiguous_color_and_shape_partials_resolve() -> None:
    scene = load_scene(SCENES[1])
    reconstruction = perfect_reconstruction(scene)
    actors = {entity["id"]: entity for entity in scene["actors"]}
    objects = {entity["id"]: entity for entity in scene["objects"]}
    for event in reconstruction["events"]:
        if event.get("actor") is not None:
            event["actor"] = actors[event["actor"]]["shape"]
        if event.get("object") is not None:
            event["object"] = objects[event["object"]]["color"]
    report = score_reconstruction(scene, reconstruction, {})
    assert report["family_scores"]["events"] == 1.0


def test_malformed_item_is_false_positive_with_diagnostic() -> None:
    scene = load_scene(SCENES[1])
    reconstruction = perfect_reconstruction(scene)
    reconstruction["events"].append({"action": "enter", "actor": "the orange circle"})
    report = score_reconstruction(scene, reconstruction, {})
    assert report["family_scores"]["events"] < 1.0
    assert {entry["family"] for entry in report["diagnostics"]} == {"events"}
    assert "frame/t" in report["diagnostics"][0]["reason"]


@pytest.mark.parametrize("reconstruction", [None, [], "junk", {"qa": []}, {"qa": "junk"}])
def test_invalid_root_or_qa_returns_zero_report(reconstruction: object) -> None:
    report = score_reconstruction(load_scene(SCENES[1]), reconstruction, {})  # type: ignore[arg-type]
    assert report["quality"] == report["score"] == 0.0
    assert report["invalid_reason"]
    assert report["diagnostics"]


def test_random_junk_never_raises_and_scores_stay_bounded() -> None:
    rng = random.Random(90210)
    atoms = [None, True, False, "junk", -1, 10**1000, float("nan"), float("inf")]

    def junk(depth: int = 0) -> object:
        if depth >= 2 or rng.random() < 0.45:
            return rng.choice(atoms)
        if rng.random() < 0.5:
            return [junk(depth + 1) for _ in range(rng.randrange(4))]
        return {str(rng.randrange(6)): junk(depth + 1) for _ in range(rng.randrange(4))}

    scene = load_scene(SCENES[2])
    families = ("events", "dialogue", "shots", "on_screen_text", "audio_events", "intentional_errors", "qa")
    cases: list[object] = list(atoms)
    cases.extend(
        {
            family: [junk(), {"frame": junk(), "t": junk()}, "not-an-item"],
            "qa": {} if family != "qa" else junk(),
        }
        for family in families
        for _ in range(25)
    )
    for reconstruction in cases:
        report = score_reconstruction(scene, reconstruction, {})  # type: ignore[arg-type]
        assert 0.0 <= report["quality"] <= 1.0
        assert 0.0 <= report["score"] <= 1.0
        assert all(0.0 <= value <= 1.0 for value in report["family_scores"].values())
        assert isinstance(report["diagnostics"], list)


def test_dialogue_partial_credit_by_wer_and_interval_only() -> None:
    from witness.score.scorer import _dialogue_score

    truth = [{"start": 10.0, "end": 16.0, "speaker": "A", "text": "I counted 67 people waiting outside the cafe this morning."}]
    degraded = [{"start": 11.0, "end": 16.0, "speaker": "B", "text": "I counted 67 people <unk> <unk> the cafe this morning."}]
    exact = [{"start": 10.0, "end": 16.0, "text": "I counted 67 people waiting outside the cafe this morning."}]
    wrong_time = [{"start": 40.0, "end": 46.0, "text": "I counted 67 people waiting outside the cafe this morning."}]
    partial = _dialogue_score(truth, degraded, 24, 2, 1)
    assert 0.6 < partial < 1.0
    assert _dialogue_score(truth, exact, 24, 2, 1) == 1.0
    assert _dialogue_score(truth, wrong_time, 24, 2, 1) == 0.0
