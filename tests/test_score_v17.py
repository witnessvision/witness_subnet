"""Constructed counterexamples; no held-out answers are used here."""
from copy import deepcopy
import json
import itertools
import random
from pathlib import Path

import pytest

from witness.score_v16 import score_reconstruction as previous_score
from witness.score_v17 import score_reconstruction, _maximum_credit
from witness.score.oracle import perfect_reconstruction


def example():
    scene = {
        "difficulty": 1, "fps": 24, "duration_frames": 240,
        "events": [{"frame": 48, "action": "walk"}],
        "shots": [
            {"id": "private-a", "start_frame": 0, "end_frame": 120},
            {"id": "private-b", "start_frame": 120, "end_frame": 240},
        ],
    }
    return scene, {"events": deepcopy(scene["events"]), "shots": deepcopy(scene["shots"])}


def test_event_duplicates_count_as_false_positives():
    scene, prediction = example()
    prediction["events"] *= 100
    assert previous_score(scene, prediction)["family_scores"]["events"] == 1
    result = score_reconstruction(scene, prediction)
    assert result["family_scores"]["events"] < .02


def test_distinct_nearby_true_events_are_not_collapsed():
    scene, prediction = example()
    scene["events"].append({"frame": 72, "action": "walk"})
    prediction["events"] = deepcopy(scene["events"])
    assert previous_score(scene, prediction)["family_scores"]["events"] < 1
    assert score_reconstruction(scene, prediction)["quality"] == 1


def test_shot_identity_is_temporal_and_duplicates_are_penalized():
    scene, prediction = example()
    prediction["shots"][1]["id"] = "miner-local-name"
    assert previous_score(scene, prediction)["family_scores"]["shots"] == 0
    assert score_reconstruction(scene, prediction)["quality"] == 1
    prediction["shots"].extend([deepcopy(prediction["shots"][1])] * 99)
    assert score_reconstruction(scene, prediction)["family_scores"]["shots"] < .02


def test_duplicate_initial_shots_are_not_free_predictions():
    scene, prediction = example()
    prediction["shots"].extend([deepcopy(prediction["shots"][0])] * 99)
    assert score_reconstruction(scene, prediction)["family_scores"]["shots"] < .02


def test_public_text_and_error_contract_requires_no_private_metadata():
    scene, prediction = example()
    scene["on_screen_text"] = [{"start_frame": 24, "end_frame": 72, "text": "The door is open", "role": "subtitle", "id": "private-text-42"}]
    scene["intentional_errors"] = [{"frame": 24, "type": "misleading_subtitle", "text_id": "private-text-42"}]
    prediction["on_screen_text"] = [{"start": 1, "end": 3, "text": "The door is open"}]
    prediction["intentional_errors"] = [{"t": 1, "type": "misleading_subtitle"}]
    assert previous_score(scene, prediction)["quality"] < 1
    assert score_reconstruction(scene, prediction)["quality"] == 1
    prediction["on_screen_text"][0]["text"] = "The window is closed"
    prediction["intentional_errors"][0]["type"] = "tint_change"
    result = score_reconstruction(scene, prediction)
    assert result["family_scores"]["on_screen_text"] == 0
    assert result["family_scores"]["intentional_errors"] == 0


def test_dialogue_matching_optimizes_lexical_credit_within_valid_intervals():
    scene, prediction = example()
    scene["dialogue"] = [
        {"speaker": "a", "start": 0, "end": 2, "text": "red"},
        {"speaker": "b", "start": .4, "end": 2.4, "text": "blue"},
    ]
    prediction["dialogue"] = deepcopy(scene["dialogue"])
    prediction["dialogue"][0]["text"] = "blue"
    prediction["dialogue"][1]["text"] = "red"
    assert previous_score(scene, prediction)["family_scores"]["dialogue"] == 0
    assert score_reconstruction(scene, prediction)["family_scores"]["dialogue"] == 1
    for item in prediction["dialogue"]:
        item["start"] += 5
        item["end"] += 5
    assert score_reconstruction(scene, prediction)["family_scores"]["dialogue"] == 0


def test_maximum_credit_agrees_with_exhaustive_small_assignments():
    rng = random.Random(42)
    assert _maximum_credit([]) == _maximum_credit([[]]) == 0
    for n in range(1, 4):
        for m in range(1, 4):
            for _ in range(5):
                weights = [[rng.randrange(11) / 10 for _ in range(m)] for _ in range(n)]
                exact = max(sum(weights[i][j] if j < m else 0 for i, j in enumerate(assignment))
                            for assignment in itertools.permutations(range(m + n), n))
                assert _maximum_credit(weights) == pytest.approx(exact)


def test_dialogue_duplicates_cannot_reuse_the_same_truth_line():
    scene, prediction = example()
    scene["dialogue"] = [{"speaker": "a", "start": 0, "end": 2, "text": "red"}]
    prediction["dialogue"] = deepcopy(scene["dialogue"]) * 100
    assert score_reconstruction(scene, prediction)["family_scores"]["dialogue"] < .02


@pytest.mark.parametrize("name", ["scene_101", "scene_202", "scene_303"])
def test_development_oracle_keeps_full_credit(name):
    path = Path(__file__).parents[1] / "data/scenes/synthetic" / name / "scene.json"
    from witness.scene import build_scene
    seed = int(name.split("_")[-1])
    scene = build_scene(seed, {101: 1, 202: 2, 303: 3}[seed])
    result = score_reconstruction(scene, perfect_reconstruction(scene))
    assert result["quality"] == 1
    assert result["benchmark_version"] == "1.7-candidate"


@pytest.mark.parametrize("item", [{"frame": 240}, {"frame": 10000}, {"t": True}, {"t": -.001}])
def test_point_items_must_be_inside_the_video(item):
    scene, prediction = example()
    prediction["events"] = [{**item, "action": "walk"}]
    result = score_reconstruction(scene, prediction)
    assert result["family_scores"]["events"] == 0
    assert any(row["family"] == "events" for row in result["diagnostics"])


@pytest.mark.parametrize("family, extra", [("shots", {}), ("on_screen_text", {"text": "hello"}), ("dialogue", {"speaker": "unknown", "text": "hello"})])
def test_interval_items_cannot_extend_beyond_video(family, extra):
    scene, prediction = example()
    prediction[family] = [{"start_frame": 0, "end_frame": 241, **extra}]
    result = score_reconstruction(scene, prediction)
    assert result["family_scores"][family] == 0
    assert any(row["family"] == family for row in result["diagnostics"])
