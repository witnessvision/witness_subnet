import json
from pathlib import Path

import pytest

from witness.score.oracle import perfect_reconstruction
from witness.score_v16 import _f_score, _qa_match, score_reconstruction


def test_every_additional_false_positive_reduces_precision():
    values = [_f_score(5, 5, count, family="events") for count in (5, 10, 100, 1000)]
    assert values[0] == 1
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[-1] < .01


def test_qa_rejects_answer_bags_but_normalizes_case_and_articles():
    assert _qa_match("yes no", "yes") == 0
    assert _qa_match("YES!", "yes") == 1
    assert _qa_match("The red mug.", "red mug") == 1
    assert _qa_match("1 2 3 4 5", "3") == 0


@pytest.mark.parametrize("name", ["scene_101", "scene_202", "scene_303"])
def test_perfect_reconstruction_keeps_full_quality(name):
    from witness.scene import build_scene
    seed = int(name.split("_")[-1])
    scene = build_scene(seed, {101: 1, 202: 2, 303: 3}[seed])
    report = score_reconstruction(scene, perfect_reconstruction(scene), {})
    assert report["quality"] == 1
    assert report["score"] == 1
    assert report["benchmark_version"] == "1.6-candidate"
