from copy import deepcopy

import pytest

from witness.score_v1_0_0 import score_reconstruction as old_score
from witness.score_v1_1_0 import reward_metrics, score_reconstruction
from witness.subnet.validator import apply_duplicate_sharing, apply_relative_gate


@pytest.mark.parametrize("quality,expected", [(0, 0), (.25, 0), (.399999999999, 0),
                                              (.4, .32), (.65, .52), (1, .8)])
def test_inclusive_fixed_boundary_without_partial_credit(quality, expected):
    metrics = reward_metrics(quality, .8)
    assert metrics["score"] == pytest.approx(expected)
    assert metrics["reward_factor"] == float(quality >= .4)
    assert reward_metrics(quality, .8, valid=False)["score"] == 0


def test_other_miners_and_tiers_cannot_raise_gate_and_duplicates_still_share():
    rows = [{"uid": uid, "scene_id": "scene", "tier": tier, "responded": valid,
             "quality": quality, "efficiency_factor": .8, "reconstruction": {"qa": {"q": answer}}}
            for uid, tier, quality, valid, answer in [(1, 1, .4, True, "same"),
                (2, 2, .4, True, "same"), (3, 3, 1, True, "perfect"), (4, 1, 1, False, "failure")]]
    apply_relative_gate(rows)
    apply_duplicate_sharing(rows)
    assert [r["score"] for r in rows] == pytest.approx([.16, .16, .8, 0])
    assert all(r["gate"]["threshold"] == .4 for r in rows)
    assert rows[0]["duplicate_count"] == 2
    old = deepcopy(rows)
    apply_relative_gate(old, score_version="1.0.0")
    assert old[0]["gate"]["threshold"] == .8
    assert old[0]["score_before_duplicates"] == 0


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_quality_components_and_efficiency_stay_identical(tier):
    scene = {"difficulty": tier, "fps": 24, "duration_frames": 240,
             "qa": [{"id": "q", "a": "red"}]}
    reconstruction, cost = {"qa": {"q": "red"}}, {"visual_tokens": 4408}
    old = old_score(scene, reconstruction, cost)
    new = score_reconstruction(scene, reconstruction, cost, q_min={1: 0, 2: 0, 3: 0})
    assert new["benchmark_version"] == "1.1.0"
    assert new["gate"]["q_min"] == .4
    for key in ("quality", "family_scores", "cost", "diagnostics"):
        assert old[key] == new[key]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 1.1])
def test_nonfinite_and_invalid_inputs_are_rejected(value):
    with pytest.raises(ValueError):
        reward_metrics(value, .8)
