from copy import deepcopy
import json

import pytest

from witness.reward import REWARD_VERSION, reward_metrics, summarize_records
from witness.reward_report import compare_rounds
from witness.score_v18 import score_reconstruction as old_score
from witness.score_v19 import score_reconstruction
from witness.score_v1_0_0 import SCORER_VERSION, score_reconstruction as production_score
from witness.subnet.validator import apply_relative_gate, apply_duplicate_sharing


@pytest.mark.parametrize("threshold", [.60, .65, .70, .80])
def test_narrow_band_is_continuous_monotonic_and_preserves_full_credit(threshold):
    floor = threshold - .15
    assert reward_metrics(floor, .85, threshold)["score"] == 0
    middle = reward_metrics(threshold - .075, .85, threshold)
    assert middle["reward_factor"] == pytest.approx(.5)
    assert not middle["full_reward_eligible"]
    assert reward_metrics(threshold, .85, threshold)["score"] == pytest.approx(threshold * .85)
    values = [reward_metrics(i / 1000, .85, threshold)["score"] for i in range(1001)]
    assert values == sorted(values)
    assert max(b - a for a, b in zip(values, values[1:])) < .007
    assert reward_metrics(.95, .85, threshold)["score"] == pytest.approx(.95 * .85)


@pytest.mark.parametrize("quality", [0, .34, .44, .59, .9, 1.])
def test_failures_and_exhausted_efficiency_receive_zero(quality):
    assert reward_metrics(quality, .9, .7, valid=False)["score"] == 0
    assert reward_metrics(quality, 0, .7)["score"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_reward_numbers_are_rejected(value):
    with pytest.raises(ValueError):
        reward_metrics(value, .9, .7)


@pytest.mark.parametrize("threshold", [0., .2, .35])
def test_custom_threshold_without_partial_band_is_well_defined(threshold):
    assert reward_metrics(threshold, .8, threshold)["score"] == pytest.approx(threshold * .8)
    assert reward_metrics(max(0., threshold - .01), .8, threshold)["score"] == 0


def row(uid, quality, responded=True):
    return {"uid": uid, "scene_id": "same", "tier": 1, "responded": responded,
            "quality": quality, "efficiency_factor": .8,
            "family_scores": {"qa": quality}, "reconstruction": {"qa": {"q": str(uid)}}}


def test_relative_competition_and_duplicates_still_reduce_partial_credit():
    rows = [row(1, 1.), row(2, .72), row(3, .72), row(4, .99, False)]
    rows[2]["reconstruction"] = deepcopy(rows[1]["reconstruction"])
    apply_relative_gate(rows, score_version=REWARD_VERSION)
    assert rows[1]["gate"]["threshold"] == .8
    assert rows[1]["gate"]["partial_floor"] == .65
    assert not rows[1]["gate"]["passed"]
    partial = rows[1]["score_before_duplicates"]
    assert 0 < partial < .72 * .8
    apply_duplicate_sharing(rows)
    assert rows[1]["score"] == pytest.approx(partial / 2)
    assert rows[1]["metrics"]["score"] == rows[1]["score"]
    assert rows[3]["score"] == 0


def test_historical_gate_stays_hard_and_failed_rows_remain_in_denominator():
    rows = [row(1, .63), row(2, .9, False)]
    apply_relative_gate(rows, score_version="1.5")
    apply_duplicate_sharing(rows)
    assert all(r["score"] == 0 for r in rows)
    metrics = summarize_records(rows)
    assert metrics["quality"] == .315
    assert metrics["continuous_score"] == pytest.approx(.63 * .8 / 2)
    assert metrics["valid_rate"] == .5
    assert metrics["full_reward_rate"] == 0


def test_candidate_preserves_quality_and_cost_exactly():
    scene = {"difficulty": 1, "fps": 24, "duration_frames": 240,
             "qa": [{"id": "q", "a": "red"}]}
    old = old_score(scene, {"qa": {"q": "wrong"}}, {"visual_tokens": 100})
    new = score_reconstruction(scene, {"qa": {"q": "wrong"}}, {"visual_tokens": 100})
    for key in ("quality", "family_scores", "cost", "diagnostics"):
        assert old[key] == new[key]
    assert new["benchmark_version"] == REWARD_VERSION
    assert old["benchmark_version"] == "1.8"


def test_saved_round_rescoring_is_paired_and_does_not_mutate_input(tmp_path):
    path = tmp_path / "round.json"
    artifact = {"scoring_identity": {"version": "1.8"},
                "miners": {"1": {"scenes": [row(1, .63)]}}}
    path.write_text(json.dumps(artifact))
    before = path.read_bytes()
    result = compare_rounds([path])["miners"]["1"]
    assert result["1.8"]["reward"] == 0
    assert result[SCORER_VERSION]["reward"] > 0
    assert result["1.8"]["quality"] == result[SCORER_VERSION]["quality"]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="duplicate"):
        compare_rounds([path, path])
    different = tmp_path / "different.json"
    artifact["scoring_identity"]["transcript_source"] = "none"
    different.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="different scoring"):
        compare_rounds([path, different])
    artifact["scoring_identity"]["version"] = "1.7-candidate"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="v1.8"):
        compare_rounds([path])


@pytest.mark.parametrize("tier", [1, 2, 3])
@pytest.mark.parametrize("visual_tokens", [0, 4408, 100_000])
def test_production_release_preserves_candidate_report(tier, visual_tokens):
    scene = {"difficulty": tier, "fps": 24, "duration_frames": 240,
             "qa": [{"id": "q", "a": "red"}]}
    reconstruction = {"qa": {"q": "red"}}
    cost = {"visual_tokens": visual_tokens, "audio_seconds": 3.5, "transcript_chars": 47}
    candidate = score_reconstruction(scene, reconstruction, cost)
    production = production_score(scene, reconstruction, cost)
    assert candidate.pop("benchmark_version") == "1.9-candidate"
    assert production.pop("benchmark_version") == "1.0.0"
    assert production == candidate


def test_production_gate_matches_candidate_at_every_boundary_and_duplicate():
    rows = [row(0, 1.)]
    for uid, quality in enumerate([.35, .55, .6, .65, .7, .72, .8, .95], 1):
        rows.append(row(uid, quality))
    rows.extend([row(9, .99, False), row(10, .72)])
    rows[-1]["reconstruction"] = deepcopy(rows[6]["reconstruction"])
    candidate, production = deepcopy(rows), deepcopy(rows)
    apply_relative_gate(candidate, score_version="1.9-candidate")
    apply_relative_gate(production, score_version="1.0.0")
    apply_duplicate_sharing(candidate)
    apply_duplicate_sharing(production)
    assert production == candidate
    assert production[6]["duplicate_count"] == 2
    assert production[6]["score"] == pytest.approx(.72 * .8 * (.72 - .65) / .15 / 2)
