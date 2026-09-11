"""Controls for the specific template, temporal-order and duplicate exploits."""
from copy import deepcopy
import json

import pytest

from witness.scene import build_scene as old_scene, state_at
from witness.score.oracle import perfect_reconstruction
from witness.score_v2_0_0 import score_reconstruction
from witness.subnet.protocol import WitnessTask
from witness.subnet.validator import (ValidatorConfig, apply_duplicate_sharing,
                                       apply_relative_gate, public_task_spec)
from witness.temporal import build_scene


def scene(seed=42, tier=1, **kwargs):
    result = build_scene(seed, tier, include_tts_timing=False, **kwargs)
    # Unit fixtures omit TTS execution. Media tests use the actual generated PCM.
    for line in result["dialogue"]:
        line["end"] = line["start"] + 3
        line["end_sample"] = round(line["end"] * result["audio"]["sample_rate"])
    return result


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_paired_histories_share_public_information_and_endpoints(tier):
    for seed in range(25):
        a, b = scene(seed, tier), scene(seed, tier, counterfactual=True)
        assert public_task_spec(a) == public_task_spec(b)
        assert state_at(a, 0) == state_at(b, 0)
        assert state_at(a, a["duration_frames"] - 1) == state_at(b, b["duration_frames"] - 1)
        for first, second in zip(a["qa"], b["qa"]):
            assert first["actor_sequence"] == second["actor_sequence"][::-1]
            assert first["actor_sequence"] != second["actor_sequence"]
        assert score_reconstruction(b, perfect_reconstruction(a))["score"] == 0


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_empty_old_template_and_orderless_controls_fail(tier):
    template = perfect_reconstruction(old_scene(0, tier, include_tts_timing=False))
    for seed in range(20):
        s = scene(seed, tier)
        oracle = perfect_reconstruction(s)
        assert score_reconstruction(s, oracle)["quality"] == 1
        assert score_reconstruction(s, {})["score"] == 0
        assert score_reconstruction(s, template)["score"] == 0
        unordered = deepcopy(oracle)
        unordered["qa"] = {q["id"]: " > ".join(sorted(q["a"].split(" > "))) for q in s["qa"]}
        assert score_reconstruction(s, unordered)["score"] == 0
        qa_only = {"qa": oracle["qa"]}
        assert score_reconstruction(s, qa_only)["score"] == 0


def test_duplicate_identity_tracks_scored_meaning_and_noise_does_not_split_it():
    s = scene()
    a = perfect_reconstruction(s)
    b = deepcopy(a)
    actors = {x["id"]: x["visual_description"] for x in s["actors"]}
    objects = {x["id"]: x["visual_description"] for x in s["objects"]}
    for item in b["events"]:
        item.pop("frame", None)
        item["t"] += 0.00000012
        item["note"] = "unscored metadata"
        if "actor" in item:
            item["actor"] = actors[item["actor"]]
        if "object" in item:
            item["object"] = objects[item["object"]]
    b["events"].reverse()
    for line in b["dialogue"]:
        line["speaker"] = "ignored speaker label"
    first, second = (score_reconstruction(s, x) for x in (a, b))
    assert first["quality"] == second["quality"] == 1
    assert first["semantic_fingerprint"] == second["semantic_fingerprint"]
    records = [{"uid": uid, "scene_id": "shared", "responded": True,
                "score_version": "2.0.0", "quality": r["quality"],
                "efficiency_factor": r["cost"]["efficiency_factor"],
                "semantic_fingerprint": r["semantic_fingerprint"], "reconstruction": x}
               for uid, r, x in [(1, first, a), (2, second, b)]]
    apply_relative_gate(records, score_version="2.0.0")
    apply_duplicate_sharing(records)
    assert [x["score"] for x in records] == [0.5, 0.5]
    assert [x["duplicate_count"] for x in records] == [2, 2]


def test_temporal_protocol_exposes_only_questions():
    s = scene()
    task = WitnessTask(task_id="task", tool_base_url="http://localhost:1234",
        session_id="session", scene_id="scene", seed_commitment="a" * 64,
        budget={"visual_tokens": 100000, "audio_seconds": 120, "transcript_chars": 20000},
        task_spec=public_task_spec(s), deadline_s=180)
    encoded = json.dumps(task.task_spec)
    assert "actor_sequence" not in encoded and "evidence_frames" not in encoded
    assert "seed" not in encoded
    assert all(q["a"] not in encoded for q in s["qa"])
    assert WitnessTask(**task.model_dump()).body_hash == task.body_hash


@pytest.mark.parametrize("payload", [None, [], {"events": [None] * 513},
                                       {"events": [{"t": float("nan")}]},
                                       {"unscored": "x" * 262145}])
def test_invalid_submissions_fail_without_aborting_the_round(payload):
    report = score_reconstruction(scene(), payload)
    assert report["quality"] == report["score"] == 0
    assert not report["gate"]["passed"]


def test_temporal_configuration_rejects_legacy_hints_and_old_scenes():
    with pytest.raises(ValueError, match="label-derived"):
        ValidatorConfig(score_version="2.0.0", allow_unlocked=True).validate()
    with pytest.raises(ValueError, match="programmatic_share"):
        ValidatorConfig(score_version="2.0.0", allow_unlocked=True, transcript_source="none").validate()
    with pytest.raises(ValueError, match="schema2.0"):
        score_reconstruction(old_scene(0, 1, include_tts_timing=False), {})


def test_seed_changes_story_and_does_not_just_reskin_a_template():
    signatures = set()
    for seed in range(30):
        s = scene(seed)
        normalized_actors = {x["id"]: i for i, x in enumerate(s["actors"])}
        signatures.add(tuple(tuple(normalized_actors[a] for a in q["actor_sequence"]) for q in s["qa"]))
    assert len(signatures) == 30


def test_pickup_and_drop_have_immediate_visible_state_changes():
    s = scene(7511, 3)
    for event in s["events"]:
        if event["action"] not in {"pick_up", "drop"}:
            continue
        before = state_at(s, event["frame"] - 1)["objects"][event["object"]]
        after = state_at(s, event["frame"])["objects"][event["object"]]
        assert abs(before["position"][1] - after["position"][1]) >= 9
        assert before["carrier"] != after["carrier"]


def test_local_round_uses_temporal_scorer_and_semantic_duplicates(tmp_path):
    import asyncio
    from witness.subnet.chain import InMemoryChainAdapter
    from witness.subnet.validator import WitnessValidator
    from witness.temporal import generate
    from witness.tools.client import WitnessClient

    path, _ = generate(2841, 1, tmp_path / "source")
    truth = json.loads(path.read_text())
    chain = InMemoryChainAdapter()

    async def respond(task, perturb):
        client = WitnessClient(task.tool_base_url, session_id=task.session_id)
        assert client.get_meta().data["duration"] == truth["duration"]
        task.reconstruction = perfect_reconstruction(truth)
        if perturb:
            for event in task.reconstruction["events"]:
                event.pop("frame", None)
                event["t"] += .0000001
        task.trace_summary = {"status": "ok"}
        return task

    chain.add_miner(1, lambda task: respond(task, False))
    chain.add_miner(2, lambda task: respond(task, True))
    config = ValidatorConfig(round_root=tmp_path / "rounds", source_scenes=(path.parent,),
        scene_count=1, score_version="2.0.0", allow_unlocked=True,
        transcript_source="none", tool_host="127.0.0.1", tool_port=0,
        burn_uid=None, set_weights_enabled=False)
    result = asyncio.run(WitnessValidator(chain, config).run_round())
    assert result["scoring_identity"]["version"] == "2.0.0"
    assert {"temporal_generator", "renderer", "fixed_reward", "observation_server"} <= set(
        result["scoring_identity"]["code_sha256"])
    for uid in ("1", "2"):
        row = result["miners"][uid]["scenes"][0]
        assert row["quality"] == 1 and row["score"] == .5
        assert row["duplicate_count"] == 2
        assert row["temporal"]["history_accuracy"] == 1
    assert chain.weight_history == []
