from __future__ import annotations

import asyncio
import json
import subprocess
import os
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from witness import freeze
from witness.subnet.chain import InMemoryChainAdapter
from witness.subnet.miner import WitnessMiner
from witness.subnet.protocol import WitnessTask
from witness.subnet.validator import (
    ValidatorConfig,
    WitnessValidator,
    build_weight_vector,
    reconstruction_hash,
    apply_relative_gate,
    load_and_verify_benchmark_lock,
)


def test_relative_gate_cannot_weaken_frozen_absolute_minimum():
    rows = [{"scene_id": "dev", "tier": 1, "responded": True,
             "quality": .63, "efficiency_factor": .9}]
    apply_relative_gate(rows, score_version="1.5")
    assert rows[0]["gate"]["threshold"] == .7
    assert not rows[0]["gate"]["passed"]
    assert rows[0]["score_before_duplicates"] == 0


def test_missing_explicit_lock_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="lock missing"):
        load_and_verify_benchmark_lock(tmp_path / "missing.json", tmp_path, allow_unlocked=False)


def test_default_lock_includes_dev_verification(monkeypatch, tmp_path):
    from witness.subnet import validator as module
    lock = tmp_path / "lock.json"
    lock.write_text('{}')
    monkeypatch.setattr(module, "DEFAULT_BENCHMARK_LOCK", lock)
    seen = []
    monkeypatch.setattr(module.freeze, "verify_lock", lambda value, roots: seen.extend(roots) or [])
    module.load_and_verify_benchmark_lock(lock, tmp_path / "hidden", allow_unlocked=False)
    assert seen == [tmp_path / "hidden", module.REPOSITORY_ROOT / "data/scenes/dev-v15"]






FFMPEG = Path(
    os.environ.get("WITNESS_FFMPEG")
    or (str(Path.home() / "bin" / "ffmpeg") if (Path.home() / "bin" / "ffmpeg").is_file() else "")
    or shutil.which("ffmpeg")
    or "/usr/bin/ffmpeg"
)


def _task() -> WitnessTask:
    return WitnessTask(
        task_id="round:scene:7",
        tool_base_url="http://127.0.0.1:8765",
        session_id="opaque-session",
        scene_id="opaque-scene",
        seed_commitment="a" * 64,
        budget={
            "visual_tokens": 10_000,
            "audio_seconds": 10.0,
            "transcript_chars": 1_000,
        },
        task_spec={
            "duration": 2.0,
            "fps": 24,
            "tier": 1,
            "schema_version": "1.2",
            "qa": [{"id": "q1", "q": "Which object was picked up?"}],
        },
        deadline_s=30,
        reconstruction={"qa": {"q1": "the red mug"}},
        trace_summary={"status": "ok"},
    )


def test_protocol_serialization_round_trip_and_public_task_boundary() -> None:
    original = _task()
    restored = WitnessTask.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.deserialize() == {
        "reconstruction": {"qa": {"q1": "the red mug"}},
        "trace_summary": {"status": "ok"},
    }
    leaked = original.model_dump()
    leaked["task_spec"]["qa"][0]["a"] = "private answer"
    with pytest.raises(ValidationError, match="only id and q"):
        WitnessTask.model_validate(leaked)


def test_weight_vector_zeros_nonresponders_and_burns_an_empty_round() -> None:
    weighted = build_weight_vector(
        [0, 1, 2],
        {0: 0.4, 1: 0.9, 2: 0.6},
        {0, 2},
        burn_uid=1,
        burn_rate=0.1,
    )
    assert weighted == pytest.approx([0.36, 0.1, 0.54])
    assert build_weight_vector(
        [0, 1, 2],
        {0: 0.0, 1: 0.0, 2: 0.0},
        set(),
        burn_uid=1,
        burn_rate=0.0,
    ) == [0.0, 1.0, 0.0]


def test_duplicate_hash_ignores_unscored_nonce_fields() -> None:
    reconstruction = {"events": [{"frame": 1, "action": "cut"}], "qa": {}}
    assert reconstruction_hash(reconstruction) == reconstruction_hash(
        {**reconstruction, "unscored_nonce": "evade-duplicate-check"}
    )


def _tiny_scene(directory: Path, seed: int) -> Path:
    directory.mkdir(parents=True)
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=22050",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(directory / "video.mp4"),
        ],
        check=True,
    )
    scene = {
        "schema_version": "1.2",
        "seed": seed,
        "duration": 2.0,
        "fps": 24,
        "duration_frames": 48,
        "difficulty": 1,
        "actors": [
            {
                "id": "A",
                "visual_description": "the blue circle",
                "color": "blue",
                "shape": "circle",
            }
        ],
        "objects": [
            {
                "id": "obj1",
                "visual_description": "the red mug",
                "color": "red",
                "kind": "mug",
            }
        ],
        "events": [
            {"frame": 12, "t": 0.5, "actor": "A", "action": "pick_up", "object": "obj1"}
        ],
        "dialogue": [{"speaker": "A", "start": 0.2, "end": 0.6, "text": "Take the mug."}],
        "shots": [{"start_frame": 0, "end_frame": 48, "start": 0.0, "end": 2.0}],
        "on_screen_text": [
            {"start_frame": 19, "end_frame": 29, "start": 0.8, "end": 1.2, "role": "sign", "text": "GO"}
        ],
        "audio_events": [{"frame": 36, "t": 1.5, "kind": "door_slam"}],
        "intentional_errors": [
            {"frame": 40, "t": 1.666667, "type": "continuity", "object": "obj1"}
        ],
        "qa": [
            {"id": "q1", "q": "Which object was picked up?", "a": "the red mug"}
        ],
        "audio": {"sample_rate": 22050, "channels": 1},
    }
    (directory / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    return directory


def test_full_dry_run_round_relative_gate_duplicates_ema_and_weights(tmp_path: Path) -> None:
    source_a = _tiny_scene(tmp_path / "source_a", 101)
    source_b = _tiny_scene(tmp_path / "source_b", 202)
    lookup: dict[str, Path] = {}
    chain = InMemoryChainAdapter()
    # Validator-only oracle fixture exercises scoring; never used by the base miner.
    async def oracle(task):
        from witness.score.oracle import perfect_reconstruction
        scene = json.loads((lookup[task.scene_id] / "scene.json").read_text())
        task.reconstruction = perfect_reconstruction(scene)
        task.trace_summary = {"status": "ok"}
        return task

    async def garbage(task: WitnessTask) -> WitnessTask:
        task.reconstruction = {"garbage": True}
        task.trace_summary = {"status": "garbage"}
        return task

    chain.add_miner(0, oracle)
    chain.add_miner(1, oracle)
    chain.add_miner(2, garbage)
    round_root = tmp_path / "rounds"
    round_root.mkdir()
    (round_root / "ema.json").write_text(
        json.dumps({"scores": {"0": 0.2, "1": 0.4, "2": 0.0}}),
        encoding="utf-8",
    )
    validator = WitnessValidator(
        chain,
        ValidatorConfig(
            score_version="1.5",
            round_root=round_root,
            scene_count=2,
            source_scenes=(source_a, source_b),
            benchmark_lock=None,
            programmatic_share=1.0,
            tiers=(1,),
            deadline_s=45,
            ema_alpha=0.5,
            tool_host="127.0.0.1",
            tool_port=0,
            burn_uid=0,
            burn_rate=0.0,
        ),
        dry_scene_lookup=lookup,
    )

    commitments_seen = {}
    original_query = chain.query

    async def capture_query(endpoint, task, timeout):
        committed = list(round_root.glob("round_*/commitments.json"))
        assert len(committed) == 1  # materialized before the first response
        on_disk = json.loads(committed[0].read_text())["scenes"]
        assert on_disk[task.scene_id] == task.seed_commitment
        assert "seed" not in task.model_dump()
        assert "seed_commitment_nonce_revealed" not in task.model_dump()
        commitments_seen[task.scene_id] = task.seed_commitment
        return await original_query(endpoint, task, timeout=timeout)

    chain.query = capture_query
    artifact = asyncio.run(validator.run_round())
    from witness.subnet.validator import seed_commitment
    nonce = artifact["seed_commitment_nonce_revealed"]
    assert {row["seed"] for row in artifact["scene_seeds_revealed"]} == {101, 202}
    for row in artifact["scene_seeds_revealed"]:
        assert commitments_seen[row["scene_id"]] == seed_commitment(row["scene_id"], row["seed"], nonce)
    private = round_root / f"round_{artifact['round_id']}" / "private"
    assert private.stat().st_mode & 0o777 == 0o700
    assert (private / "seed-commitment.json").stat().st_mode & 0o777 == 0o600

    assert len(artifact["scene_seeds_revealed"]) == 2
    good_records = artifact["miners"]["0"]["scenes"] + artifact["miners"]["1"]["scenes"]
    assert all(record["quality"] == 1.0 for record in good_records)
    assert all(record["gate"]["threshold"] == 0.8 for record in good_records)
    assert all(record["gate"]["passed"] for record in good_records)
    assert all(record["duplicate_count"] == 2 for record in good_records)

    garbage_records = artifact["miners"]["2"]["scenes"]
    assert all(not record["gate"]["passed"] and record["score"] == 0 for record in garbage_records)

    for uid, previous in (("0", 0.2), ("1", 0.4), ("2", 0.0)):
        row = artifact["miners"][uid]
        assert row["ema_score"] == pytest.approx(0.5 * row["round_score"] + 0.5 * previous)

    assert artifact["weights"]["uids"] == [0, 1, 2]
    assert len(artifact["weights"]["values"]) == 3
    assert sum(artifact["weights"]["values"]) == pytest.approx(1.0)
    assert artifact["weights"]["values"][2] == 0.0
    assert chain.weight_history[-1]["uids"] == [0, 1, 2]
    assert (round_root / f"round_{artifact['round_id']}" / "round.json").is_file()



def test_validator_verifies_and_draws_from_temporary_lock(tmp_path: Path) -> None:
    hidden_root = tmp_path / "hidden"
    source = _tiny_scene(hidden_root / "locked_scene", 909)
    lock_path = tmp_path / "benchmark_v1.5.lock.json"
    lock_path.write_text(
        json.dumps(freeze.build_lock("1.5", [hidden_root])), encoding="utf-8"
    )
    config = ValidatorConfig(
        round_root=tmp_path / "rounds",
        scene_count=1,
        programmatic_share=0.0,
        tiers=(1,),
        benchmark_lock=lock_path,
        locked_scene_root=hidden_root,
        tool_host="127.0.0.1",
        tool_port=0,
    )
    validator = WitnessValidator(InMemoryChainAdapter(), config)
    scenes = validator._prepare_scenes(tmp_path / "prepared", "0xlocked")

    assert len(scenes) == 1
    assert scenes[0].kind == "locked_hidden"
    assert scenes[0].truth["seed"] == 909
    assert scenes[0].seed == 909
    assert scenes[0].directory.name != source.name

    truth_path = source / "scene.json"
    changed = json.loads(truth_path.read_text(encoding="utf-8"))
    changed["difficulty"] = 2
    truth_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock verification failed.*scene.json changed"):
        WitnessValidator(InMemoryChainAdapter(), config)

    unlocked = ValidatorConfig(
        round_root=tmp_path / "unlocked-rounds",
        scene_count=1,
        programmatic_share=0.0,
        tiers=(1,),
        benchmark_lock=lock_path,
        locked_scene_root=hidden_root,
        allow_unlocked=True,
        tool_host="127.0.0.1",
        tool_port=0,
    )
    assert WitnessValidator(InMemoryChainAdapter(), unlocked).benchmark_lock is not None


def test_same_public_chain_inputs_do_not_reproduce_private_scenes(tmp_path, monkeypatch):
    from witness.subnet import validator as module
    def fake_generate(seed, tier, destination):
        destination.mkdir(parents=True)
        (destination / "scene.json").write_text(json.dumps({
            "seed": seed, "difficulty": tier, "duration": 1, "fps": 24, "qa": [],
        }))
    monkeypatch.setattr(module, "generate", fake_generate)
    validator = WitnessValidator(InMemoryChainAdapter(), ValidatorConfig(
        scene_count=1, programmatic_share=1, tiers=(1,), benchmark_lock=None))
    first = validator._prepare_scenes(tmp_path / "first", "same-public-block")[0]
    second = validator._prepare_scenes(tmp_path / "second", "same-public-block")[0]
    assert first.seed != second.seed
    assert first.scene_id != second.scene_id
    assert first.truth["seed"] == first.seed
    assert second.truth["seed"] == second.seed


def test_seed_commitment_hides_known_fixture_seed_and_binds_reveal():
    from witness.subnet.validator import seed_commitment
    first = seed_commitment("opaque-scene", 101, "a" * 64)
    assert first != seed_commitment("opaque-scene", 101, "b" * 64)
    assert first != seed_commitment("opaque-scene", 102, "a" * 64)
    assert first != seed_commitment("different-scene", 101, "a" * 64)
    payload = _task().model_dump()
    payload["seed_commitment"] = "not-a-sha256"
    with pytest.raises(ValidationError):
        WitnessTask(**payload)


def test_recomposition_schema_reaches_miner_without_private_answers():
    from witness.subnet.validator import public_task_spec
    scene = {"duration": 2, "fps": 24, "difficulty": 1, "schema_version": "1.5",
             "qa": [{"id": "q1", "q": "Which object?", "a": "private answer"}]}
    payload = _task().model_dump()
    payload["task_spec"] = public_task_spec(scene)
    task = WitnessTask(**payload)
    normalized = task.task_spec
    assert normalized["schema_version"] == "1.5"
    assert normalized["qa"] == [{"id": "q1", "q": "Which object?"}]
    payload["task_spec"]["schema_version"] = "unknown-private-format"
    with pytest.raises(ValidationError, match="supported scene schema"):
        WitnessTask(**payload)
