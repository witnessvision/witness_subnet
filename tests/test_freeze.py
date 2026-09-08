import json
from pathlib import Path

from witness import freeze


def test_lock_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_1"
    scene_dir.mkdir()
    (scene_dir / "scene.json").write_text(json.dumps({"schema_version": "1.5", "difficulty": 1, "source": {"id": "x"}}), encoding="utf-8")
    lock = freeze.build_lock("1.5", [tmp_path])
    assert lock["scene_count"] == 1 and lock["schema_versions"] == ["1.5"]
    assert freeze.verify_lock(lock, [tmp_path]) == []
    (scene_dir / "scene.json").write_text(json.dumps({"schema_version": "1.5", "difficulty": 2}), encoding="utf-8")
    problems = freeze.verify_lock(lock, [tmp_path])
    assert any("scene.json changed" in p for p in problems)


def test_code_hashes_cover_scorer_and_recomposer() -> None:
    hashes = freeze.code_hashes()
    assert any(k.startswith("witness/score/") for k in hashes)
    assert any(k.startswith("witness/recompose/") for k in hashes)
    assert "witness/contract.py" in hashes
