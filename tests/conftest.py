"""Public tests generate their media in temporary directories."""
import json
import shutil
import pytest
from witness.gen import generate


@pytest.fixture(scope="session")
def generated_scenes(tmp_path_factory):
    root = tmp_path_factory.mktemp("witness-scenes")
    for seed, tier in ((101, 1), (202, 2)):
        generate(seed, tier, root / f"scene_{seed}")
    return root


@pytest.fixture(autouse=True)
def tool_scene_paths(request, monkeypatch):
    if request.module.__name__ != "test_tools":
        return
    root = request.getfixturevalue("generated_scenes")
    monkeypatch.setattr(request.module, "SCENES", root)
    fractional = root / "scene_8202"
    if not fractional.exists():
        shutil.copytree(root / "scene_101", fractional)
        path = fractional / "scene.json"
        scene = json.loads(path.read_text())
        scene["duration"] -= 0.125
        scene["duration_frames"] -= 3
        path.write_text(json.dumps(scene))
    monkeypatch.setattr(request.module, "REAL_SCENE_8202", fractional)
