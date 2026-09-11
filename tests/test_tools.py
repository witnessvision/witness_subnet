from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from witness.tools.metering import PATCH_SIZE, visual_token_cost
from witness.tools.server import create_app, degraded_transcript, discover_scenes


def test_validator_sessions_cannot_be_reissued_by_miners(tmp_path):
    app = create_app([SCENES], log_dir=tmp_path / "logs", allow_session_creation=False)
    client = TestClient(app)
    response = client.post("/session", json={"scene_id": "scene_101", "budget": _budget()})
    assert response.status_code == 403
    assert not app.state.store.sessions


def test_inflight_request_cannot_charge_a_finalized_session(tmp_path):
    import pytest
    from witness.tools.server import Budget, SessionClosed
    from witness.tools.metering import Cost
    app = create_app([SCENES], log_dir=tmp_path / "logs")
    store = app.state.store
    session = store.create("scene_101", Budget(**_budget()))
    with store.lock:
        store.sessions.pop(session.session_id)
    with pytest.raises(SessionClosed):
        store.charge(session, "GET /frame", {}, Cost(visual_tokens=100))
    assert session.cost.visual_tokens == 0


SCENES = Path(__file__).parents[1] / "data/scenes/synthetic"
REAL_SCENE_8202 = Path(__file__).parents[1] / "data/scenes/real-v14" / "scene_8202"


def _budget(**overrides: int | float) -> dict[str, int | float]:
    budget: dict[str, int | float] = {
        "visual_tokens": 100_000,
        "audio_seconds": 60,
        "transcript_chars": 10_000,
    }
    budget.update(overrides)
    return budget


def test_session_lifecycle_and_frame_archive(tmp_path: Path) -> None:
    app = create_app([SCENES], log_dir=tmp_path / "logs")
    client = TestClient(app)

    created = client.post("/session", json={"scene_id": "scene_101", "budget": _budget()})
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    assert created.json()["cost"]["visual_tokens"] == 0

    meta = client.get(f"/s/{session_id}/meta")
    assert meta.json() == {
        "duration": 25.0,
        "cost": {"visual_tokens": 0, "audio_seconds": 0.0, "transcript_chars": 0},
    }

    frames = client.get(
        f"/s/{session_id}/frames",
        params={"t0": 0, "t1": 1, "fps": 2, "res": "28x28"},
    )
    assert frames.status_code == 200
    assert frames.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(BytesIO(frames.content)) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".jpg"))
        assert names == ["frame_000000.jpg", "frame_000001.jpg"]
        assert all(archive.read(name).startswith(b"\xff\xd8") for name in names)
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["timestamps"] == [0.0, 0.5]
    assert json.loads(frames.headers["x-witness-cost"])["visual_tokens"] == 8

    log_lines = (tmp_path / "logs" / f"{session_id}.jsonl").read_text().splitlines()
    assert [json.loads(line)["endpoint"] for line in log_lines] == [
        "POST /session",
        "GET /meta",
        "GET /frames",
    ]


def test_visual_metering_uses_ceil_fourteen_pixel_patches() -> None:
    assert PATCH_SIZE == 14
    assert visual_token_cost(14, 14) == 1
    assert visual_token_cost(15, 14) == 2
    assert visual_token_cost(28, 28, frames=3) == 12
    assert visual_token_cost(640, 360) == 46 * 26


def test_batch_matches_individual_seek_frames_and_repeated_samples():
    from witness.tools.server import _extract_frame, _extract_frames, DEFAULT_FFMPEG
    video = SCENES / 'scene_101' / 'video.mp4'
    timestamps = [0., .001, .04, 1 / 24, .041667, 1/24+.000041, 1/24+.000042,
                  3., 3.001, 3.125, 7.499,
                  7.5, 7.501, 15.125, 24.958333333]
    ffmpeg = Path(DEFAULT_FFMPEG)
    batched = _extract_frames(video, timestamps, 160, 90, 24., ffmpeg)
    singles = [_extract_frame(video, value, 160, 90, ffmpeg) for value in timestamps]
    assert batched == singles


def test_grounded_batch_keeps_archive_order_and_cost_across_decoder_batches(tmp_path):
    from witness.tools.server import _extract_frame, DEFAULT_FFMPEG
    app = create_app(SCENES, log_dir=tmp_path/'logs')
    # This fixture is also a zero-origin CFR render; select the versioned path.
    app.state.store.scenes['scene_101'].truth['schema_version'] = '3.0'
    client = TestClient(app)
    session = client.post('/session', json={'scene_id': 'scene_101', 'budget': _budget()}).json()
    response = client.get(f'/s/{session["session_id"]}/frames',
        params={'t0': 0., 't1': 520/24, 'fps': 24, 'res': '14x14'})
    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert len(manifest['timestamps']) == 520
        assert manifest['cost']['visual_tokens'] == 520
        for index in [0, 511, 512, 519]:
            assert archive.read(f'frame_{index:06d}.jpg') == _extract_frame(
                SCENES/'scene_101/video.mp4', index/24, 14, 14, Path(DEFAULT_FFMPEG))


def test_frame_batch_near_fractional_scene_end_clamps_to_last_frame(tmp_path: Path) -> None:
    app = create_app([REAL_SCENE_8202], log_dir=tmp_path / "logs")
    client = TestClient(app)
    created = client.post(
        "/session", json={"scene_id": "scene_8202", "budget": _budget()}
    )
    session_id = created.json()["session_id"]
    scene = json.loads((REAL_SCENE_8202 / "scene.json").read_text(encoding="utf-8"))
    end = float(scene["duration"])
    frames = client.get(
        f"/s/{session_id}/frames",
        params={"t0": end - 2.541333, "t1": end, "fps": 4, "res": "160x90"},
    )
    assert frames.status_code == 200
    with zipfile.ZipFile(BytesIO(frames.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["timestamps"]
    assert manifest["timestamps"][-1] <= (scene["duration_frames"] - 1) / scene["fps"]


def test_budget_refusal_is_429_and_does_not_charge(tmp_path: Path) -> None:
    app = create_app([SCENES], log_dir=tmp_path / "logs")
    client = TestClient(app)
    created = client.post(
        "/session",
        json={"scene_id": "scene_101", "budget": _budget(visual_tokens=3)},
    )
    session_id = created.json()["session_id"]

    refused = client.get(f"/s/{session_id}/frame", params={"t": 0, "res": "28x28"})
    assert refused.status_code == 429
    assert refused.json()["requested_cost"]["visual_tokens"] == 4
    assert refused.json()["cost"]["visual_tokens"] == 0

    last_log = json.loads(
        (tmp_path / "logs" / f"{session_id}.jsonl").read_text().splitlines()[-1]
    )
    assert last_log["status"] == 429
    assert last_log["charged"] is False


def test_audio_and_transcript_costs_accumulate(tmp_path: Path) -> None:
    app = create_app(SCENES, log_dir=tmp_path / "logs", transcript_wer=1.0)
    client = TestClient(app)
    created = client.post(
        "/session", json={"scene_id": "scene_101", "budget": _budget()}
    )
    session_id = created.json()["session_id"]

    audio = client.get(f"/s/{session_id}/audio", params={"t0": 0, "t1": 0.25})
    assert audio.status_code == 200 and audio.content.startswith(b"RIFF")
    assert json.loads(audio.headers["x-witness-cost"])["audio_seconds"] == 0.25

    transcript = client.get(
        f"/s/{session_id}/transcript", params={"t0": 0, "t1": 25}
    )
    returned_chars = sum(len(entry["text"]) for entry in transcript.json()["entries"])
    assert transcript.json()["cost"] == {
        "visual_tokens": 0,
        "audio_seconds": 0.25,
        "transcript_chars": returned_chars,
    }


def test_degraded_transcript_is_seeded_and_coarsely_timed() -> None:
    scene = discover_scenes([SCENES])["scene_202"]
    first = degraded_transcript(scene, word_error_rate=1.0, seed=77, timestamp_quantum=2.0)
    second = degraded_transcript(scene, word_error_rate=1.0, seed=77, timestamp_quantum=2.0)

    assert first == second
    assert first[0]["text"].split() == ["<unk>"] * len(
        scene.truth["dialogue"][0]["text"].split()
    )
    assert first[0]["start"] % 2 == 0
    assert "start_frame" not in first[0] and "start_sample" not in first[0]
