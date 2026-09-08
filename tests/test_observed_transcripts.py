import json

from fastapi.testclient import TestClient
import pytest

from witness.tools.server import create_app
from witness.tools.transcripts import load_observed_transcript, sha256


@pytest.fixture
def scene(tmp_path):
    folder = tmp_path / "scene_1"
    (folder / "observations").mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"test-media")
    (folder / "scene.json").write_text(json.dumps({
        "duration": 5, "fps": 24,
        "dialogue": [{"speaker": "secret-speaker", "text": "SECRET ANSWER", "start": 1, "end": 2}],
    }))
    (folder / "observations/transcript.json").write_text(json.dumps({
        "schema_version": "1", "source": "decoded_audio",
        "video_sha256": sha256(folder / "video.mp4"), "model_sha256": {"model.bin": "abc"},
        "entries": [{"start": 1, "end": 2, "text": "Observed words", "speaker": "untrusted"}],
    }))
    return folder


@pytest.mark.parametrize("source,expected", [("asr", "Observed words"), ("none", None)])
def test_both_transcript_endpoints_never_read_label_hints(scene, tmp_path, monkeypatch, source, expected):
    def forbidden(*args):
        raise AssertionError("label-derived transcript called")
    monkeypatch.setattr("witness.tools.server.degraded_transcript", forbidden)
    app = create_app(scene, log_dir=tmp_path / "logs", transcript_source=source)
    with TestClient(app) as client:
        session = client.post("/session", json={"scene_id": "scene_1", "budget": 100}).json()["session_id"]
        for endpoint,params in (("transcript", {"t0": 0, "t1": 5}),
                                ("search_transcript", {"q": "Observed"})):
            response = client.get(f"/s/{session}/{endpoint}", params=params)
            assert response.status_code == 200
            assert "SECRET" not in response.text and "untrusted" not in response.text
            entries = response.json().get("entries", response.json().get("matches"))
            if expected:
                assert entries[0]["text"] == expected
                assert entries[0]["speaker"] is None
            else:
                assert entries == []


def test_missing_asr_and_stale_media_fail_closed(scene, tmp_path):
    (scene / "video.mp4").write_bytes(b"changed-media")
    with pytest.raises(ValueError, match="media identity"):
        load_observed_transcript(scene, 5)
    (scene / "observations/transcript.json").unlink()
    with pytest.raises(FileNotFoundError):
        create_app(scene, log_dir=tmp_path / "logs", transcript_source="asr")


def test_impossible_transcript_timestamps_are_rejected(scene):
    path = scene / "observations/transcript.json"
    artifact = json.loads(path.read_text())
    artifact["entries"][0]["end"] = 6
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="interval"):
        load_observed_transcript(scene, 5)
