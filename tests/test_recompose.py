from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import subprocess

import numpy as np
import pytest

from witness.contract import ACTIONS, ACTIONS_EDIT
from witness.recompose.generator import (
    AUDIO_RATE,
    DIALOGUE_DUCK_PADDING_SECONDS,
    FONT_FILES,
    MIN_DIALOGUE_SOURCE_SNR_DB,
    OVERLAY_STYLES,
    SOURCE_GAIN_CHOICES_DB,
    _resolve_text_collisions,
    _text_items_collide,
    build_edit_plan,
    build_scene_from_plan,
    generate_recomposition,
    TTS_SPEEDS,
    TTS_VOICES,
)
from witness.recompose.tts import MAX_TTS_CALLS, _cache_key, synthesize_openai
from witness.score.oracle import perfect_reconstruction
from witness.score.scorer import score_reconstruction
from witness.sources.pool import CC_LICENSE, audit_licenses
from witness.tts import SpeechClip
from witness.validate import validate


FFMPEG = Path(
    os.environ.get("WITNESS_FFMPEG")
    or (str(Path.home() / "bin" / "ffmpeg") if (Path.home() / "bin" / "ffmpeg").is_file() else "")
    or shutil.which("ffmpeg")
    or "/usr/bin/ffmpeg"
)


def _manifest(path: Path, source_path: Path, *, duration: float = 300.0) -> dict:
    videos = []
    for video_id in ("source_a", "source_b"):
        videos.append(
            {
                "id": video_id,
                "url": f"https://example.invalid/{video_id}",
                "title": video_id,
                "uploader": "test",
                "license": CC_LICENSE,
                "duration": duration,
                "path": source_path.name,
            }
        )
    manifest = {"schema_version": "1.0", "videos": videos}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _mock_tts(text: str, *, voice: str, speed: float, model: str) -> SpeechClip:
    del text, speed, model
    length = round(0.75 * AUDIO_RATE)
    t = np.arange(length, dtype=np.float64) / AUDIO_RATE
    samples = (np.sin(2 * np.pi * 330 * t) * 12_000).astype(np.int16)
    return SpeechClip(samples=samples, sample_rate=AUDIO_RATE, engine="mock", voice=voice)


def test_edit_plan_is_deterministic_and_source_segments_do_not_overlap(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, tmp_path / "source.mp4")
    for tier in (1, 2, 3):
        first = build_edit_plan(801 + tier, tier, manifest)
        second = build_edit_plan(801 + tier, tier, manifest)
        assert first == second
        assert first != build_edit_plan(901 + tier, tier, manifest)
        base_shots = [shot for shot in first["shots"] if shot["kind"] == "source"]
        base = sorted(
            (shot["source_start"], shot["source_end"])
            for shot in base_shots
        )
        assert all(left[1] <= right[0] for left, right in zip(base, base[1:]))
        low, high = {1: (2, 5), 2: (4, 9), 3: (6, 12)}[tier]
        assert low <= len(base) <= high
        assert all(int(shot["output_frames"]) % 24 for shot in base_shots)
        scene, _clips = build_scene_from_plan(first, manifest_path, synthesizer=_mock_tts)
        report = score_reconstruction(scene, perfect_reconstruction(scene), {})
        assert report["quality"] == 1.0


def test_seeded_dialogue_and_text_have_high_diversity(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, tmp_path / "source.mp4")
    lines: set[str] = set()
    texts: set[str] = set()
    voices: set[str] = set()
    speeds: set[float] = set()
    styles: set[tuple] = set()
    style_names: set[str] = set()
    fonts: set[str] = set()
    fleeting = 0
    for seed in range(1000, 2000):
        plan = build_edit_plan(seed, 3, manifest)
        scene, _ = build_scene_from_plan(plan, manifest_path, synthesizer=_mock_tts)
        lines.update(item["text"] for item in scene["dialogue"])
        texts.update(item["text"] for item in scene["on_screen_text"])
        voices.update(item["tts"]["voice"] for item in scene["dialogue"])
        speeds.update(item["tts"]["speed"] for item in scene["dialogue"])
        styles.update((tuple(item["bbox"]), item["font_size"], item["overlay_color"], item["opacity"], item["background_box"]) for item in scene["on_screen_text"])
        style_names.update(item["style"] for item in scene["on_screen_text"])
        fonts.update(item["font_family"] for item in scene["on_screen_text"])
        fleeting += sum(1 for item in scene["on_screen_text"] if item.get("ephemeral"))
        for item in scene["dialogue"]:
            assert 6 <= len(item["text"].rstrip(".?!").split()) <= 14
            assert item["text"] == next(q["a"] for q in scene["qa"] if q["id"] == "said_at_time") or item["id"] != "dialogue_1"
        for item in scene["on_screen_text"]:
            assert item["font_size"] >= 18
            if item.get("ephemeral"):
                assert scene["difficulty"] == 3
                assert item["end_frame"] - item["start_frame"] >= 3
                assert item["observability"]["exception"] == "tier_3_flash"
            else:
                assert item["end_frame"] - item["start_frame"] >= 20
                assert item["observability"]["guaranteed"] is True
                assert item["observability"]["sample_frame"] % 6 == 0
        for index, left in enumerate(scene["on_screen_text"]):
            for right in scene["on_screen_text"][index + 1:]:
                time_overlap = max(left["start_frame"], right["start_frame"]) < min(
                    left["end_frame"], right["end_frame"]
                )
                left_x0, left_y0, left_x1, left_y1 = left["bbox"]
                right_x0, right_y0, right_x1, right_y1 = right["bbox"]
                space_overlap = (
                    max(left_x0, right_x0) < min(left_x1, right_x1)
                    and max(left_y0, right_y0) < min(left_y1, right_y1)
                )
                assert not (time_overlap and space_overlap)
        subtitle = next((item for item in scene["on_screen_text"] if item.get("misleading")), None)
        if subtitle is not None:
            referenced = next(item for item in scene["dialogue"] if item["id"] == subtitle["references_dialogue_id"])
            assert all(term in referenced["text"] and term in subtitle["text"] for term in subtitle["reference_terms"])
    assert len(lines) > 3000
    assert len(texts) > 2500
    assert voices == set(TTS_VOICES)
    assert speeds == set(TTS_SPEEDS)
    assert len(styles) > 3000
    assert style_names == set(OVERLAY_STYLES)
    assert fonts == set(FONT_FILES)
    assert 0 < fleeting < 1000


def test_seed_randomizes_structure_edits_errors_dialogue_and_overlay_positions(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, tmp_path / "source.mp4")
    for tier, bounds in ((1, (2, 5)), (2, (4, 9)), (3, (6, 12))):
        plans = [build_edit_plan(seed, tier, manifest) for seed in range(300, 420)]
        base_counts = {
            sum(shot["kind"] == "source" for shot in plan["shots"])
            for plan in plans
        }
        assert min(base_counts) >= bounds[0] and max(base_counts) <= bounds[1]
        assert len(base_counts) > 1
        for edit in ("repeat", "freeze", "foreign"):
            presence = [any(shot["kind"] == edit for shot in plan["shots"]) for plan in plans]
            assert any(presence) and not all(presence)
        for field in ("mirror", "tint"):
            presence = [any(bool(shot[field]) for shot in plan["shots"]) for plan in plans]
            assert any(presence) and not all(presence)
        speed_presence = [
            any(float(shot["playback_rate"]) != 1.0 for shot in plan["shots"])
            for plan in plans
        ]
        assert any(speed_presence) and not all(speed_presence)
        for field in ("misleading_subtitle", "audio_visual_contradiction"):
            presence = [bool(plan[field]) for plan in plans]
            assert any(presence) and not all(presence)
        assert len({int(plan["dialogue_count"]) for plan in plans}) > 1
        assert {float(plan["source_gain_db"]) for plan in plans} == set(SOURCE_GAIN_CHOICES_DB)

        positions = set()
        for plan in plans[:30]:
            scene, _ = build_scene_from_plan(plan, manifest_path, synthesizer=_mock_tts)
            positions.update(tuple(item["bbox"]) for item in scene["on_screen_text"])
        assert len(positions) > 20


def test_scene_truth_is_deterministic_and_tts_cache_includes_speed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, tmp_path / "source.mp4")
    plan = build_edit_plan(4242, 2, manifest)
    first, _ = build_scene_from_plan(plan, manifest_path, synthesizer=_mock_tts)
    second, _ = build_scene_from_plan(plan, manifest_path, synthesizer=_mock_tts)
    assert first == second
    assert _cache_key("same", "marin", 0.96, "gpt-4o-mini-tts", "clear") != _cache_key(
        "same", "marin", 1.04, "gpt-4o-mini-tts", "clear"
    )


def test_overlapping_subtitle_and_chip_are_relaid_deterministically() -> None:
    subtitle = {
        "id": "misleading_subtitle",
        "start_frame": 513,
        "end_frame": 578,
        "bbox": [41, 301, 599, 342],
    }
    chip = {
        "id": "code",
        "start_frame": 521,
        "end_frame": 570,
        "bbox": [529, 294, 612, 333],
    }
    assert _text_items_collide(subtitle, chip)
    first = _resolve_text_collisions([dict(subtitle), dict(chip)])
    second = _resolve_text_collisions([dict(subtitle), dict(chip)])
    assert first == second
    assert first[0]["bbox"] == subtitle["bbox"]
    assert first[1]["bbox"] == [8, 8, 91, 47]
    assert not _text_items_collide(first[0], first[1])


def test_dialogue_is_shifted_to_fit_actual_tts_duration(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, tmp_path / "source.mp4")
    plan = build_edit_plan(4242, 1, manifest)
    plan["dialogue_count"] = 1
    duration_samples = int(plan["duration_frames"]) * AUDIO_RATE // 24

    def long_tts(text: str, *, voice: str, speed: float, model: str) -> SpeechClip:
        del text, speed, model
        return SpeechClip(
            samples=np.zeros(duration_samples - AUDIO_RATE // 2, dtype=np.int16),
            sample_rate=AUDIO_RATE,
            engine="mock",
            voice=voice,
        )

    scene, clips = build_scene_from_plan(plan, manifest_path, synthesizer=long_tts)
    assert len(scene["dialogue"]) == len(clips) == 1
    assert scene["dialogue"][0]["start_frame"] == 12
    assert scene["dialogue"][0]["end_sample"] <= duration_samples
    said_at_time = next(item for item in scene["qa"] if item["id"] == "said_at_time")
    assert said_at_time["a"] == scene["dialogue"][0]["text"]


def test_unfit_last_dialogue_is_dropped_with_exact_truth(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, tmp_path / "source.mp4")
    plan = build_edit_plan(4243, 1, manifest)
    plan["dialogue_count"] = 2
    duration_samples = int(plan["duration_frames"]) * AUDIO_RATE // 24
    call_index = 0

    def tail_tts(text: str, *, voice: str, speed: float, model: str) -> SpeechClip:
        nonlocal call_index
        del text, speed, model
        call_index += 1
        length = AUDIO_RATE if call_index == 1 else duration_samples + AUDIO_RATE
        return SpeechClip(
            samples=np.zeros(length, dtype=np.int16),
            sample_rate=AUDIO_RATE,
            engine="mock",
            voice=voice,
        )

    scene, clips = build_scene_from_plan(plan, manifest_path, synthesizer=tail_tts)
    assert len(scene["dialogue"]) == len(clips) == 1
    assert scene["dialogue"][0]["id"] == "dialogue_1"
    assert any(
        event.get("anchor_id") == "dialogue_1" for event in scene["events"]
    )
    said_at_time = next(item for item in scene["qa"] if item["id"] == "said_at_time")
    assert said_at_time["a"] == scene["dialogue"][0]["text"]


def test_recomposed_media_validates_and_oracle_is_perfect(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=24000",
            "-t", "45", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
        ],
        check=True,
    )
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path, source, duration=45.0)
    scene_path, video_path = generate_recomposition(
        6101, 1, manifest_path, tmp_path / "scene", synthesizer=_mock_tts
    )
    assert validate(scene_path, video_path) == []
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    keyframes = subprocess.check_output(
        [
            str(FFMPEG.with_name("ffprobe")), "-v", "error", "-skip_frame", "nokey",
            "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0",
            str(video_path),
        ],
        text=True,
    ).splitlines()
    assert len(keyframes) >= int(scene["duration"])
    report = score_reconstruction(scene, perfect_reconstruction(scene), {})
    assert report["quality"] == 1.0
    assert set(report["family_scores"].values()) == {1.0}
    assert scene["schema_version"] == "1.5"
    assert scene["audio"]["dialogue_truth"] == "scripted_overlay_only"
    assert scene["audio"]["source_audio_role"] == "background"
    source_mix = scene["audio"]["source_mix"]
    assert source_mix["gain_db"] in SOURCE_GAIN_CHOICES_DB
    assert source_mix["dialogue_duck_padding_seconds"] == DIALOGUE_DUCK_PADDING_SECONDS
    assert source_mix["minimum_dialogue_source_snr_db"] == MIN_DIALOGUE_SOURCE_SNR_DB
    assert min(item["tts"]["source_snr_lower_bound_db"] for item in scene["dialogue"]) >= 12.0
    assert set(event["action"] for event in scene["events"]) <= set(ACTIONS)
    assert set(ACTIONS_EDIT) <= set(ACTIONS)

    raw = subprocess.check_output(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "pipe:1",
        ]
    )
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    dialogue_start = float(scene["dialogue"][0]["start"])

    def tone_amplitude(start: float, frequency: float = 220.0, duration: float = 0.5) -> float:
        begin = round(start * AUDIO_RATE)
        samples = pcm[begin:begin + round(duration * AUDIO_RATE)]
        phase = np.exp(-2j * np.pi * frequency * np.arange(len(samples)) / AUDIO_RATE)
        return float(2 * abs(np.dot(samples, phase)) / len(samples))

    background_tone = tone_amplitude(dialogue_start - 2.0)
    ducked_tone = tone_amplitude(dialogue_start + 0.1)
    assert ducked_tone < background_tone * 0.3

    broken = json.loads(scene_path.read_text(encoding="utf-8"))
    normal_text = next(item for item in broken["on_screen_text"] if not item.get("ephemeral"))
    normal_text["observability"]["sample_frame"] += 1
    broken_path = tmp_path / "broken-observability.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    assert any("not guaranteed observable" in failure for failure in validate(broken_path, video_path))


def test_foreign_insert_prefers_same_query_tag(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, tmp_path / "source.mp4")
    manifest["videos"] = [
        {**manifest["videos"][0], "id": "food_a", "query_tag": "street food"},
        {**manifest["videos"][1], "id": "food_b", "query_tag": "street food"},
        {**manifest["videos"][0], "id": "sport_a", "query_tag": "sports"},
        {**manifest["videos"][1], "id": "sport_b", "query_tag": "sports"},
    ]
    plans = [build_edit_plan(seed, 3, manifest) for seed in range(300, 450)]
    with_foreign = [plan for plan in plans if plan["foreign_source"]]
    assert with_foreign
    assert all(
        plan["source"]["query_tag"] == plan["foreign_source"]["query_tag"]
        and plan["foreign_selection"]["same_genre_match"] is True
        for plan in with_foreign
    )


def test_openai_tts_hard_cap_is_enforced_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_log = tmp_path / "calls.jsonl"
    call_log.write_text("{}\n" * MAX_TTS_CALLS, encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    with pytest.raises(RuntimeError, match="call cap reached"):
        synthesize_openai(
            "This request must never be sent.",
            call_log=call_log,
            cache_dir=tmp_path / "cache",
        )


def test_license_audit_marks_revoked_and_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"videos": [{"id": "ok"}, {"id": "revoked"}, {"id": "gone"}]}), encoding="utf-8")

    def metadata(video_id: str) -> dict:
        if video_id == "gone":
            raise RuntimeError("not available")
        return {"license": CC_LICENSE if video_id == "ok" else "Standard YouTube License"}

    monkeypatch.setattr("witness.sources.pool._metadata", metadata)
    manifest, failures = audit_licenses(manifest_path)
    assert manifest["license_audit"] == {
        "audited_at": manifest["license_audit"]["audited_at"],
        "verified": 1,
        "revoked": 1,
        "unavailable": 1,
    }
    by_id = {item["id"]: item["license_audit"] for item in manifest["videos"]}
    assert by_id["ok"]["revoked"] is False
    assert by_id["revoked"]["revoked"] is True
    assert by_id["gone"]["revoked"] is None
    assert len(failures) == 1
