"""Independent media/contract validation for generated Witness scenes."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .contract import ACTIONS, ACTIONS_EDIT
from .render import FFMPEG, load_scene, world_to_screen
from .scene import COLORS

FFPROBE = FFMPEG.with_name("ffprobe")


def _probe(video: Path) -> dict[str, Any]:
    raw = subprocess.check_output(
        [str(FFPROBE), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)]
    )
    return json.loads(raw)


def _frame(video: Path, number: int) -> np.ndarray:
    payload = subprocess.check_output(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", f"select=eq(n\\,{number})", "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ]
    )
    if not payload:
        raise AssertionError(f"could not decode frame {number}")
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.int16)


def _rgb(hex_color: str) -> np.ndarray:
    value = hex_color.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def validate(scene_path: Path, video_path: Path | None = None) -> list[str]:
    scene = load_scene(scene_path)
    video = video_path or scene_path.with_name("video.mp4")
    if scene.get("schema_version") == "3.0":
        from witness.validate_grounded import validate_grounded
        return validate_grounded(scene, video)
    failures: list[str] = []
    required = {"seed", "difficulty", "debug_labels", "duration_frames", "fps", "resolution", "events", "dialogue", "on_screen_text", "audio_events", "qa"}
    missing = required - scene.keys()
    if missing:
        failures.append(f"scene.json missing keys: {sorted(missing)}")
        return failures
    if not isinstance(scene["debug_labels"], bool):
        failures.append("debug_labels must be a boolean")
    if scene["difficulty"] in (2, 3) and scene["debug_labels"]:
        failures.append("debug labels are forbidden at tiers 2 and 3")
    if scene.get("schema_version") in {"1.3", "1.4", "1.5"}:
        source = scene.get("source")
        required_source = {"id", "url", "title", "uploader", "license", "duration", "path", "manifest_sha256"}
        if not isinstance(source, dict) or not required_source <= source.keys():
            failures.append("recomposed scene.json requires complete source provenance")
        if not scene.get("shots"):
            failures.append("recomposed scene.json requires at least one recomposed shot")
        else:
            cursor = 0
            for shot in scene["shots"]:
                if shot.get("start_frame") != cursor or shot.get("end_frame", 0) <= cursor:
                    failures.append(f"non-contiguous or invalid recomposed shot: {shot.get('id')}")
                cursor = int(shot.get("end_frame", cursor))
                if shot.get("kind") not in ("source", "repeat", "freeze", "foreign"):
                    failures.append(f"invalid recomposed shot kind: {shot.get('kind')}")
                if float(shot.get("source_end", 0)) <= float(shot.get("source_start", 0)):
                    failures.append(f"invalid source interval: {shot.get('id')}")
            if cursor != scene["duration_frames"]:
                failures.append("recomposed shots do not cover duration_frames")
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    for entity_type, group_name in (("actor", "actors"), ("object", "objects")):
        for entity in scene.get(group_name, []):
            identity_fields = {"name", "visual_description", "name_grounded_by"}
            if not identity_fields <= entity.keys():
                failures.append(f"{entity_type} {entity.get('id')} lacks observable identity metadata")
                continue
            if entity["name_grounded_by"] not in (None, "dialogue", "label"):
                failures.append(f"{entity_type} {entity['id']} has invalid name grounding")
            entities[(entity_type, entity["id"])] = entity
    for entry in scene["qa"]:
        text = f'{entry["q"]} {entry["a"]}'.casefold()
        for reference in entry.get("references", []):
            entity = entities.get((reference.get("entity_type"), reference.get("id")))
            if entity is None:
                failures.append(f"QA {entry['id']} references an unknown entity")
                continue
            surface = reference.get("surface", "")
            if surface.casefold() not in text:
                failures.append(f"QA {entry['id']} omits its declared entity surface")
            if reference.get("grounded_by") == "visual_description":
                if surface != entity["visual_description"]:
                    failures.append(f"QA {entry['id']} does not use the canonical visual description")
            elif surface != entity["name"] or entity["name_grounded_by"] not in ("dialogue", "label"):
                failures.append(f"QA {entry['id']} uses an ungrounded entity name")

    probe = _probe(video)
    streams = probe["streams"]
    video_streams = [s for s in streams if s["codec_type"] == "video"]
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        failures.append("expected exactly one video and one audio stream")
        return failures
    stream = video_streams[0]
    if [stream["width"], stream["height"]] != scene["resolution"]:
        failures.append("encoded resolution does not match scene.json")
    if stream.get("r_frame_rate") != f'{scene["fps"]}/1':
        failures.append("encoded fps does not match scene.json")
    if int(stream.get("nb_frames", -1)) != scene["duration_frames"]:
        failures.append("encoded frame count does not match scene.json")
    duration = float(probe["format"]["duration"])
    if abs(duration - scene["duration"]) > 1 / scene["fps"]:
        failures.append(f"encoded duration differs by more than one frame: {duration}")

    for event in scene["events"]:
        if abs(event["t"] - event["frame"] / scene["fps"]) > 1e-6:
            failures.append(f"event timestamp is not frame-derived: {event}")
        if event.get("action") not in ACTIONS:
            failures.append(f"unknown event action: {event.get('action')}")
    if scene.get("schema_version") in {"1.3", "1.4", "1.5"} and not any(
        event.get("action") in ACTIONS_EDIT for event in scene["events"]
    ):
        failures.append("recomposed scene.json has no edit events")
    for item in scene["on_screen_text"]:
        if item["end_frame"] <= item["start_frame"]:
            failures.append(f"invalid text interval: {item['id']}")
        if scene.get("schema_version") == "1.5":
            observability = item.get("observability")
            if not isinstance(observability, dict):
                failures.append(f"text {item.get('id')} lacks observability metadata")
                continue
            duration_ms = (item["end_frame"] - item["start_frame"]) / scene["fps"] * 1000
            if observability.get("exception") == "tier_3_flash":
                if scene["difficulty"] != 3 or not item.get("ephemeral"):
                    failures.append(f"text {item.get('id')} uses tier-3 flash exception outside tier 3")
                if duration_ms + 1e-6 < 125:
                    failures.append(f"tier-3 flash {item.get('id')} is shorter than 125 ms")
            else:
                if duration_ms + 1e-6 < 800:
                    failures.append(f"text {item.get('id')} is shorter than 0.8 seconds")
                if int(item.get("font_size", 0)) < 18:
                    failures.append(f"text {item.get('id')} is shorter than 18 px at 640x360")
                sample_frame = observability.get("sample_frame")
                stride = int(scene["fps"]) // 4
                valid_sample = (
                    isinstance(sample_frame, int)
                    and item["start_frame"] <= sample_frame < item["end_frame"]
                    and sample_frame % stride == 0
                )
                if (
                    observability.get("guaranteed") is not True
                    or observability.get("sampling_fps") != 4
                    or observability.get("resolution") != [320, 180]
                    or not valid_sample
                ):
                    failures.append(f"text {item.get('id')} is not guaranteed observable at 4 fps 320x180")
    for item in scene["dialogue"]:
        expected_end = item["end_sample"] / scene["audio"]["sample_rate"]
        if abs(item["end"] - expected_end) > 1e-6:
            failures.append("dialogue end is not sample-derived")
        if scene.get("schema_version") == "1.5":
            source_mix = scene["audio"].get("source_mix", {})
            try:
                mixed_tts_dbfs = float(item["tts"]["rms_dbfs"]) + float(item["tts"]["mix_gain_db"])
                lower_bound = mixed_tts_dbfs - float(source_mix["dialogue_duck_gain_db"])
                declared = float(item["tts"]["source_snr_lower_bound_db"])
                minimum = float(source_mix["minimum_dialogue_source_snr_db"])
            except (KeyError, TypeError, ValueError):
                failures.append(f"dialogue {item.get('id')} lacks SNR validation metadata")
            else:
                if abs(declared - lower_bound) > 0.01:
                    failures.append(f"dialogue {item.get('id')} has inconsistent SNR metadata")
                if lower_bound + 1e-6 < minimum or minimum < 12:
                    failures.append(f"dialogue {item.get('id')} source SNR is below 12 dB")

    frame_cache: dict[int, np.ndarray] = {}
    for check in scene["validation_checks"]:
        kind = check["kind"]
        if kind == "object_color":
            number = check["frame"]
            frame = frame_cache.setdefault(number, _frame(video, number))
            x, y = world_to_screen(scene, number, check["sample_world"])
            if not (3 <= x < frame.shape[1] - 3 and 3 <= y < frame.shape[0] - 3):
                failures.append(f"object check is outside camera at frame {number}")
                continue
            observed = np.median(frame[y - 3:y + 4, x - 3:x + 4], axis=(0, 1))
            expected = _rgb(COLORS[check["expected"]])
            distance = float(np.linalg.norm(observed - expected))
            if distance > 35:
                failures.append(f"object color mismatch at frame {number}: distance {distance:.1f}")
        elif kind == "hard_cut":
            number = check["frame"]
            before = frame_cache.setdefault(number - 1, _frame(video, number - 1))
            after = frame_cache.setdefault(number, _frame(video, number))
            difference = float(np.mean(np.abs(after - before)))
            if difference < check["minimum_mean_difference"]:
                failures.append(f"cut at frame {number} too weak: {difference:.2f}")
        elif kind == "short_text":
            duration_ms = (check["end_frame"] - check["start_frame"]) / scene["fps"] * 1000
            if duration_ms > check["maximum_duration_ms"]:
                failures.append(f"short text lasts {duration_ms:.1f} ms")
        elif kind == "text_presence":
            number = check["frame"]
            frame = frame_cache.setdefault(number, _frame(video, number))
            x0, y0, x1, y1 = (int(value) for value in check["bbox"])
            if not (0 <= x0 < x1 <= frame.shape[1] and 0 <= y0 < y1 <= frame.shape[0]):
                failures.append(f"text check is outside frame at frame {number}")
                continue
            expected = _rgb(check["expected_color"])
            pixels = frame[y0:y1, x0:x1].reshape(-1, 3)
            pixel_distances = np.linalg.norm(pixels - expected, axis=1)
            if check.get("pixel_statistic") == "nearest":
                distance = float(np.min(pixel_distances))
            elif check.get("pixel_statistic") == "percentile_25":
                distance = float(np.percentile(pixel_distances, 25))
            else:
                observed = np.median(pixels, axis=0)
                distance = float(np.linalg.norm(observed - expected))
            if distance > float(check["maximum_color_distance"]):
                failures.append(
                    f"text overlay {check.get('text_id', '?')} missing at frame {number}: color distance {distance:.1f}"
                )

    if scene.get("schema_version") == "1.5":
        for item in scene["on_screen_text"]:
            observability = item["observability"]
            if observability.get("exception") == "tier_3_flash":
                continue
            number = int(observability["sample_frame"])
            frame = frame_cache.setdefault(number, _frame(video, number))
            small = np.asarray(
                Image.fromarray(frame.astype(np.uint8)).resize((320, 180), Image.Resampling.LANCZOS),
                dtype=np.int16,
            )
            x0, y0, x1, y1 = (int(value) for value in item["bbox"])
            x0, x1 = x0 // 2, max(x0 // 2 + 1, (x1 + 1) // 2)
            y0, y1 = y0 // 2, max(y0 // 2 + 1, (y1 + 1) // 2)
            expected_color = item["overlay_color"] if item["background_box"] else item["font_color"]
            expected = _rgb(expected_color)
            pixels = small[y0:y1, x0:x1].reshape(-1, 3)
            if item["background_box"]:
                distance = float(np.percentile(np.linalg.norm(pixels - expected, axis=1), 25))
                maximum = min(235.0, 442 * (1 - float(item["opacity"])) + 55)
            else:
                distance = float(np.min(np.linalg.norm(pixels - expected, axis=1)))
                maximum = min(190.0, 442 * (1 - float(item["opacity"])) + 65)
            if distance > maximum:
                failures.append(
                    f"text {item.get('id')} is not observable at 4 fps 320x180: color distance {distance:.1f}"
                )

    # Decode PCM and confirm each declared synthetic sound has substantial energy.
    with tempfile.TemporaryDirectory(prefix="witness-validate-") as temp_dir:
        pcm_path = Path(temp_dir) / "audio.raw"
        subprocess.run(
            [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(scene["audio"]["sample_rate"]), "-f", "s16le", str(pcm_path)],
            check=True,
        )
        pcm = np.fromfile(pcm_path, dtype=np.int16).astype(np.float64)
        for item in scene["audio_events"]:
            chunk = pcm[item["start_sample"]:item["end_sample"]]
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            if rms < 1000:
                failures.append(f"audio event {item['kind']} is missing or too quiet: RMS {rms:.1f}")
        for item in scene["dialogue"]:
            chunk = pcm[item["start_sample"]:item["end_sample"]]
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            if rms < 400:
                failures.append(f"dialogue by {item['speaker']} is missing or too quiet: RMS {rms:.1f}")
    if scene.get("schema_version") == "2.0":
        from .validate_temporal import validate_temporal
        failures.extend(validate_temporal(scene, video))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scene.json against its video.mp4")
    parser.add_argument("scene", type=Path, help="path to scene.json or its directory")
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    scene_path = args.scene / "scene.json" if args.scene.is_dir() else args.scene
    failures = validate(scene_path, args.video)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"PASS: {scene_path}")


if __name__ == "__main__":
    main()
