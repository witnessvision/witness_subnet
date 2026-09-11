"""Pillow frame renderer and NumPy audio compositor."""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scene import COLORS, state_at
from .tts import SpeechClip, synthesize

FFMPEG = Path(
    os.environ.get("WITNESS_FFMPEG")
    or (str(Path.home() / "bin" / "ffmpeg") if (Path.home() / "bin" / "ffmpeg").is_file() else "")
    or shutil.which("ffmpeg")
    or "/usr/bin/ffmpeg"
)
ROOM_BG = "#ebe5d7"
WALL = "#d6cdbb"
FLOOR = "#a88968"
INK = "#17212b"


def camera_crop(scene: dict[str, Any], frame: int) -> list[float]:
    shot = next(s for s in scene["shots"] if s["start_frame"] <= frame < s["end_frame"])
    if shot["camera"] != "pan":
        return [float(v) for v in shot["crop"]]
    span = max(1, shot["end_frame"] - shot["start_frame"] - 1)
    ratio = (frame - shot["start_frame"]) / span
    return [
        shot["crop_start"][i] + (shot["crop_end"][i] - shot["crop_start"][i]) * ratio
        for i in range(4)
    ]


def world_to_screen(scene: dict[str, Any], frame: int, point: list[float]) -> tuple[int, int]:
    x0, y0, x1, y1 = camera_crop(scene, frame)
    width, height = scene["resolution"]
    return (
        round((point[0] - x0) * width / (x1 - x0)),
        round((point[1] - y0) * height / (y1 - y0)),
    )


def render_frame(scene: dict[str, Any], frame: int) -> Image.Image:
    width, height = scene["resolution"]
    world = Image.new("RGB", (width, height), ROOM_BG)
    draw = ImageDraw.Draw(world)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width - 1, 235), fill=WALL, outline=INK, width=2)
    draw.rectangle((0, 236, width - 1, height - 1), fill=FLOOR)
    draw.line((0, 236, width, 236), fill=INK, width=3)
    draw.rectangle((282, 146, 455, 258), fill="#78563e", outline=INK, width=3)
    draw.rectangle((300, 116, 438, 150), fill="#9b7253", outline=INK, width=3)
    draw.rectangle((552, 112, 628, 236), fill="#7a5742", outline=INK, width=3)
    draw.rectangle((563, 126, 617, 235), fill="#5f88a8", outline=INK, width=2)
    draw.ellipse((603, 180, 610, 187), fill="#e6bb4b", outline=INK)

    resolved = state_at(scene, frame)
    object_defs = {item["id"]: item for item in scene["objects"]}
    for object_id, obj in resolved["objects"].items():
        if obj["visible"]:
            _draw_object(
                draw,
                object_defs[object_id]["kind"],
                obj["position"],
                COLORS[obj["color"]],
                object_id if scene["debug_labels"] else None,
                font,
            )
    actor_defs = {item["id"]: item for item in scene["actors"]}
    for actor_id, actor in resolved["actors"].items():
        if actor["visible"]:
            _draw_actor(draw, actor_defs[actor_id], actor["position"], font, scene["debug_labels"])

    crop = camera_crop(scene, frame)
    cropped = world.crop(tuple(round(v) for v in crop))
    if cropped.size != (width, height):
        cropped = cropped.resize((width, height), Image.Resampling.BILINEAR)
    overlay = ImageDraw.Draw(cropped)
    for item in scene["on_screen_text"]:
        if item["start_frame"] <= frame < item["end_frame"]:
            _draw_overlay(overlay, item, font)
    if scene["debug_labels"]:
        shot = next(s for s in scene["shots"] if s["start_frame"] <= frame < s["end_frame"])
        overlay.rectangle((8, height - 25, 112, height - 7), fill="#111827")
        overlay.text((13, height - 22), shot["camera"].upper(), fill="white", font=font)
    return cropped


def _draw_actor(
    draw: ImageDraw.ImageDraw,
    definition: dict[str, Any],
    position: list[float],
    font: ImageFont.ImageFont,
    debug_labels: bool,
) -> None:
    x, y = (round(position[0]), round(position[1]))
    color = COLORS[definition["color"]]
    if definition["shape"] == "circle":
        draw.ellipse((x - 20, y - 48, x + 20, y - 8), fill=color, outline=INK, width=3)
    else:
        draw.rectangle((x - 19, y - 47, x + 19, y - 9), fill=color, outline=INK, width=3)
    draw.line((x - 10, y - 7, x - 15, y + 18), fill=INK, width=4)
    draw.line((x + 10, y - 7, x + 15, y + 18), fill=INK, width=4)
    if debug_labels:
        label = definition["name"]
        box = draw.textbbox((0, 0), label, font=font)
        tw = box[2] - box[0]
        draw.rectangle((x - tw // 2 - 3, y - 68, x + tw // 2 + 3, y - 53), fill="#ffffff", outline=INK)
        draw.text((x - tw // 2, y - 66), label, fill=INK, font=font)


def _draw_object(
    draw: ImageDraw.ImageDraw,
    kind: str,
    position: list[float],
    color: str,
    label: str | None,
    font: ImageFont.ImageFont,
) -> None:
    x, y = (round(position[0]), round(position[1]))
    if kind == "mug":
        draw.rectangle((x - 10, y - 9, x + 8, y + 8), fill=color, outline=INK, width=2)
        draw.ellipse((x + 5, y - 5, x + 15, y + 5), fill=color, outline=INK, width=2)
    elif kind == "key":
        draw.ellipse((x - 10, y - 9, x + 8, y + 9), fill=color, outline=INK, width=2)
        draw.line((x + 7, y, x + 22, y), fill=color, width=5)
        draw.line((x + 18, y, x + 18, y + 7), fill=color, width=4)
    else:
        draw.rectangle((x - 13, y - 9, x + 13, y + 9), fill=color, outline=INK, width=2)
        draw.line((x - 8, y - 8, x - 8, y + 8), fill="#ffffff", width=1)
    if label is not None:
        draw.text((x - 11, y + 11), label, fill=INK, font=font)


def _draw_overlay(draw: ImageDraw.ImageDraw, item: dict[str, Any], font: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = item["bbox"]
    if item["role"] == "subtitle":
        draw.rectangle((x0, y0, x1, y1), fill="#101010", outline="#ffffff")
        draw.text((x0 + 5, y0 + 7), item["text"], fill="#ffffff", font=font)
    else:
        draw.rectangle((x0, y0, x1, y1), fill="#fff8d8", outline=INK, width=2)
        draw.text((x0 + 5, y0 + 8), item["text"], fill=INK, font=font)


def _event_sound(kind: str, length: int, sample_rate: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float64) / sample_rate
    if kind == "beep":
        env = np.sin(np.pi * np.minimum(t / max(t[-1], 1e-6), 1.0)) ** 2
        signal = np.sin(2 * np.pi * 880 * t) * env
    elif kind == "door_slam":
        rng = np.random.default_rng(9471)
        noise = rng.standard_normal(length)
        signal = (0.7 * noise + 0.3 * np.sin(2 * np.pi * 72 * t)) * np.exp(-9 * t)
    elif kind == "alarm":
        pulse = (np.mod(t, 0.30) < 0.20).astype(np.float64)
        signal = 0.65 * np.sin(2 * np.pi * 690 * t) * pulse
    else:
        raise ValueError(f"unknown audio event kind {kind}")
    return signal.astype(np.float32)


def compose_audio(scene: dict[str, Any], *, speech_clips: list[SpeechClip] | None = None) -> np.ndarray:
    sample_rate = scene["audio"]["sample_rate"]
    total = round(scene["duration_frames"] / scene["fps"] * sample_rate)
    mix = np.zeros(total, dtype=np.float32)
    if speech_clips is not None and len(speech_clips) != len(scene["dialogue"]):
        raise ValueError("speech clips must match the scene dialogue")
    for index, item in enumerate(scene["dialogue"]):
        clip = (speech_clips[index] if speech_clips is not None else
                synthesize(item["text"], voice=item["tts"]["voice"], rate=item["tts"]["rate"]))
        if clip.sample_rate != sample_rate:
            raise RuntimeError("TTS sample rate differs from the scene contract")
        if len(clip.samples) != item["end_sample"] - item["start_sample"]:
            raise RuntimeError("TTS output length changed after scene contract creation")
        start, end = item["start_sample"], item["end_sample"]
        mix[start:end] += clip.samples.astype(np.float32) / 32768.0 * 0.78
    for item in scene["audio_events"]:
        start, end = item["start_sample"], item["end_sample"]
        mix[start:end] += _event_sound(item["kind"], end - start, sample_rate) * 0.72
    peak = float(np.max(np.abs(mix)))
    if peak > 0.96:
        mix *= 0.96 / peak
    return (mix * 32767).astype(np.int16)


def render_video(scene: dict[str, Any], target: Path, *, speech_clips: list[SpeechClip] | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = scene["resolution"]
    audio = compose_audio(scene, speech_clips=speech_clips)
    with tempfile.TemporaryDirectory(prefix="witness-render-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(scene["audio"]["sample_rate"])
            wav.writeframes(audio.tobytes())
        command = [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(scene["fps"]), "-i", "pipe:0", "-i", str(wav_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", str(scene["audio"]["sample_rate"]),
            "-frames:v", str(scene["duration_frames"]), "-movflags", "+faststart", str(target),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for frame in range(scene["duration_frames"]):
                process.stdin.write(render_frame(scene, frame).tobytes())
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with status {code}")


def load_scene(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
