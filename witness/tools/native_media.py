"""Original-media decoding with session deadline and explicit cancellation."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_original(video: Path, ffprobe: Path) -> dict:
    result = subprocess.run([str(ffprobe), "-v", "error", "-show_streams", "-show_format",
                             "-of", "json", str(video)], capture_output=True, check=True, timeout=30)
    metadata = json.loads(result.stdout)
    streams = metadata["streams"]
    visual = next(s for s in streams if s["codec_type"] == "video")
    duration = float(metadata["format"]["duration"])
    fps = float(Fraction(visual["avg_frame_rate"]))
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid native media timing")
    audio = next((s for s in streams if s["codec_type"] == "audio"), {})
    return {"schema_version": "5.0", "duration": duration, "fps": fps,
            "video_duration": float(visual.get("duration", duration)),
            "has_audio": bool(audio), "media_sha256": file_sha256(video),
            "audio": {"sample_rate": int(audio.get("sample_rate", 48000)),
                      "channels": int(audio.get("channels", 1))}}


def run_decoder(command: list[str], session, *, capture_timing: bool = False):
    from witness.tools.server import _FFMPEG_SLOTS, SessionClosed

    def check():
        if session.closed.is_set() or (session.deadline_at is not None and time.monotonic() >= session.deadline_at):
            raise SessionClosed()

    while not _FFMPEG_SLOTS.acquire(timeout=.05):
        check()
    process = None
    try:
        check()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            try:
                output, diagnostic = process.communicate(timeout=.05)
                break
            except subprocess.TimeoutExpired:
                check()
        check()
        if process.returncode:
            raise RuntimeError("original media decoder failed")
        return (output, diagnostic) if capture_timing else output
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        _FFMPEG_SLOTS.release()


def decoded_times(diagnostic: bytes, *, offset: float, count: int) -> list[float]:
    values = [float(value) + offset for value in re.findall(
        rb"\bpts_time:([-+0-9.eE]+)\s", diagnostic)]
    if len(values) < count:
        raise RuntimeError("decoder did not report original frame timestamps")
    return values[:count]


def frames(session, t0: float, t1: float, fps: float, width: int, height: int,
           count: int, ffmpeg: Path) -> tuple[list[bytes], list[float]]:
    """Seek to the requested window, retaining the original time origin.

    FFmpeg samples VFR or CFR input at the requested observation times. This is
    an observation sample, never a replacement or time-normalized source video.
    """
    last = max(0., float(session.scene.truth.get("video_duration", session.scene.duration))
               - 1/float(session.scene.truth["fps"]))
    if fps > float(session.scene.truth["fps"]):
        observed = [frame(session, min(t0+i/fps, last),
                          width, height, ffmpeg) for i in range(count)]
        return [value[0] for value in observed], [value[1] for value in observed]
    # Container/audio duration can exceed the final native visual frame. Return
    # that actual terminal observation (and its original timestamp), never a
    # made-up frame or a timestamp at which no visual sample existed.
    available = max(0, min(count, math.floor((last-t0)*fps+1e-9)+1))
    if available < count:
        images, timestamps = (frames(session,t0,t1,fps,width,height,available,ffmpeg)
                              if available else ([],[]))
        terminal, timestamp = frame(session,last,width,height,ffmpeg)
        return images+[terminal]*(count-available), timestamps+[timestamp]*(count-available)
    with tempfile.TemporaryDirectory(prefix="witness-native-frames-") as directory:
        source_t0=t0+float(session.scene.truth.get("source_offset_s",0.))
        _, diagnostic = run_decoder([str(ffmpeg), "-v", "info", "-threads", "1", "-filter_threads", "1",
            "-ss", f"{source_t0:.9f}", "-t", f"{t1-t0:.9f}", "-i", str(session.scene.video_path),
            "-an", "-frames:v", str(count), "-fps_mode", "passthrough",
            "-vf", f"select=gte(t\\,selected_n/{fps}),showinfo,scale={width}:{height}:flags=lanczos",
            "-q:v", "2", "-threads", "1", "-map_metadata", "-1", "-map_metadata:s", "-1",
            "-map_chapters", "-1", str(Path(directory) / "%06d.jpg")], session, capture_timing=True)
        paths = sorted(Path(directory).glob("*.jpg"))
        if len(paths) != count:
            raise RuntimeError("original media did not supply every requested observation")
        return [p.read_bytes() for p in paths], decoded_times(diagnostic, offset=t0, count=count)


def frame(session, t: float, width: int, height: int, ffmpeg: Path) -> tuple[bytes, float]:
    source_t=t+float(session.scene.truth.get("source_offset_s",0.))
    output, diagnostic = run_decoder([str(ffmpeg), "-v", "info", "-threads", "1", "-filter_threads", "1",
        "-ss", f"{source_t:.9f}", "-i", str(session.scene.video_path), "-frames:v", "1",
        "-vf", f"showinfo,scale={width}:{height}:flags=lanczos", "-q:v", "2", "-threads", "1",
        "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"], session, capture_timing=True)
    return output, decoded_times(diagnostic, offset=t, count=1)[0]


def audio(session, t0: float, t1: float, ffmpeg: Path) -> bytes:
    # Original rate and channel layout; lossless PCM observation, no synthetic TTS.
    with tempfile.TemporaryDirectory(prefix="witness-native-audio-") as directory:
        path = Path(directory) / "audio.wav"
        source_t0=t0+float(session.scene.truth.get("source_offset_s",0.))
        # Output seeking retains AAC priming samples; input -ss (even at zero)
        # can discard the first packet and silently shift the audio observation.
        run_decoder([str(ffmpeg), "-v", "error", "-threads", "1",
            "-i", str(session.scene.video_path), "-ss", f"{source_t0:.9f}", "-t", f"{t1-t0:.9f}", "-map", "0:a:0",
            "-vn", "-c:a", "pcm_s16le", "-threads", "1",
            "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1", "-y", str(path)], session)
        return path.read_bytes()
