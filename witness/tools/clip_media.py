"""Produce a standalone, anonymous MP4 with native rate and zero-based time."""
from pathlib import Path
import json
from witness.subnet.processes import run_process
from witness.tools.native_media import inspect_original


async def crop_mp4(source: Path, destination: Path, *, start: float, duration: float) -> dict:
    if start < 0 or not 60 <= duration <= 120 or destination.exists():
        raise ValueError("invalid_clip_request")
    native = inspect_original(source, Path("ffprobe"))
    timing = json.loads(await run_process(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=time_base", "-of", "json", str(source)], timeout=30))
    time_base = timing["streams"][0]["time_base"]
    if start+duration > native["duration"]+.05:
        raise ValueError("clip_outside_source")
    try:
        await run_process(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
            "-ss", f"{start:.9f}", "-i", str(source), "-t", f"{duration:.9f}",
            "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "-1", "-map_chapters", "-1",
            "-vf", "setpts=PTS-STARTPTS", "-af", "asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-fps_mode", "passthrough",
            "-enc_time_base:v", time_base,
            "-c:a", "alac", "-threads", "1", "-movflags", "+faststart", str(destination)], timeout=180)
        destination.chmod(0o600)
        measured = inspect_original(destination, Path("ffprobe"))
        if abs(measured["duration"]-duration) > max(.1, 2/native["fps"]):
            raise ValueError("clip_duration_mismatch")
        # avg_frame_rate describes this interval, not a fixed playback speed:
        # a legitimate VFR crop can have a different average from its source.
        # Passthrough retains presentation times; no fps/rate filter is applied.
        if (not 60 <= measured["duration"] <= 120 or measured["audio"] != native["audio"]
                or measured["has_audio"] != native["has_audio"]):
            raise ValueError("clip_changed_native_rate")
        return measured
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
