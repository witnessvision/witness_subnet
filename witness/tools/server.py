"""FastAPI server exposing metered, deliberately lossy observations of videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
import random
import subprocess
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from .metering import Cost, visual_token_cost
from .transcripts import load_observed_transcript


DEFAULT_FFMPEG = (
    os.environ.get("WITNESS_FFMPEG")
    or (str(Path.home() / "bin" / "ffmpeg") if (Path.home() / "bin" / "ffmpeg").is_file() else "")
    or shutil.which("ffmpeg")
    or "/usr/bin/ffmpeg"
)
COST_HEADER = "X-Witness-Cost"
# Each active miner may issue concurrent HTTP requests. Bound decoder processes
# independently of HTTP workers, and keep their native thread pools small.
_FFMPEG_SLOTS = threading.BoundedSemaphore(4)


class Budget(BaseModel):
    """Independent ceilings for each observation channel."""

    visual_tokens: int = Field(ge=0)
    audio_seconds: float = Field(ge=0)
    transcript_chars: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def accept_max_prefixes(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {
                "visual_tokens": int(value),
                "audio_seconds": float(value),
                "transcript_chars": int(value),
            }
        if isinstance(value, dict):
            value = dict(value)
            for name in ("visual_tokens", "audio_seconds", "transcript_chars"):
                alias = f"max_{name}"
                if name not in value and alias in value:
                    value[name] = value.pop(alias)
        return value

    def as_dict(self) -> dict[str, int | float]:
        return self.model_dump()


class SessionRequest(BaseModel):
    scene_id: str
    budget: Budget


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    directory: Path
    video_path: Path
    truth: dict[str, Any]

    @property
    def duration(self) -> float:
        return float(self.truth["duration"])


@dataclass(slots=True)
class Session:
    session_id: str
    scene: Scene
    budget: Budget
    log_path: Path
    cost: Cost = field(default_factory=Cost)


class BudgetExceeded(Exception):
    def __init__(self, session: Session, requested: Cost):
        self.session = session
        self.requested = requested


class SessionClosed(Exception):
    """The validator has finalized this task's cost."""


class SessionStore:
    def __init__(self, scenes: dict[str, Scene], log_dir: Path):
        self.scenes = scenes
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()

    def create(self, scene_id: str, budget: Budget) -> Session:
        scene = self.scenes.get(scene_id)
        if scene is None:
            raise KeyError(scene_id)
        session_id = uuid.uuid4().hex
        session = Session(session_id, scene, budget, self.log_dir / f"{session_id}.jsonl")
        with self.lock:
            self.sessions[session_id] = session
            self._append(
                session,
                endpoint="POST /session",
                params={"scene_id": scene_id, "budget": budget.as_dict()},
                delta=Cost(),
                status=200,
            )
        return session

    def get(self, session_id: str) -> Session:
        with self.lock:
            try:
                return self.sessions[session_id]
            except KeyError as exc:
                raise KeyError(session_id) from exc

    def charge(self, session: Session, endpoint: str, params: dict[str, Any], delta: Cost) -> None:
        with self.lock:
            if self.sessions.get(session.session_id) is not session:
                raise SessionClosed()
            projected = session.cost + delta
            budget = session.budget
            if (
                projected.visual_tokens > budget.visual_tokens
                or projected.audio_seconds > budget.audio_seconds + 1e-9
                or projected.transcript_chars > budget.transcript_chars
            ):
                self._append(session, endpoint, params, delta, 429, charged=False)
                raise BudgetExceeded(session, delta)
            session.cost = projected

    def finish(
        self,
        session: Session,
        endpoint: str,
        params: dict[str, Any],
        delta: Cost,
        status: int = 200,
    ) -> None:
        with self.lock:
            self._append(session, endpoint, params, delta, status, charged=True)

    def rollback(
        self, session: Session, endpoint: str, params: dict[str, Any], delta: Cost
    ) -> None:
        with self.lock:
            session.cost = session.cost - delta
            self._append(session, endpoint, params, delta, 500, charged=False)

    def record_free(self, session: Session, endpoint: str, params: dict[str, Any]) -> None:
        with self.lock:
            self._append(session, endpoint, params, Cost(), 200, charged=True)

    def _append(
        self,
        session: Session,
        endpoint: str,
        params: dict[str, Any],
        delta: Cost,
        status: int,
        charged: bool = True,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session.session_id,
            "scene_id": session.scene.scene_id,
            "endpoint": endpoint,
            "params": params,
            "status": status,
            "charged": charged,
            "call_cost": delta.as_dict(),
            "running_cost": session.cost.as_dict(),
        }
        with session.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def discover_scenes(roots: Iterable[str | Path] | str | Path) -> dict[str, Scene]:
    if isinstance(roots, (str, Path)):
        roots = [roots]
    scenes: dict[str, Scene] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        candidates = [root] if (root / "scene.json").is_file() else sorted(root.glob("*/"))
        for directory in candidates:
            scene_path = directory / "scene.json"
            video_path = directory / "video.mp4"
            if not scene_path.is_file() or not video_path.is_file():
                continue
            truth = json.loads(scene_path.read_text(encoding="utf-8"))
            scene_id = directory.name
            if scene_id in scenes and scenes[scene_id].directory != directory:
                raise ValueError(f"duplicate scene id: {scene_id}")
            scenes[scene_id] = Scene(scene_id, directory, video_path, truth)
    if not scenes:
        raise ValueError("no directories containing scene.json and video.mp4 were found")
    return scenes


def _parse_resolution(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x", maxsplit=1)
    if len(parts) == 1:
        parts *= 2
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("res must be WIDTHxHEIGHT, for example 320x180") from exc
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError("resolution dimensions must be between 1 and 4096")
    return width, height


def _validate_interval(t0: float, t1: float, duration: float) -> None:
    if not (0 <= t0 < t1 <= duration):
        raise ValueError(f"expected 0 <= t0 < t1 <= {duration}")


def _cost_headers(cost: Cost) -> dict[str, str]:
    return {COST_HEADER: json.dumps(cost.as_dict(), separators=(",", ":"))}


def _run_ffmpeg(command: list[str]) -> bytes:
    with _FFMPEG_SLOTS:
        result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError(message[-1] if message else "ffmpeg failed")
    return result.stdout


def _extract_frame(video: Path, t: float, width: int, height: int, ffmpeg: Path) -> bytes:
    return _run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads", "1",
            "-filter_threads", "1",
            "-ss",
            f"{t:.9f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:{height}:flags=lanczos",
            "-q:v",
            "2",
            "-threads", "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
    )


def _extract_frames(video: Path, timestamps: list[float], width: int, height: int,
                    source_fps: float, ffmpeg: Path) -> list[bytes]:
    """Decode CFR scene frames in bounded batches, preserving seek-to-frame timing."""
    metadata = json.loads(_run_ffmpeg([str(ffmpeg.with_name('ffprobe')), '-v', 'error',
        '-select_streams', 'v:0', '-show_entries',
        'stream=time_base,start_time,r_frame_rate,avg_frame_rate', '-of', 'json', str(video)]))
    stream = metadata['streams'][0]
    fps = Fraction(str(source_fps))
    if (Fraction(stream['r_frame_rate']) != fps or Fraction(stream['avg_frame_rate']) != fps
            or float(stream.get('start_time', 0)) != 0):
        raise ValueError('grounded batch requires a zero-origin constant-rate video')
    time_base = Fraction(stream['time_base'])
    # -ss parses microseconds, then rounds to the stream time base. Preserve that
    # boundary behavior even when a requested time lies just after a frame PTS.
    def frame_number(timestamp: float) -> int:
        microseconds = int(Fraction(f'{timestamp:.9f}') * 1_000_000)
        ticks = math.floor(Fraction(microseconds, 1_000_000) / time_base + Fraction(1, 2))
        return math.ceil(ticks * time_base * fps)
    numbers = [frame_number(timestamp) for timestamp in timestamps]
    unique = sorted(set(numbers))
    images: dict[int, bytes] = {}

    def expression(items: list[int]) -> str:
        if len(items) == 1:
            return f'eq(n\\,{items[0]})'
        middle = len(items) // 2
        return f'({expression(items[:middle])}+{expression(items[middle:])})'

    for start in range(0, len(unique), 512):
        selected = unique[start:start + 512]
        with tempfile.TemporaryDirectory(prefix='witness-frames-') as directory:
            pattern = str(Path(directory) / '%06d.jpg')
            _run_ffmpeg([str(ffmpeg), '-hide_banner', '-loglevel', 'error',
                '-threads', '1', '-filter_threads', '1',
                '-i', str(video), '-frames:v', str(len(selected)),
                '-vf', f'select={expression(selected)},scale={width}:{height}:flags=lanczos',
                '-fps_mode', 'passthrough', '-q:v', '2', '-threads', '1',
                '-vcodec', 'mjpeg', pattern])
            paths = sorted(Path(directory).glob('*.jpg'))
            if len(paths) != len(selected):
                raise RuntimeError('frame batch did not decode every requested timestamp')
            images.update((number, path.read_bytes()) for number, path in zip(selected, paths))
    return [images[number] for number in numbers]


def _extract_audio(
    video: Path,
    t0: float,
    t1: float,
    sample_rate: int,
    channels: int,
    ffmpeg: Path,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="witness-audio-") as temp_dir:
        output = Path(temp_dir) / "audio.wav"
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads", "1",
            "-filter_threads", "1",
            "-ss",
            f"{t0:.9f}",
            "-i",
            str(video),
            "-t",
            f"{t1 - t0:.9f}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-threads", "1",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-y",
            str(output),
        ]
        _run_ffmpeg(command)
        return output.read_bytes()


def degraded_transcript(
    scene: Scene,
    word_error_rate: float,
    seed: int,
    timestamp_quantum: float,
) -> list[dict[str, Any]]:
    """Build a stable ASR-like hint without exposing canonical timestamps."""
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(scene.truth.get("dialogue", [])):
        identity = f"{scene.truth.get('seed')}:{seed}:{index}:{item.get('text', '')}"
        local_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
        rng = random.Random(local_seed)
        words = str(item.get("text", "")).split()
        noisy = ["<unk>" if rng.random() < word_error_rate else word for word in words]
        start = round(float(item["start"]) / timestamp_quantum) * timestamp_quantum
        end = round(float(item["end"]) / timestamp_quantum) * timestamp_quantum
        entries.append(
            {
                "speaker": item.get("speaker"),
                "start": round(max(0.0, start), 6),
                "end": round(min(scene.duration, max(start, end)), 6),
                "text": " ".join(noisy),
            }
        )
    return entries


def create_app(
    scene_roots: Iterable[str | Path] | str | Path,
    *,
    log_dir: str | Path = "data/sessions",
    ffmpeg_path: str | Path = DEFAULT_FFMPEG,
    transcript_wer: float = 0.2,
    transcript_seed: int = 0,
    timestamp_quantum: float = 1.0,
    allow_session_creation: bool = True,
    transcript_source: str = "legacy_labels",
) -> FastAPI:
    if transcript_source not in {"legacy_labels", "asr", "none"}:
        raise ValueError("unknown transcript_source")
    if not 0 <= transcript_wer <= 1:
        raise ValueError("transcript_wer must be between 0 and 1")
    if timestamp_quantum <= 0:
        raise ValueError("timestamp_quantum must be positive")
    ffmpeg = Path(ffmpeg_path).expanduser().resolve()
    if not ffmpeg.is_file():
        raise ValueError(f"ffmpeg not found: {ffmpeg}")

    store = SessionStore(discover_scenes(scene_roots), Path(log_dir))
    observed = {
        scene_id: load_observed_transcript(scene.directory, scene.duration)
        for scene_id, scene in store.scenes.items()
    } if transcript_source == "asr" else {}
    app = FastAPI(title="Witness metered tool server", version="1.0")
    app.state.store = store

    @app.exception_handler(SessionClosed)
    async def session_closed_handler(_request: Request, _exc: SessionClosed) -> JSONResponse:
        return JSONResponse({"detail": "task session is closed"}, status_code=410)

    @app.exception_handler(BudgetExceeded)
    async def budget_exceeded_handler(_request: Request, exc: BudgetExceeded) -> JSONResponse:
        body = {
            "detail": "budget exhausted",
            "requested_cost": exc.requested.as_dict(),
            "cost": exc.session.cost.as_dict(),
            "budget": exc.session.budget.as_dict(),
        }
        return JSONResponse(body, status_code=429, headers=_cost_headers(exc.session.cost))

    def session_or_404(session_id: str) -> Session:
        try:
            return store.get(session_id)
        except KeyError:
            raise_api(404, "unknown session_id")

    def raise_api(status: int, detail: str) -> None:
        from fastapi import HTTPException

        raise HTTPException(status_code=status, detail=detail)

    def interval_or_400(t0: float, t1: float, duration: float) -> None:
        try:
            _validate_interval(t0, t1, duration)
        except ValueError as exc:
            raise_api(400, str(exc))

    def resolution_or_400(res: str) -> tuple[int, int]:
        try:
            return _parse_resolution(res)
        except ValueError as exc:
            raise_api(400, str(exc))

    @app.post("/session")
    def new_session(request: SessionRequest) -> JSONResponse:
        if not allow_session_creation:
            raise_api(403, "sessions are issued by the validator")
        try:
            session = store.create(request.scene_id, request.budget)
        except KeyError:
            raise_api(404, "unknown scene_id")
        body = {
            "session_id": session.session_id,
            "budget": session.budget.as_dict(),
            "cost": session.cost.as_dict(),
        }
        return JSONResponse(body, headers=_cost_headers(session.cost))

    @app.get("/s/{session_id}/meta")
    def get_meta(session_id: str) -> JSONResponse:
        session = session_or_404(session_id)
        store.record_free(session, "GET /meta", {})
        body = {"duration": session.scene.duration, "cost": session.cost.as_dict()}
        return JSONResponse(body, headers=_cost_headers(session.cost))

    @app.get("/s/{session_id}/frame")
    def get_frame(session_id: str, t: float, res: str = "640x360") -> Response:
        session = session_or_404(session_id)
        if not 0 <= t < session.scene.duration:
            raise_api(400, f"expected 0 <= t < {session.scene.duration}")
        width, height = resolution_or_400(res)
        params = {"t": t, "res": f"{width}x{height}"}
        delta = Cost(visual_tokens=visual_token_cost(width, height))
        store.charge(session, "GET /frame", params, delta)
        try:
            last_frame_t = max(
                0.0,
                session.scene.duration - 1.0 / float(session.scene.truth["fps"]),
            )
            jpeg = _extract_frame(
                session.scene.video_path, min(t, last_frame_t), width, height, ffmpeg
            )
        except Exception:
            store.rollback(session, "GET /frame", params, delta)
            raise
        store.finish(session, "GET /frame", params, delta)
        return Response(jpeg, media_type="image/jpeg", headers=_cost_headers(session.cost))

    @app.get("/s/{session_id}/frames")
    def get_frames(
        session_id: str,
        t0: float,
        t1: float,
        fps: float = Query(gt=0, le=120),
        res: str = "640x360",
    ) -> Response:
        session = session_or_404(session_id)
        interval_or_400(t0, t1, session.scene.duration)
        width, height = resolution_or_400(res)
        frame_count = int((t1 - t0) * fps - 1e-12) + 1
        timestamps = [t0 + index / fps for index in range(frame_count)]
        timestamps = [timestamp for timestamp in timestamps if timestamp < t1 - 1e-12]
        last_frame_t = max(
            0.0,
            session.scene.duration - 1.0 / float(session.scene.truth["fps"]),
        )
        timestamps = list(
            dict.fromkeys(round(min(timestamp, last_frame_t), 9) for timestamp in timestamps)
        )
        params = {"t0": t0, "t1": t1, "fps": fps, "res": f"{width}x{height}"}
        delta = Cost(visual_tokens=visual_token_cost(width, height, len(timestamps)))
        store.charge(session, "GET /frames", params, delta)
        try:
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
                images = (_extract_frames(session.scene.video_path, timestamps, width, height,
                                          float(session.scene.truth['fps']), ffmpeg)
                          if session.scene.truth.get('schema_version') == '3.0' else
                          [_extract_frame(session.scene.video_path, timestamp, width, height, ffmpeg)
                           for timestamp in timestamps])
                for index, jpeg in enumerate(images):
                    archive.writestr(f"frame_{index:06d}.jpg", jpeg)
                manifest = {
                    "timestamps": [round(value, 9) for value in timestamps],
                    "resolution": [width, height],
                    "cost": (session.cost).as_dict(),
                    "call_cost": delta.as_dict(),
                }
                archive.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        except Exception:
            store.rollback(session, "GET /frames", params, delta)
            raise
        store.finish(session, "GET /frames", params, delta)
        return Response(
            output.getvalue(), media_type="application/zip", headers=_cost_headers(session.cost)
        )

    @app.get("/s/{session_id}/audio")
    def get_audio(session_id: str, t0: float, t1: float) -> Response:
        session = session_or_404(session_id)
        interval_or_400(t0, t1, session.scene.duration)
        params = {"t0": t0, "t1": t1}
        delta = Cost(audio_seconds=t1 - t0)
        store.charge(session, "GET /audio", params, delta)
        audio_config = session.scene.truth.get("audio", {})
        try:
            wav = _extract_audio(
                session.scene.video_path,
                t0,
                t1,
                int(audio_config.get("sample_rate", 22050)),
                int(audio_config.get("channels", 1)),
                ffmpeg,
            )
        except Exception:
            store.rollback(session, "GET /audio", params, delta)
            raise
        store.finish(session, "GET /audio", params, delta)
        return Response(wav, media_type="audio/wav", headers=_cost_headers(session.cost))

    def transcript_window(session: Session, t0: float, t1: float) -> list[dict[str, Any]]:
        transcript = session_transcript(session)
        return [entry for entry in transcript if entry["start"] < t1 and entry["end"] > t0]

    def session_transcript(session: Session) -> list[dict[str, Any]]:
        if transcript_source == "none":
            return []
        if transcript_source == "asr":
            return observed[session.scene.scene_id]
        return degraded_transcript(session.scene, transcript_wer, transcript_seed, timestamp_quantum)

    @app.get("/s/{session_id}/transcript")
    def get_transcript(session_id: str, t0: float, t1: float) -> JSONResponse:
        session = session_or_404(session_id)
        interval_or_400(t0, t1, session.scene.duration)
        entries = transcript_window(session, t0, t1)
        delta = Cost(transcript_chars=sum(len(entry["text"]) for entry in entries))
        params = {"t0": t0, "t1": t1}
        store.charge(session, "GET /transcript", params, delta)
        store.finish(session, "GET /transcript", params, delta)
        body = {"entries": entries, "cost": session.cost.as_dict()}
        return JSONResponse(body, headers=_cost_headers(session.cost))

    @app.get("/s/{session_id}/search_transcript")
    def search_transcript(session_id: str, q: str = Query(min_length=1)) -> JSONResponse:
        session = session_or_404(session_id)
        transcript = session_transcript(session)
        matches = [entry for entry in transcript if q.casefold() in entry["text"].casefold()]
        delta = Cost(transcript_chars=sum(len(entry["text"]) for entry in matches))
        params = {"q": q}
        store.charge(session, "GET /search_transcript", params, delta)
        store.finish(session, "GET /search_transcript", params, delta)
        body = {"entries": matches, "cost": session.cost.as_dict()}
        return JSONResponse(body, headers=_cost_headers(session.cost))

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Witness metered tool server")
    parser.add_argument("--scenes", action="append", required=True, help="scene directory or parent")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log-dir", default="data/sessions")
    parser.add_argument("--ffmpeg", default=str(DEFAULT_FFMPEG))
    parser.add_argument("--transcript-wer", type=float, default=0.2)
    parser.add_argument("--transcript-source", choices=("legacy_labels", "asr", "none"), default="legacy_labels")
    parser.add_argument("--transcript-seed", type=int, default=0)
    parser.add_argument("--timestamp-quantum", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    app = create_app(
        args.scenes,
        log_dir=args.log_dir,
        ffmpeg_path=args.ffmpeg,
        transcript_wer=args.transcript_wer,
        transcript_seed=args.transcript_seed,
        timestamp_quantum=args.timestamp_quantum,
        transcript_source=args.transcript_source,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
