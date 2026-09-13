"""Signed full-clip HTTP transport, independent of chain/wallet machinery.

Transport 5.1 retains structured_events 5.0 and scorer 5.0.0. Implementations
receive a temporary MP4, never a source URL or reference annotations.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Awaitable, Callable, Literal
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import httpx
from pydantic import Field

from witness.events import EventsTaskSpec, StrictModel, canonical_bytes, validate_events, MAX_RESPONSE_BYTES

MAX_CLIP_BYTES = 128 * 1024 * 1024
DOMAIN = b"witness-mp4-request-5.1\0POST\0/events\0"


class ClipTask(StrictModel):
    transport_version: Literal["5.1"] = "5.1"
    task_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    issued_at: float
    expires_at: float
    clip_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip_bytes: int = Field(gt=0, le=MAX_CLIP_BYTES)
    task_spec: EventsTaskSpec


def signed_headers(task: ClipTask, key: Ed25519PrivateKey) -> dict[str, str]:
    body = canonical_bytes(task.model_dump())
    return {"content-type": "video/mp4", "content-length": str(task.clip_bytes),
            "x-witness-task": base64.b64encode(body).decode(),
            "x-witness-signature": base64.b64encode(key.sign(DOMAIN + body)).decode()}


class ReplayStore:
    """Consume each signed task once, including failed tasks, across restarts."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, expires REAL NOT NULL)")
        path.chmod(0o600)

    def claim(self, task: ClipTask) -> bool:
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM tasks WHERE expires < ?", (time.time()-60,))
            try:
                db.execute("INSERT INTO tasks VALUES (?, ?)", (task.task_id, task.expires_at))
            except sqlite3.IntegrityError:
                return False
        return True


def create_app(handler: Callable[[Path, ClipTask], Awaitable[dict]], *,
               validator_key: Ed25519PublicKey | None, state: Path, internal_timeout: float = 170.,
               verify_task=None, sign_response=None) -> FastAPI:
    """Empty model-independent serving adapter. Bind one trusted validator key."""
    if not 0 < internal_timeout <= 170:
        raise ValueError("invalid_internal_timeout")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    replay, lock = ReplayStore(state / "requests.sqlite3"), asyncio.Lock()

    @app.post("/events")
    async def events(request: Request):
        started = time.monotonic()
        try:
            encoded = request.headers.get("x-witness-task", "")
            signature = request.headers.get("x-witness-signature", "")
            if len(encoded) > 8192 or len(signature) > 128:
                raise ValueError()
            body = base64.b64decode(encoded, validate=True)
            if verify_task is None:
                validator_key.verify(base64.b64decode(signature, validate=True), DOMAIN + body)
                task = ClipTask.model_validate(json.loads(body))
            else:
                task = verify_task(body, base64.b64decode(signature, validate=True))
            if canonical_bytes(task.model_dump()) != body:
                raise ValueError()
            now = time.time()
            if not now-180 <= task.issued_at <= now+10 or not now < task.expires_at <= task.issued_at+180:
                raise ValueError()
            if request.headers.get("content-type") != "video/mp4":
                raise ValueError()
            if int(request.headers.get("content-length", "0")) != task.clip_bytes:
                raise ValueError()
            if request.headers.get("content-encoding", "identity") != "identity":
                raise ValueError()
        except Exception:
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        if lock.locked():
            return JSONResponse({"error": "busy"}, status_code=429)
        async with lock:
            if not replay.claim(task):
                return JSONResponse({"error": "duplicate_task"}, status_code=409)
            worker = disconnected = None
            try:
                remaining = min(internal_timeout-(time.monotonic()-started), task.expires_at-time.time())
                async with asyncio.timeout(remaining):
                    with tempfile.TemporaryDirectory(prefix="clip-", dir=state) as directory:
                        clip = Path(directory) / "video.mp4"
                        digest, total = hashlib.sha256(), 0
                        with clip.open("xb") as stream:
                            clip.chmod(0o600)
                            async for chunk in request.stream():
                                total += len(chunk)
                                if total > task.clip_bytes:
                                    raise ValueError("invalid_clip")
                                digest.update(chunk)
                                stream.write(chunk)
                        if total != task.clip_bytes or digest.hexdigest() != task.clip_sha256:
                            raise ValueError("invalid_clip")

                        async def disconnect():
                            while True:
                                message = await request.receive()
                                if message["type"] == "http.disconnect":
                                    return
                                await asyncio.sleep(.01)

                        worker = asyncio.create_task(handler(clip, task))
                        disconnected = asyncio.create_task(disconnect())
                        done, _ = await asyncio.wait([worker, disconnected], return_when=asyncio.FIRST_COMPLETED)
                        if disconnected in done:
                            raise ConnectionError()
                        result = await worker
                        validate_events(result, task.task_spec.duration)
                        response = canonical_bytes(result)
                        headers = {"x-witness-response-sha256": hashlib.sha256(response).hexdigest()}
                        if sign_response is not None:
                            headers.update(sign_response(task, response))
                        return Response(response, media_type="application/json", headers=headers)
            except TimeoutError:
                return JSONResponse({"error": "deadline_exceeded"}, status_code=504)
            except ConnectionError:
                return JSONResponse({"error": "disconnected"}, status_code=499)
            except Exception:
                return JSONResponse({"error": "task_failed"}, status_code=422)
            finally:
                pending = [t for t in (worker, disconnected) if t is not None]
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    return app


async def send_clip(url: str, clip: Path, spec: EventsTaskSpec, key: Ed25519PrivateKey, *, timeout=180.,
                    on_dispatch: Callable[[ClipTask], None] | None = None) -> dict:
    """No retry. Deadline includes upload and receipt of the complete response."""
    if not 0 < clip.stat().st_size <= MAX_CLIP_BYTES:
        raise ValueError("clip_too_large_or_empty")
    raw = clip.read_bytes()
    now = time.time()
    task = ClipTask(task_id=uuid.uuid4().hex, issued_at=now, expires_at=now+180,
                    clip_sha256=hashlib.sha256(raw).hexdigest(), clip_bytes=len(raw), task_spec=spec)
    if on_dispatch is not None:
        on_dispatch(task)
    started = time.monotonic()
    async with asyncio.timeout(min(timeout, 180.)) as deadline:
        async with httpx.AsyncClient(timeout=None, follow_redirects=False, trust_env=False) as client:
            async with client.stream("POST", url.rstrip("/")+"/events", content=raw,
                                     headers={**signed_headers(task, key),"accept-encoding":"identity"}) as response:
                if response.headers.get('content-encoding','identity') != 'identity':
                    raise ValueError('encoded_response_not_supported')
                if int(response.headers.get('content-length','0')) > MAX_RESPONSE_BYTES:
                    raise ValueError('response_too_large')
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError("response_too_large")
                elapsed_s=time.monotonic()-started
                if elapsed_s >= min(timeout,180.):
                    raise TimeoutError('complete_response_arrived_late')
                # Validation is validator work. Stop the external miner clock
                # when the body completes, before parsing/validating its fields.
                deadline.reschedule(None)
                response.raise_for_status()
                if hashlib.sha256(body).hexdigest() != response.headers.get("x-witness-response-sha256"):
                    raise ValueError("response_hash_mismatch")
                def unique_object(pairs):
                    value = {}
                    for k, v in pairs:
                        if k in value:
                            raise ValueError("duplicate_response_field")
                        value[k] = v
                    return value
                result = json.loads(body, object_pairs_hook=unique_object)
                validate_events(result, spec.duration)
    return {"task": task.model_dump(), "response": result, "elapsed_s": elapsed_s,
            "response_sha256": hashlib.sha256(body).hexdigest()}
