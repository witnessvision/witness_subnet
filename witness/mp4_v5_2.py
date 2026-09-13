"""Production MP4 5.2: both hotkeys, task and complete bodies are authenticated."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Literal
import uuid

import httpx
from pydantic import Field

from witness.events import canonical_bytes, validate_events, MAX_RESPONSE_BYTES
from witness.mp4 import ClipTask as ExperimentalClipTask, MAX_CLIP_BYTES, create_app as serving_app

REQUEST_DOMAIN = b"witness-mp4-request-5.2\0POST\0/events\0"
RESPONSE_DOMAIN = b"witness-mp4-response-5.2\0POST\0/events\0"


class ClipTask(ExperimentalClipTask):
    transport_version: Literal["5.2"] = "5.2"
    netuid: int = Field(ge=0, le=65535)
    validator_hotkey: str = Field(min_length=47, max_length=49)
    miner_hotkey: str = Field(min_length=47, max_length=49)


def verify(hotkey, message, signature):
    from bittensor_wallet import Keypair
    if not Keypair(ss58_address=hotkey).verify(message, signature):
        raise ValueError("invalid_hotkey_signature")


def signed_headers(task, key):
    if key.ss58_address != task.validator_hotkey:
        raise ValueError("wrong_validator_signer")
    body = canonical_bytes(task.model_dump())
    return {"content-type": "video/mp4", "content-length": str(task.clip_bytes),
            "x-witness-task": base64.b64encode(body).decode(),
            "x-witness-signature": base64.b64encode(key.sign(REQUEST_DOMAIN + body)).decode()}


def response_message(task, body):
    # The entire canonical signed request is bound, including media hash and both identities.
    return RESPONSE_DOMAIN + hashlib.sha256(canonical_bytes(task.model_dump())).digest() + hashlib.sha256(body).digest()


def create_app(handler, *, hotkey, validator_hotkeys, netuid, state, internal_timeout=170.):
    def verify_task(body, signature):
        task = ClipTask.model_validate(json.loads(body))
        allowed = validator_hotkeys() if callable(validator_hotkeys) else validator_hotkeys
        if task.netuid != netuid or task.miner_hotkey != hotkey.ss58_address or task.validator_hotkey not in allowed:
            raise ValueError("wrong_request_identity")
        verify(task.validator_hotkey, REQUEST_DOMAIN + body, signature)
        return task

    def sign_response(task, body):
        return {"x-witness-response-signature": base64.b64encode(hotkey.sign(response_message(task, body))).decode()}

    return serving_app(handler, validator_key=None, state=state, internal_timeout=internal_timeout,
                       verify_task=verify_task, sign_response=sign_response)


async def send_clip(url, clip: Path, spec, key, *, miner_hotkey, netuid=20, task_id=None,
                    timeout=180., on_dispatch=None):
    if not 0 < timeout <= 180 or not 0 < clip.stat().st_size <= MAX_CLIP_BYTES:
        raise ValueError("invalid_request_limits")
    raw = clip.read_bytes()
    now = time.time()
    task = ClipTask(task_id=task_id or uuid.uuid4().hex, issued_at=now, expires_at=now+timeout,
                    clip_sha256=hashlib.sha256(raw).hexdigest(), clip_bytes=len(raw), task_spec=spec,
                    netuid=netuid, validator_hotkey=key.ss58_address, miner_hotkey=miner_hotkey)
    if on_dispatch is not None:
        on_dispatch(task)
    started = time.monotonic()
    async with asyncio.timeout(timeout) as deadline:
        async with httpx.AsyncClient(timeout=None, follow_redirects=False, trust_env=False) as client:
            async with client.stream("POST", url.rstrip("/")+"/events", content=raw,
                                     headers={**signed_headers(task, key), "accept-encoding": "identity"}) as response:
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ValueError("encoded_response_not_supported")
                if int(response.headers.get("content-length", "0")) > MAX_RESPONSE_BYTES:
                    raise ValueError("response_too_large")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ValueError("response_too_large")
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise TimeoutError("complete_response_arrived_late")
                deadline.reschedule(None)
                response.raise_for_status()
                if hashlib.sha256(body).hexdigest() != response.headers.get("x-witness-response-sha256"):
                    raise ValueError("response_hash_mismatch")
                signature = response.headers.get("x-witness-response-signature", "")
                if len(signature) > 128:
                    raise ValueError("invalid_response_signature")
                verify(miner_hotkey, response_message(task, body), base64.b64decode(signature, validate=True))
                def unique_object(pairs):
                    value = {}
                    for k, v in pairs:
                        if k in value:
                            raise ValueError("duplicate_response_field")
                        value[k] = v
                    return value
                result = json.loads(body, object_pairs_hook=unique_object)
                validate_events(result, spec.duration)
    return {"task": task.model_dump(), "response": result, "elapsed_s": elapsed,
            "response_sha256": hashlib.sha256(body).hexdigest(), "response_signature": signature}
