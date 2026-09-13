"""v5: sign all application request fields, bound the entire response body."""
from __future__ import annotations

import asyncio
import hashlib
import json
import aiohttp

from witness.events import MAX_RESPONSE_BYTES, canonical_bytes, validate_events
from .protocol import WitnessTask

# Bound the full wire envelope as well as the parsed complete miner response.
MAX_WIRE_BYTES = MAX_RESPONSE_BYTES


class ResponseRejected(ValueError):
    pass


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ResponseRejected("duplicate_json_key")
        value[key] = item
    return value


def parse_response(content: bytes) -> WitnessTask:
    if len(content) > MAX_WIRE_BYTES:
        raise ResponseRejected("response_too_large")
    try:
        return WitnessTask.model_validate(json.loads(content, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json"))))
    except (ValueError, TypeError, RecursionError) as exc:
        raise ResponseRejected("invalid_or_incomplete_response") from exc


def validate_response(request: WitnessTask, response: WitnessTask) -> None:
    if response.body_hash != request.body_hash:
        raise ResponseRejected("response_task_mismatch")
    if len(canonical_bytes({"reconstruction": response.reconstruction,
                            "trace_summary": response.trace_summary})) > MAX_RESPONSE_BYTES:
        raise ResponseRejected("response_too_large")
    trace = response.trace_summary
    if not isinstance(trace, dict) or set(trace) != {"status"}:
        raise ResponseRejected("invalid_trace")
    if trace["status"] != "ok":
        allowed = {"busy", "deadline_exceeded", "error", "invalid_response", "response_too_large"}
        raise ResponseRejected(trace["status"] if trace["status"] in allowed else "invalid_response")
    try:
        validate_events(response.reconstruction, request.task_spec["duration"])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ResponseRejected("invalid_response") from exc


async def query_bounded(dendrite, endpoint, task: WitnessTask, timeout: float):
    # The outer clock includes preprocessing, connect, headers, streaming body,
    # parsing and validation. There are no retries or redirects.
    async with asyncio.timeout(timeout):
        task._transport_evidence = {"send_attempted": False, "send_confirmed": False}
        request = dendrite.preprocess_synapse_for_request(endpoint.axon, task.model_copy(deep=True), timeout)
        url = dendrite._get_endpoint_url(endpoint.axon, request_name="WitnessTask")
        body = canonical_bytes(request.model_dump())
        task._transport_evidence.update(request_body_sha256=hashlib.sha256(body).hexdigest(),
                                        signed_request_hash=request.body_hash,
                                        send_attempted=True, send_confirmed=None)
        async with (await dendrite.session).post(url=url, headers={**request.to_headers(), "Content-Type": "application/json"},
                data=body,
                timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=False) as response:
            task._transport_evidence["send_confirmed"] = True
            if response.status != 200:
                raise ResponseRejected(f"http_{response.status}")
            content = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                if len(content) + len(chunk) > MAX_WIRE_BYTES:
                    raise ResponseRejected("response_too_large")
                content.extend(chunk)
            task._transport_evidence["wire_response_sha256"] = hashlib.sha256(content).hexdigest()
            task._transport_evidence["wire_response_bytes"] = len(content)
            result = parse_response(content)
            validate_response(request, result)
            dendrite.process_server_response(response, result.model_dump(), request)
            return request
