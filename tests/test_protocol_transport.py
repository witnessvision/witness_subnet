import asyncio
import json
from types import SimpleNamespace

import bittensor as bt
import pytest
from pydantic import ValidationError
from starlette.requests import Request

from test_subnet import _task
from witness.subnet.protocol import WitnessTask


def request(payload, headers):
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(payload).encode()}
    return Request({'type':'http','method':'POST','path':'/WitnessTask',
                    'headers':[(k.lower().encode(),v.encode()) for k,v in headers.items()]},receive)


def test_sdk_header_placeholders_accept_metadata_but_full_body_remains_strict():
    task = _task()
    headers = task.to_headers()
    parsed = WitnessTask.from_headers(headers)
    assert isinstance(parsed, WitnessTask)
    assert parsed.computed_body_hash == task.body_hash
    assert parsed.axon is not None and parsed.dendrite is not None
    with pytest.raises(ValidationError):
        WitnessTask.model_validate(parsed.model_dump())
    axon = SimpleNamespace(forward_class_types={'WitnessTask':WitnessTask})
    body = asyncio.run(bt.Axon.verify_body_integrity(axon, request(task.model_dump(),headers)))
    assert body['task_id'] == task.task_id
    body['task_id'] = 'tampered'
    with pytest.raises(ValueError, match='Hash mismatch'):
        asyncio.run(bt.Axon.verify_body_integrity(axon, request(body,headers)))


def test_hash_covers_task_but_not_response_and_is_independent_of_dict_order():
    task = _task()
    digest = task.body_hash
    task.budget = dict(reversed(list(task.budget.items())))
    task.task_spec = dict(reversed(list(task.task_spec.items())))
    assert task.body_hash == digest
    task.reconstruction = {'qa': {'q1':'another answer'}}
    assert task.body_hash == digest
    task.deadline_s = 31
    assert task.body_hash != digest
