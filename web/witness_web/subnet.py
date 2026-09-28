"""Read-only HTTP adapter to the subnet-owned display projection."""
import json
import os
import sqlite3
from pathlib import Path

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from witness.benchmark.dashboard import Projection


def add_subnet_routes(app, sources=None):
    if sources is None:
        sources = json.loads(os.environ.get('WITNESS_VALIDATOR_SOURCES', '[]'))
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise ValueError('WITNESS_VALIDATOR_SOURCES_must_be_a_JSON_string_array')
    view = Projection(sources)
    app.state.subnet_projection = view

    def read(call):
        try:
            return call()
        except (ValueError, KeyError):
            raise HTTPException(400, 'Invalid request or evidence') from None
        except (OSError, StopIteration, sqlite3.Error, httpx.HTTPError):
            raise HTTPException(503, 'Evaluator evidence unavailable') from None

    @app.get('/api/subnet')
    def state():
        return view.state()

    @app.get('/api/hotkeys')
    def hotkeys(validator: str, state: str = 'consumed', q: str = '', page: int = 1):
        return read(lambda: view.hotkeys(validator, state, q, page))

    @app.get('/api/evaluations/{validator}/{window}/{model}')
    def evaluation(validator: str, window: int, model: str):
        return read(lambda: view.evaluation(validator, window, model))

    @app.get('/api/media/{validator}/{window}/{digest}.mp4')
    def media(validator: str, window: int, digest: str, request: Request):
        target = read(lambda: view.media(validator, window, digest))
        if isinstance(target, Path):
            return FileResponse(target, media_type='video/mp4')
        client = httpx.Client(timeout=30)
        try:
            response = client.send(client.build_request('GET', target,
                                   headers={'Range': request.headers.get('Range', 'bytes=0-')}), stream=True)
            response.raise_for_status()
        except Exception:
            client.close()
            raise HTTPException(503, 'Evaluator media unavailable') from None
        def chunks():
            size = 0
            try:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 256 * 1024 * 1024:
                        break
                    yield chunk
            finally:
                response.close()
                client.close()
        headers = {key: response.headers[key] for key in ('content-length', 'content-range', 'accept-ranges')
                   if key in response.headers}
        return StreamingResponse(chunks(), status_code=response.status_code, headers=headers, media_type='video/mp4')
