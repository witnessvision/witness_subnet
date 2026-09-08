"""Small standard-library client for the Witness metered tool API."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Any

COST_HEADER = "X-Witness-Cost"
class ToolAPIError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"Witness tool API returned HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class BudgetExhausted(ToolAPIError):
    pass


@dataclass(frozen=True, slots=True)
class ToolResult:
    data: Any
    cost: dict[str, int | float]


class WitnessClient:
    def __init__(self, base_url: str, session_id: str | None = None, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout

    def create_session(self, scene_id: str, budget: dict[str, int | float] | float) -> str:
        result = self._request_json(
            "POST", "/session", {"scene_id": scene_id, "budget": budget}
        ).data
        self.session_id = result["session_id"]
        return self.session_id

    def get_meta(self) -> ToolResult:
        return self._request_json("GET", self._path("meta"))

    def get_frame(self, t: float, res: str | tuple[int, int] = "640x360") -> ToolResult:
        return self._request_binary("frame", {"t": t, "res": self._res(res)})

    def get_frames(
        self,
        t0: float,
        t1: float,
        fps: float,
        res: str | tuple[int, int] = "640x360",
    ) -> ToolResult:
        bundle = self._request_binary(
            "frames", {"t0": t0, "t1": t1, "fps": fps, "res": self._res(res)}
        )
        with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
            names = sorted(name for name in archive.namelist() if name.endswith(".jpg"))
            frames = [archive.read(name) for name in names]
            manifest = json.loads(archive.read("manifest.json"))
        return ToolResult({"frames": frames, "manifest": manifest}, bundle.cost)

    def get_audio(self, t0: float, t1: float) -> ToolResult:
        return self._request_binary("audio", {"t0": t0, "t1": t1})

    def get_transcript(self, t0: float, t1: float) -> ToolResult:
        return self._request_json("GET", self._path("transcript"), query={"t0": t0, "t1": t1})

    def search_transcript(self, q: str) -> ToolResult:
        return self._request_json("GET", self._path("search_transcript"), query={"q": q})

    def _path(self, endpoint: str) -> str:
        if not self.session_id:
            raise RuntimeError("create_session must be called first or session_id supplied")
        return f"/s/{self.session_id}/{endpoint}"

    @staticmethod
    def _res(value: str | tuple[int, int]) -> str:
        return f"{value[0]}x{value[1]}" if isinstance(value, tuple) else value

    def _request_binary(self, endpoint: str, query: dict[str, Any]) -> ToolResult:
        request = urllib.request.Request(self._url(self._path(endpoint), query), method="GET")
        body, headers = self._open(request)
        return ToolResult(body, self._header_cost(headers))

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> ToolResult:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = urllib.request.Request(
            self._url(path, query), data=payload, headers=headers, method=method
        )
        raw, response_headers = self._open(request)
        data = json.loads(raw)
        return ToolResult(data, data.get("cost", self._header_cost(response_headers)))

    def _open(self, request: urllib.request.Request) -> tuple[bytes, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = raw.decode(errors="replace")
            error_type = BudgetExhausted if exc.code == 429 else ToolAPIError
            raise error_type(exc.code, detail) from exc

    def _url(self, path: str, query: dict[str, Any] | None) -> str:
        suffix = f"?{urllib.parse.urlencode(query)}" if query else ""
        return f"{self.base_url}{path}{suffix}"

    @staticmethod
    def _header_cost(headers: Any) -> dict[str, int | float]:
        value = headers.get(COST_HEADER)
        return json.loads(value) if value else {}


ToolClient = WitnessClient
