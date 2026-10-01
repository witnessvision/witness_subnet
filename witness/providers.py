"""Bounded Responses adapter (OpenAI or SayGM) for labeling and judging: cached, budgeted, never silently re-sent.

Every answer is cached by the exact request. Before a paid call the worst-case
price is reserved in the daily budget; after it, the reservation settles to the
reported usage. A call the provider rejects is settled at zero and may be sent
again; a call lost in transport keeps its full reservation and may be sent again;
a call whose outcome was never recorded (a crash mid-call) is never re-sent.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sqlite3
import time
import threading
import uuid

import httpx

from witness.budget import BudgetUnavailable, DailyBudget
from witness.events import canonical_bytes, content_hash
from witness.storage import write_private

# (endpoint, key variable). SayGM resells the same OpenAI models and settles each call in nano-dollars.
PROVIDERS = {"openai": ("https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
             "saygm": ("https://api.saygm.com/v1/responses", "GM_API_KEY")}
# OpenAI standard <=272k rates per million tokens (input, cached input, output), checked
# 2026-09-26 at developers.openai.com/api/docs/pricing. SayGM lists about 0.815 of these,
# so they are safe reservation ceilings for both providers.
RATES = {"gpt-5.6-sol": (4., .4, 20.), "gpt-5.6-terra": (2., .2, 12.), "gpt-5.6-luna": (.2, .02, 1.2),
         "gpt-6-luna": (.1, .01, .5)}
MAX_CONTEXT = 272000
IMAGE_TOKENS = 4096  # upper bound per high-detail image


class ApiText:
    _slots = threading.BoundedSemaphore(4)
    _requests = [threading.RLock() for _ in range(64)]

    def __init__(self, model: str, cache: Path, *, provider: str = "openai", effort: str = "low",
                 max_tokens: int = 2048, budget_path: str | None = None, cache_only: bool = False,
                 daily_limit_usd: float | None = 9.):
        if model not in RATES:
            raise ValueError("model_has_no_verified_rate")
        if provider not in PROVIDERS:
            raise ValueError("unsupported_api_provider")
        self.provider = provider
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.cache, self.cache_only = cache, cache_only
        self.budget_path = budget_path or os.environ.get("WITNESS_BUDGET_DB")
        self.daily_limit_usd = daily_limit_usd
        self.calls: list[dict] = []
        self.request_timeout_s = lambda: 300.
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ledger = cache / "usage.sqlite3"
        with sqlite3.connect(self.ledger, timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, input_hash TEXT, reserve REAL, "
                       "cost REAL, record TEXT)")
        self.ledger.chmod(0o600)

    @property
    def identity(self) -> str:
        return self.provider + "-responses:" + content_hash({"model": self.model, "effort": self.effort,
                                                             "max_tokens": self.max_tokens, "adapter": "api-text-v3"})

    def _record(self, call_id: str, record: dict, cost: float | None = None) -> None:
        with sqlite3.connect(self.ledger, timeout=30) as db:
            db.execute("UPDATE calls SET cost=?, record=? WHERE id=?", (cost, json.dumps(record), call_id))

    def __call__(self, prompt: str, value: dict, *, schema: dict, images: list[bytes] = ()) -> dict:
        # Single flight for identical requests, including across adapter instances.
        # Cache keys and durable reservations remain authoritative after restart.
        import hashlib
        key = content_hash({'cache': str(self.cache.resolve()), 'identity': self.identity,
                            'prompt': prompt, 'value': value, 'schema': schema,
                            'images': [hashlib.sha256(i).hexdigest() for i in images]})
        lock = self._requests[int(key[:8], 16) % len(self._requests)]
        if not lock.acquire(timeout=max(0., self.request_timeout_s())):
            raise TimeoutError('api_queue_deadline_expired')
        try:
            if not self._slots.acquire(timeout=max(0., self.request_timeout_s())):
                raise TimeoutError('api_queue_deadline_expired')
            try:
                return self._call(prompt, value, schema=schema, images=images)
            finally:
                self._slots.release()
        finally:
            lock.release()

    def _call(self, prompt: str, value: dict, *, schema: dict, images: list[bytes] = ()) -> dict:
        """``images`` are JPEG bytes sent at high detail after the JSON value."""
        text = canonical_bytes(value).decode()
        content = [{"type": "input_text", "text": text}] + [
            {"type": "input_image", "detail": "high", "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}
            for image in images]
        request = {"model": self.model, "store": False, "instructions": prompt,
                   "input": [{"role": "user", "content": content}],
                   "max_output_tokens": self.max_tokens, "reasoning": {"effort": self.effort},
                   "text": {"format": {"type": "json_schema", "name": "result", "strict": True, "schema": schema}}}
        digest = content_hash({"adapter": self.identity, "request": request})
        path = self.cache / (digest + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            self.calls.append({"input_hash": digest, "cache_hit": True, "cost_usd_this_call": 0.,
                               "response_id": cached.get("response_id")})
            return cached["result"]
        if self.cache_only:
            raise ValueError("api_cache_miss")
        endpoint, key_name = PROVIDERS[self.provider]
        key = os.environ.get(key_name)
        if not key:
            raise ValueError("api_key_unavailable")
        # Bytes over-count tokens, so this bound is conservative.
        upper_input = (len(prompt.encode()) + len(text.encode()) + len(canonical_bytes(schema)) + 4096
                       + IMAGE_TOKENS * len(images))
        if upper_input > MAX_CONTEXT:
            raise ValueError("request_exceeds_priced_context_bound")
        rate_in, rate_cached, rate_out = RATES[self.model]
        reserve = (upper_input * rate_in * 1.25 + self.max_tokens * rate_out) / 1e6
        if not self.budget_path:
            raise BudgetUnavailable("persistent_budget_path_required")
        timeout = self.request_timeout_s()
        if timeout <= 0:
            raise TimeoutError('api_evaluation_deadline_expired')
        budget = DailyBudget(Path(self.budget_path), daily_limit_usd=self.daily_limit_usd)
        call_id = uuid.uuid4().hex
        with sqlite3.connect(self.ledger, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            # A dispatched call without a recorded outcome may have been charged:
            # never send the same paid request again after a crash.
            for (record,) in db.execute("SELECT record FROM calls WHERE input_hash=?", (digest,)):
                if json.loads(record)["status"] == "usage_not_received":
                    raise ValueError("api_outcome_unknown")
            budget.reserve(role="validator", provider=self.provider, input_hash=digest, upper_usd=reserve, call_id=call_id)
            db.execute("INSERT INTO calls VALUES (?,?,?,NULL,?)", (call_id, digest, reserve, json.dumps(
                {"model_requested": self.model, "status": "usage_not_received", "requested_unix": time.time()})))
        started = time.monotonic()
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.post(endpoint, json=request, headers={"Authorization": "Bearer " + key})
        except httpx.TransportError as error:
            # No answer: the call may have been billed, so its full reservation stays charged,
            # but the outcome is known to have failed and the request may be sent again.
            self._record(call_id, {"status": "transport_error", "error": type(error).__name__,
                                   "elapsed_s": time.monotonic() - started}, cost=reserve)
            budget.settle(call_id, reserve, {"provider": self.provider, "transport_error": type(error).__name__})
            raise RuntimeError(f"{self.provider}_transport_{type(error).__name__}") from error
        if response.status_code != 200:
            # A rejected request is not billed; keep no raw error body, which may echo request data.
            self._record(call_id, {"status": "provider_error", "http_status": response.status_code,
                                   "elapsed_s": time.monotonic() - started}, cost=0.)
            budget.settle(call_id, 0., {"provider": self.provider, "http_status": response.status_code})
            raise RuntimeError(f"{self.provider}_status_{response.status_code}")
        raw = response.json()
        usage = raw["usage"]
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        if any(type(usage.get(field)) is not int or usage[field] < 0 for field in ("input_tokens", "output_tokens")) \
                or type(cached) is not int or not 0 <= cached <= usage["input_tokens"]:
            raise ValueError("invalid_provider_usage")
        cost = ((usage["input_tokens"] - cached) * rate_in + cached * rate_cached + usage["output_tokens"] * rate_out) / 1e6
        if self.provider == "saygm":
            nano = usage.get("cost_nano_usd")
            if type(nano) is not int or nano < 0:
                raise ValueError("invalid_provider_settlement")
            cost = nano / 1e9  # SayGM settles the actual charge
        record = {"status": raw["status"], "model_requested": self.model, "model_returned": raw["model"],
                  "response_id": raw["id"], "usage": usage, "cost_usd": cost, "elapsed_s": time.monotonic() - started}
        self._record(call_id, record, cost=cost)
        budget.settle(call_id, cost, {"provider": self.provider, "response_id": raw["id"]})
        if raw["model"] != self.model and not raw["model"].startswith(self.model + "-"):
            raise ValueError("provider_changed_model")
        if raw["status"] != "completed":
            raise ValueError("api_output_incomplete")
        output = "".join(content["text"] for item in raw["output"] if item["type"] == "message"
                         for content in item["content"] if content["type"] == "output_text")
        result = json.loads(output)
        write_private(path, {"input_hash": digest, "result": result, **record})
        self.calls.append({"input_hash": digest, "cache_hit": False, "cost_usd_this_call": cost,
                           "response_id": raw["id"]})
        return result
