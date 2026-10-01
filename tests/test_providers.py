import sqlite3

import httpx
import pytest

from witness.budget import BudgetUnavailable, DailyBudget
from witness.providers import ApiText

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


@pytest.fixture(autouse=True)
def daily_budget_path(tmp_path, monkeypatch):
    monkeypatch.setenv("WITNESS_BUDGET_DB", str(tmp_path / "daily.sqlite3"))


def fake_client(monkeypatch, calls, statuses=(200,), usage=None):
    statuses = list(statuses)

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            calls.append({"url": url, **kwargs})
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            return httpx.Response(status, json={
                "model": kwargs["json"]["model"], "id": "test-response", "status": "completed",
                "usage": {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 20},
                          **(usage or {})},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}]})
    monkeypatch.setattr("witness.providers.httpx.Client", Client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-credential")


def test_cache_cost_and_no_key_persistence(tmp_path, monkeypatch):
    calls = []
    fake_client(monkeypatch, calls)
    model = ApiText("gpt-5.6-terra", tmp_path)
    for _ in range(2):
        assert model("system", {"x": 1}, schema=SCHEMA) == {"ok": True}
    assert [row["cache_hit"] for row in model.calls] == [False, True]
    assert len(calls) == 1 and calls[0]["json"]["store"] is False and "tools" not in calls[0]["json"]
    for path in tmp_path.glob("*.json"):
        assert "test-only-not-a-credential" not in path.read_text()
    with sqlite3.connect(model.ledger) as db:
        assert db.execute("SELECT cost FROM calls").fetchone()[0] == pytest.approx((80 * 2 + 20 * .2 + 20 * 12) / 1e6)


def test_daily_cap_blocks_before_dispatch(tmp_path, monkeypatch):
    calls = []
    fake_client(monkeypatch, calls)
    DailyBudget(tmp_path / "daily.sqlite3").reserve(role="validator", provider="openai", input_hash="prior", upper_usd=9.)
    with pytest.raises(BudgetUnavailable):
        ApiText("gpt-5.6-terra", tmp_path / "cache")("system", {"x": 1}, schema=SCHEMA)
    assert calls == []


def test_rejected_call_frees_budget_and_may_be_sent_again(tmp_path, monkeypatch):
    calls = []
    fake_client(monkeypatch, calls, statuses=(500, 200))
    model = ApiText("gpt-5.6-terra", tmp_path / "cache")
    with pytest.raises(RuntimeError, match="openai_status_500"):
        model("system", {"x": 1}, schema=SCHEMA)
    assert DailyBudget(tmp_path / "daily.sqlite3").totals()["validator"]["reserved_usd"] == 0
    assert model("system", {"x": 1}, schema=SCHEMA) == {"ok": True} and len(calls) == 2


def test_unknown_outcome_is_never_sent_again(tmp_path, monkeypatch):
    calls = []
    fake_client(monkeypatch, calls)

    def crash(*args, **kwargs):
        raise httpx.ReadTimeout("no answer")
    model = ApiText("gpt-5.6-terra", tmp_path / "cache")
    monkeypatch.setattr(model, "_record", crash)  # the response is lost before it is recorded
    with pytest.raises(httpx.ReadTimeout):
        model("system", {"x": 1}, schema=SCHEMA)
    with pytest.raises(ValueError, match="api_outcome_unknown"):
        ApiText("gpt-5.6-terra", tmp_path / "cache")("system", {"x": 1}, schema=SCHEMA)
    assert len(calls) == 1


def test_configuration_fails_closed(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="model_has_no_verified_rate"):
        ApiText("unpriced-model", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key_unavailable"):
        ApiText("gpt-5.6-terra", tmp_path)("system", {"x": 1}, schema=SCHEMA)
    with pytest.raises(ValueError, match="api_cache_miss"):
        ApiText("gpt-5.6-terra", tmp_path, cache_only=True)("system", {"x": 1}, schema=SCHEMA)


def test_saygm_settles_the_reported_charge_and_images_raise_the_reservation(tmp_path, monkeypatch):
    calls = []
    fake_client(monkeypatch, calls, usage={"cost_nano_usd": 1_500_000})
    monkeypatch.setenv("GM_API_KEY", "gm-test-only")
    model = ApiText("gpt-6-luna", tmp_path / "cache", provider="saygm")
    assert model("system", {"x": 1}, schema=SCHEMA, images=[b"\xff\xd8jpeg"]) == {"ok": True}
    assert calls[0]["url"] == "https://api.saygm.com/v1/responses"
    assert calls[0]["headers"]["Authorization"] == "Bearer gm-test-only"
    assert calls[0]["json"]["input"][0]["content"][1]["type"] == "input_image"
    assert model.calls[0]["cost_usd_this_call"] == pytest.approx(.0015)
    with pytest.raises(ValueError, match="unsupported_api_provider"):
        ApiText("gpt-6-luna", tmp_path, provider="other")


def test_transport_failure_keeps_the_reservation_and_may_be_sent_again(tmp_path, monkeypatch):
    calls = []
    fake_client(monkeypatch, calls)
    import witness.providers as providers
    fake = providers.httpx.Client
    failures = [httpx.ReadTimeout("slow")]

    class Flaky(fake):
        def post(self, url, **kwargs):
            if failures:
                raise failures.pop()
            return super().post(url, **kwargs)
    monkeypatch.setattr("witness.providers.httpx.Client", Flaky)
    model = ApiText("gpt-5.6-terra", tmp_path / "cache")
    with pytest.raises(RuntimeError, match="openai_transport_ReadTimeout"):
        model("system", {"x": 1}, schema=SCHEMA)
    reserved = DailyBudget(tmp_path / "daily.sqlite3").totals()["validator"]["settled_usd"]
    assert reserved > 0  # a call that may have been billed stays charged at its ceiling
    assert model("system", {"x": 1}, schema=SCHEMA) == {"ok": True} and len(calls) == 1


def test_connection_establishment_retries_once_under_same_reservation(tmp_path, monkeypatch):
    calls = []
    fake_client(monkeypatch, calls)
    import witness.providers as providers
    original = providers.httpx.Client
    attempts = []
    class Client(original):
        def __init__(self, **kwargs):
            attempts.append(kwargs['timeout'])
        def post(self, *args, **kwargs):
            if len(attempts) == 1:
                raise httpx.ConnectError('connection failed before request')
            return super().post(*args, **kwargs)
    monkeypatch.setattr(providers.httpx, 'Client', Client)
    model = ApiText('gpt-5.6-terra', tmp_path/'cache')
    assert model('system',{'x':1},schema=SCHEMA)=={'ok':True}
    assert len(attempts)==2 and attempts[1]<=attempts[0]
    with sqlite3.connect(model.ledger) as db:
        assert db.execute('select count(*) from calls').fetchone()[0]==1
