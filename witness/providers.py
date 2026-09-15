"""Bounded provider adapters. No retries, tools or automatic provider changes."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid
import httpx
from witness.events import canonical_bytes, content_hash
from witness.storage import write_private
from witness.budget import DailyBudget, BudgetUnavailable
from witness.events_evaluation import FIELD_JUDGE_PROMPT

# Standard <=272k rates, checked 2026-09-13 at developers.openai.com/api/docs/pricing.
RATES = {"gpt-5.6-sol": (4., .4, 20.), "gpt-5.6-terra": (2., .2, 12.),
         "gpt-5.6-luna": (.2, .02, 1.2), "gpt-6-astra": (10., 1., 50.),
         "gpt-4.1-2025-04-14": (2., .5, 8.)}
PROVIDERS = {
    'openai': ('https://api.openai.com/v1/responses', 'OPENAI_API_KEY'),
    'saygm': ('https://api.saygm.com/v1/responses', 'GM_API_KEY'),
}
# SayGM published retail ceilings verified against /v1/models on 2026-09-13.
# Reserve at those ceilings; settle from usage.cost_nano_usd, never floor quotes.
SAYGM_MODELS = {'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol'}


def load_key(path: Path, provider='openai'):
    name = PROVIDERS[provider][1]
    if path.stat().st_mode & 0o077:
        raise ValueError("unsafe_credential_permissions")
    for line in path.read_text().splitlines():
        if line.startswith(name+'='):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if not value:raise ValueError('api_key_unavailable')
            os.environ[name] = value
            return
    raise ValueError("api_key_unavailable")


class ApiText:
    def __init__(self, model: str, cache: Path, *, effort="low", max_tokens=2048, service_tier=None,
                 cache_only=False, provider='saygm', budget_path=None, budget_role='validator'):
        if provider not in PROVIDERS:raise ValueError('unsupported_api_provider')
        if provider=='saygm' and model not in SAYGM_MODELS:
            raise ValueError('model_has_no_verified_rate')
        if provider=='saygm' and service_tier=='priority':
            raise ValueError('unpriced_service_tier')
        self.provider=provider
        if model not in RATES:
            raise ValueError("model_has_no_verified_rate")
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        if service_tier not in (None,'default','priority'):
            raise ValueError('unsupported_service_tier')
        if service_tier=='priority' and model!='gpt-5.6-luna':
            raise ValueError('unpriced_fast_model')
        self.service_tier=service_tier
        self.cache_only=cache_only
        self.cache = cache
        self.budget_path = budget_path or os.environ.get("WITNESS_BUDGET_DB")
        self.budget_role = budget_role
        self.calls = []
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ledger = cache / "usage.sqlite3"
        with sqlite3.connect(self.ledger, timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, input_hash TEXT, reserve REAL, cost REAL, record TEXT)")
        self.ledger.chmod(0o600)

    @property
    def identity(self):
        value={"model":self.model,"effort":self.effort,"max_tokens":self.max_tokens,"adapter":"api-text-v2-daily"}
        if self.service_tier is not None:value['service_tier']=self.service_tier
        return self.provider+"-responses:"+content_hash(value)

    def __call__(self, prompt: str, value: dict, *, schema: dict, images: list[dict] | None = None) -> dict:
        text = canonical_bytes(value).decode()
        request = {"model":self.model,"store":False,"instructions":prompt,
            "input":[{"role":"user","content":[{"type":"input_text","text":text}, *(images or [])]}],
            "max_output_tokens":self.max_tokens,
            "text":{"format":{"type":"json_schema","name":"result","strict":True,"schema":schema}}}
        if self.model.startswith("gpt-4"):
            request["temperature"] = 0
        else:
            request["reasoning"] = {"effort":self.effort}
        if self.service_tier is not None:request['service_tier']=self.service_tier
        digest = content_hash({'adapter': self.identity, 'provider': self.provider, 'request': request})
        path = self.cache/(digest+".json")
        if path.exists():
            cached=json.loads(path.read_text())
            self.calls.append({'input_hash':digest,'model':self.model,'provider':self.provider,'cache_hit':True,
                               'estimated_usd_this_call':0.,'response_id':cached.get('response_id')})
            return cached["result"]
        if self.cache_only:
            raise ValueError('api_cache_miss')
        endpoint, key_name=PROVIDERS[self.provider]
        key=os.environ.get(key_name)
        if not key:raise ValueError('api_key_unavailable')
        # High-detail GPT-5.6/6 images have a 2500-patch budget and 1.2x
        # multiplier (images-vision guide, checked 2026-09-13). 4096/image
        # leaves room for framing tokens; our video images are only 640x360.
        if any(i.get('detail')!='high' for i in images or []):
            raise ValueError('unpriced_image_detail')
        upper_input = len(prompt.encode())+len(text.encode())+len(canonical_bytes(schema))+4096+4096*len(images or [])
        if upper_input > 272000:
            raise ValueError("request_exceeds_priced_context_bound")
        rate_in, rate_cached, rate_out = RATES[self.model]
        # Luna Fast short-context rates checked in the official pricing table:
        # $0.40 input / $0.04 cached / $2.40 output per million tokens.
        multiplier=2. if self.service_tier=='priority' else 1.
        rate_in,rate_cached,rate_out=(rate*multiplier for rate in (rate_in,rate_cached,rate_out))
        reserve = (upper_input*rate_in*1.25+self.max_tokens*rate_out)/1e6
        call_id = uuid.uuid4().hex
        if not self.budget_path:
            raise BudgetUnavailable("persistent_budget_path_required")
        budget = DailyBudget(Path(self.budget_path))
        budget.reserve(role=self.budget_role, provider=self.provider, input_hash=digest,
                       upper_usd=reserve, call_id=call_id)
        with sqlite3.connect(self.ledger, timeout=30) as db:
            pending={'model_requested':self.model,'provider':self.provider,'status':'usage_not_received','requested_unix':time.time()}
            db.execute("INSERT INTO calls VALUES (?,?,?,NULL,?)",(call_id,digest,reserve,json.dumps(pending)))
        started = time.monotonic()
        with httpx.Client(timeout=125., trust_env=False) as client:
            response = client.post(endpoint, json=request,
                headers={"Authorization":"Bearer "+key})
        if response.status_code != 200:
            # Retain no raw provider error payload that might echo request data.
            error_text=response.text.lower()
            categories=[label for label in ('rate limit','quota','budget','billing','tokens',
                'unsupported','permission','credits','balance','capacity','too many','authentication')
                if label in error_text]
            with sqlite3.connect(self.ledger,timeout=30) as db:
                db.execute('UPDATE calls SET record=? WHERE id=?',(json.dumps({
                    'model_requested':self.model,'provider':self.provider,'status':'provider_error','http_status':response.status_code,
                    'error_categories':categories,'elapsed_s':time.monotonic()-started}),call_id))
            raise RuntimeError(self.provider+"_status_"+str(response.status_code))
        raw = response.json()
        usage = raw["usage"]
        for field in ("input_tokens", "output_tokens"):
            if type(usage.get(field)) is not int or usage[field] < 0:
                raise ValueError("invalid_provider_usage")
        if self.service_tier=='priority' and raw.get('service_tier')=='default':
            rate_in,rate_cached,rate_out=(rate/2 for rate in (rate_in,rate_cached,rate_out))
        cached = usage.get("input_tokens_details",{}).get("cached_tokens",0)
        written = usage.get("input_tokens_details",{}).get("cache_write_tokens",0)
        if any(type(v) is not int or v < 0 for v in (cached, written)) or cached+written > usage["input_tokens"]:
            raise ValueError("invalid_provider_usage")
        cost = ((usage["input_tokens"]-cached-written)*rate_in+written*rate_in*1.25
                +cached*rate_cached+usage["output_tokens"]*rate_out)/1e6
        cost_basis='token_rate_estimate'
        if self.provider=='saygm':
            nano=usage.get('cost_nano_usd')
            if type(nano) is not int or nano<0:
                raise ValueError('invalid_provider_settlement')
            cost=nano/1e9
            if not math.isfinite(cost):raise ValueError('invalid_provider_settlement')
            cost_basis='provider_settled_nano_usd'
        record = {"model_requested":self.model,"model_returned":raw["model"],"response_id":raw["id"],
                  'provider':self.provider,'cost_usd':cost,'cost_basis':cost_basis,
                  "service_tier_requested":self.service_tier,"service_tier_returned":raw.get('service_tier'),
                  "usage":usage,"estimated_usd":cost,"elapsed_s":time.monotonic()-started,"status":raw["status"]}
        with sqlite3.connect(self.ledger, timeout=30) as db:
            db.execute("UPDATE calls SET cost=?, record=? WHERE id=?",(cost,json.dumps(record),call_id))
        budget.settle(call_id, cost, {"provider": self.provider, "response_id": raw["id"], "cost_basis": cost_basis})
        if raw["model"] != self.model and not raw["model"].startswith(self.model+"-"):
            raise ValueError("provider_changed_model")
        if raw["status"] != "completed":
            raise ValueError("api_output_incomplete")
        output = "".join(c["text"] for o in raw["output"] if o["type"] == "message"
                         for c in o["content"] if c["type"] == "output_text")
        result = json.loads(output)
        write_private(path, {"input_hash":digest,"result":result,**record})
        self.calls.append({'input_hash':digest,'model':self.model,'provider':self.provider,'cache_hit':False,
                           'cost_usd_this_call':cost,'cost_basis':cost_basis,
                           'estimated_usd_this_call':cost,'response_id':raw['id']})
        return result


class ApiJudge:
    def __init__(self, model: ApiText):
        self.model = model

    @property
    def identity(self):
        return self.model.identity+":all-fields-v1"

    def __call__(self, prompt, value):
        relation = {"type":"string","enum":["supported","contradiction","unbacked","uncertain"]}
        fields = list(value["event_fields"])
        schema = {"type":"object","properties":{"relation":relation,
            "fields":{"type":"object","properties":{k:relation for k in fields},
                      "required":fields,"additionalProperties":False}},
            "required":["relation","fields"],"additionalProperties":False}
        raw = self.model(prompt,value,schema=schema)
        derived = next(r for r in ("contradiction","uncertain","unbacked","supported") if r in raw["fields"].values())
        if raw["relation"] != derived:
            raise ValueError("judge_inconsistent_fields")
        return raw


class ApiFieldJudge(ApiJudge):
    """Field decisions are semantic; their precedence is computed locally.

    Raw model output remains in ApiText's provider/request-scoped cache. The
    canonical decision passed to the scorer retains every field unchanged.
    ApiJudge above remains the strict v1 adapter for historical experiments.
    """
    prompt = FIELD_JUDGE_PROMPT
    relations = ("contradiction", "uncertain", "unbacked", "supported")

    @property
    def identity(self):
        return self.model.identity + ":fields-only-v2"

    def __call__(self, prompt, value):
        if prompt != self.prompt:
            raise ValueError("judge_prompt_identity_mismatch")
        fields = value["event_fields"]
        if not isinstance(fields, dict) or not fields or any(type(k) is not str for k in fields):
            raise ValueError("invalid_judge_fields")
        relation = {"type": "string", "enum": list(self.relations)}
        schema = {"type": "object", "properties": {
            "fields": {"type": "object", "properties": {k: relation for k in fields},
                       "required": list(fields), "additionalProperties": False}},
            "required": ["fields"], "additionalProperties": False}
        raw = self.model(prompt, value, schema=schema)
        if (not isinstance(raw, dict) or set(raw) != {"fields"}
                or not isinstance(raw["fields"], dict) or set(raw["fields"]) != set(fields)
                or any(type(v) is not str or v not in self.relations for v in raw["fields"].values())):
            raise ValueError("invalid_judge_fields")
        derived = next(r for r in self.relations if r in raw["fields"].values())
        return {"relation": derived, "fields": dict(raw["fields"])}
