import copy
import pytest

from witness.events import Event, scored_text
from witness.events_evaluation import (FIELD_JUDGE_PROMPT, FIELD_PROMPT_HASH,
                                      JUDGE_PROMPT, PROMPT_HASH, calibrate)
from witness.judge import JudgeConfig, parallel_score
from witness.providers import ApiFieldJudge, ApiJudge


EVENT = {"timestamp": 20., "actor": "la persona", "action": "abre",
         "objects": ["la puerta"], "details": []}
FIELDS = scored_text(Event.model_validate(EVENT))
VALUE = {"narration": "A person opens the door.", "event_fields": FIELDS}


class Model:
    identity = "synthetic-provider"

    def __init__(self, raw):
        self.raw, self.requests = raw, []

    def __call__(self, prompt, value, *, schema):
        self.requests.append((prompt, value, schema))
        return copy.deepcopy(self.raw)


@pytest.mark.parametrize("relation", ["supported", "unbacked", "uncertain", "contradiction"])
def test_field_judge_preserves_each_decision_and_never_asks_for_a_second_aggregate(relation):
    fields = dict.fromkeys(FIELDS, "supported")
    fields["action"] = relation
    model = Model({"fields": fields})
    judge = ApiFieldJudge(model)
    assert judge(judge.prompt, VALUE) == {"relation": relation, "fields": fields}
    schema = model.requests[0][2]
    assert set(schema["properties"]) == set(schema["required"]) == {"fields"}
    assert set(schema["properties"]["fields"]["required"]) == set(FIELDS)
    assert schema["additionalProperties"] is False
    assert model.raw == {"fields": fields}


@pytest.mark.parametrize("values,expected", [
    (("uncertain", "contradiction", "unbacked"), "contradiction"),
    (("unbacked", "supported", "uncertain"), "uncertain"),
    (("supported", "unbacked", "supported"), "unbacked"),
])
def test_more_severe_field_cannot_be_hidden_by_other_fields(values, expected):
    fields = dict(zip(FIELDS, values))
    judge = ApiFieldJudge(Model({"fields": fields}))
    assert judge(judge.prompt, VALUE)["relation"] == expected


@pytest.mark.parametrize("raw", [None, [], {}, {"fields": {}}, {"fields": []},
    {"fields": {"action": "supported"}},
    {"fields": {**dict.fromkeys(FIELDS, "supported"), "extra": "supported"}},
    {"fields": dict.fromkeys(FIELDS, "unknown")},
    {"fields": dict.fromkeys(FIELDS, None)},
    {"fields": dict.fromkeys(FIELDS, [])},
    {"fields": dict.fromkeys(FIELDS, "supported"), "relation": "supported"}])
def test_incomplete_unknown_or_extra_fields_still_fail_closed(raw):
    judge = ApiFieldJudge(Model(raw))
    with pytest.raises(ValueError, match="invalid_judge_fields"):
        judge(judge.prompt, VALUE)


def test_legacy_rejection_and_prompt_identity_remain_versioned(tmp_path):
    raw = {"relation": "supported", "fields": dict.fromkeys(FIELDS, "contradiction")}
    legacy = ApiJudge(Model(raw))
    with pytest.raises(ValueError, match="judge_inconsistent_fields"):
        legacy(JUDGE_PROMPT, VALUE)
    new = ApiFieldJudge(Model({"fields": raw["fields"]}))
    assert new.identity != legacy.identity
    assert FIELD_PROMPT_HASH != PROMPT_HASH
    with pytest.raises(ValueError, match="judge_prompt_identity_mismatch"):
        new(JUDGE_PROMPT, VALUE)
    identity = JudgeConfig().identity(tmp_path/"cache", budget_path=tmp_path/"budget")
    assert identity["prompt_hash"] == FIELD_PROMPT_HASH
    assert identity["evaluator_id"].endswith(":fields-only-v2")


def test_calibration_and_stored_decision_replay_use_the_same_new_prompt():
    model = Model({"fields": dict.fromkeys(FIELDS, "supported")})
    judge = ApiFieldJudge(model)
    cases = [{"case_id": str(i), "partition": "calibration", "event": EVENT,
              "input": VALUE, "expected": {"relation": "supported", "fields": model.raw["fields"]}}
             for i in range(300)]
    calibration = calibrate(cases, judge, evaluator_id=judge.identity)
    assert calibration["prompt_hash"] == FIELD_PROMPT_HASH
    assert {p for p, _, _ in model.requests} == {FIELD_JUDGE_PROMPT}
    # This synthetic set has no contradictions and must not pass calibration.
    assert calibration["passed"] is False
    value = {"reference": {"duration": 60., "events": [{"timestamp": 20., "text": VALUE["narration"]}]},
             "response": {"schema_version": "5.0", "events": [EVENT]}}
    scored = parallel_score(value, judge, calibration)
    assert scored["prompt_hash"] == FIELD_PROMPT_HASH
    assert scored["score"]["f1"] == 1.0
    assert scored["score"]["provisional"] is True
