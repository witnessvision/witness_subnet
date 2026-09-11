"""Grounded action/result scoring and conservation of shared semantic credit.

Version 3 compares all declared semantic fields and interval evidence. Sharing is
per recovered fact, so deletion/noise in a copied answer cannot mint fresh credit.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
import re
import unicodedata
from typing import Any

from witness import score_v18 as legacy
from witness.score_v1_1_0 import QUALITY_THRESHOLD, reward_metrics
from witness.tools.metering import visual_token_cost

SCORER_VERSION = "3.0.0"
MAX_BYTES = 262_144
MAX_ITEMS = 512
WEIGHTS = {"events": .45, "qa": .30, "dialogue": .10, "shots": .05,
           "on_screen_text": .04, "audio_events": .03, "intentional_errors": .03}
SEMANTIC_FIELDS = {"action", "actor", "object", "target", "recipient", "instrument", "result"}


def normalize(value: Any) -> str | None:
    # Actor A is an identity. Article-dropping normalization would silently turn
    # an omitted actor into an accepted assertion about A.
    if not isinstance(value, str) or not value.strip():
        return None
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", value))


def _aliases(entities):
    names: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        # Appearance can change during the clip. Initial colors/shapes are not
        # durable identity aliases; use only explicit stable entity descriptions.
        for field in ("id", "visual_description", "kind"):
            if isinstance(entity.get(field), str):
                key = normalize(entity[field])
                if not key:
                    continue
                names[key].add(str(entity["id"]))
                if key.startswith("the "):
                    names[key[4:]].add(str(entity["id"]))
    return {key: next(iter(values)) for key, values in names.items() if len(values) == 1}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("timestamps must be finite numbers")
    return float(value)


def _interval(row, fps, duration):
    values = []
    for name in ("start", "end"):
        seconds = _number(row[name]) if name in row else None
        frames = _number(row[name + "_frame"]) / fps if name + "_frame" in row else None
        if seconds is None and frames is None:
            raise ValueError("missing interval boundary")
        # Frame and audio-sample grids need not align. Both timestamps must
        # describe the same nearest video frame, not the same binary float.
        if seconds is not None and frames is not None and abs(seconds - frames) > .5 / fps + 1e-6:
            raise ValueError("contradictory frame and time fields")
        values.append(seconds if seconds is not None else frames)
    if not 0 <= values[0] < values[1] <= duration + 1e-5:
        raise ValueError("interval outside video")
    return values


def _point(row, fps, duration):
    if "t" in row:
        time = _number(row["t"])
        if "frame" in row and abs(time - _number(row["frame"]) / fps) > .5 / fps + 1e-6:
            raise ValueError("contradictory frame and time fields")
    elif "frame" in row:
        time = _number(row["frame"]) / fps
    else:
        raise ValueError("missing point timestamp")
    if not 0 <= time < duration:
        raise ValueError("point outside video")
    return time


def _sequence(value):
    if isinstance(value, str):
        value = value.split(">")
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_ITEMS:
        return None
    result = [normalize(item) for item in value]
    return result if all(result) else None


def _matched_truth(edges):
    """Maximum one-to-one matching; return the individual truth facts recovered."""
    owner = {}
    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owner or augment(owner[j], visited):
                owner[j] = i
                return True
        return False
    for i in range(len(edges)):
        augment(i, set())
    return set(owner.values())


def _bounded(value):
    if not isinstance(value, Mapping):
        raise ValueError("reconstruction must be an object")
    if len(json.dumps(value, allow_nan=False).encode()) > MAX_BYTES:
        raise ValueError("reconstruction exceeds byte limit")
    for name in WEIGHTS:
        rows = value.get(name, {} if name == "qa" else [])
        if name != "qa" and not isinstance(rows, list):
            raise ValueError("reconstruction families must be arrays")
        if isinstance(rows, (list, dict)) and len(rows) > MAX_ITEMS:
            raise ValueError("reconstruction exceeds item limit")
    if not isinstance(value.get("qa", {}), dict):
        raise ValueError("qa must be an object")
    return value


def score_reconstruction(scene: Mapping[str, Any], reconstruction: Any,
                         cost: Mapping[str, Any] | None = None, *, q_min=None,
                         cost_lambda: float = legacy.LAMBDA) -> dict[str, Any]:
    if scene.get("schema_version") != "3.0":
        raise ValueError("scorer3.0.0 requires scene schema3.0")
    config = scene["evaluation"]
    fields = config["event_fields"]
    if not fields or not set(fields) <= SEMANTIC_FIELDS or "action" not in fields:
        raise ValueError("invalid private semantic event contract")
    if not scene.get("events") or not scene.get("qa"):
        raise ValueError("grounded scenes require events and questions")
    groups = config["required_qa_groups"]
    if not groups or any(not any(q.get("group") == g for q in scene["qa"]) for g in groups):
        raise ValueError("missing required private question group")
    fps, duration = int(scene["fps"]), float(scene["duration"])
    tolerance = float(config.get("interval_tolerance", .35))
    actor_aliases, object_aliases = _aliases(scene.get("actors", [])), _aliases(scene.get("objects", []))
    invalid = None
    try:
        submitted = _bounded(reconstruction)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        submitted, invalid = {}, type(exc).__name__
    active = {name: weight for name, weight in WEIGHTS.items() if scene.get(name)}
    active = {name: weight / sum(active.values()) for name, weight in active.items()}
    scores, credits, diagnostics = {}, {}, []
    asserted_claims = 0
    correct_claims = 0
    qa_groups = defaultdict(list)
    event_indexes = {event['id']: i for i, event in enumerate(scene['events'])}
    if len(event_indexes) != len(scene['events']):
        raise ValueError("private event IDs must be unique")
    if len({q['id'] for q in scene['qa']}) != len(scene['qa']):
        raise ValueError("private question IDs must be unique")
    for question in scene['qa']:
        ids = question.get('event_ids')
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or any(event_id not in event_indexes for event_id in ids)):
            raise ValueError("private question requires valid supporting event IDs")
    recovered_events = set()

    def semantic(value, field):
        if value is None:
            return None
        key = normalize(value)
        if field in {"actor", "recipient"} and actor_aliases:
            return actor_aliases.get(key, "!unresolved!")
        if field == "object" and object_aliases:
            return object_aliases.get(key, "!unresolved!")
        return key if key else "!unresolved!"

    # Bad private labels are invalid benchmark samples, not miner failures.
    # In particular, two unknown identities must never match each other.
    for event in scene["events"]:
        for field in fields:
            if field not in event:
                raise ValueError("missing private semantic event field")
            value = event[field]
            nullable = field in {"recipient", "instrument", "result"}
            if ((value is None and not nullable)
                    or (value is not None and semantic(value, field) == "!unresolved!")):
                raise ValueError("unresolved private semantic event field")
    for family, required in (("dialogue", ("text",)), ("on_screen_text", ("text",)),
                             ("audio_events", ("kind",)),
                             ("intentional_errors", ("type", "claim", "truth"))):
        if any(not normalize(row.get(field)) for row in scene.get(family, []) for field in required):
            raise ValueError("missing private semantic auxiliary field")

    for family in WEIGHTS:
        truth = scene.get(family, [])
        if family == "qa":
            answers = submitted.get("qa", {})
            asserted_claims += len(answers)
            matches = set()
            for i, question in enumerate(truth):
                expected = _sequence(question["a"])
                if expected is None:
                    raise ValueError("invalid private sequence answer")
                supported = all(event_indexes[event_id] in recovered_events
                                for event_id in question['event_ids'])
                accepted = supported and _sequence(answers.get(question["id"])) == expected
                qa_groups[question["group"]].append(float(accepted))
                if accepted:
                    matches.add(i)
            count = len(truth)
        else:
            predictions = submitted.get(family, [])
            if not isinstance(predictions, list):
                predictions = [None]
            count = len(predictions)
            asserted_claims += count
            edges = []
            valid_predictions = []
            for j, row in enumerate(predictions):
                try:
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                    timing = (_point(row, fps, duration) if family in {"audio_events", "intentional_errors"}
                              else _interval(row, fps, duration))
                    valid_predictions.append((j, row, timing))
                except (ValueError, TypeError, OverflowError):
                    diagnostics.append({"family": family, "index": j, "reason": "invalid timing or row"})
            for expected in truth:
                expected_time = (_point(expected, fps, duration) if family in {"audio_events", "intentional_errors"}
                                 else _interval(expected, fps, duration))
                candidates = []
                for j, actual, timing in valid_predictions:
                    if isinstance(expected_time, list):
                        overlap = max(0, min(expected_time[1], timing[1]) - max(expected_time[0], timing[0]))
                        union = max(expected_time[1], timing[1]) - min(expected_time[0], timing[0])
                        distance = max(abs(a - b) for a, b in zip(expected_time, timing))
                        temporal = overlap / union >= .5 or distance <= tolerance
                    else:
                        distance = abs(expected_time - timing)
                        temporal = distance <= tolerance
                    if not temporal:
                        continue
                    if family == "events":
                        # Required nullable fields must be explicitly present. Unknown
                        # or omitted destinations/recipients cannot act as wildcards.
                        accepted = all(f in actual and f in expected and
                                       semantic(expected[f], f) == semantic(actual[f], f)
                                       for f in fields)
                    elif family in {"dialogue", "on_screen_text"}:
                        accepted = normalize(expected.get("text")) == normalize(actual.get("text"))
                    elif family == "audio_events":
                        accepted = normalize(expected.get("kind")) == normalize(actual.get("kind"))
                    elif family == "intentional_errors":
                        accepted = all(normalize(expected.get(f)) == normalize(actual.get(f))
                                       for f in ("type", "claim", "truth"))
                    else:
                        accepted = True
                    if accepted:
                        candidates.append((distance, j))
                edges.append([j for _, j in sorted(candidates)])
            matches = _matched_truth(edges)
            if family == "events":
                recovered_events = matches
        score = (2 * len(matches) / (len(truth) + count)) if truth or count else 0.0
        correct_claims += len(matches)
        scores[family] = score
        if family in active:
            for i in range(len(truth)):
                credits[f"{family}:{i}"] = active[family] / len(truth) if i in matches else 0.0
    group_scores = {g: sum(qa_groups[g]) / len(qa_groups[g]) for g in groups}
    coverage = sum(credits.values())
    weighted = sum(active[family] * scores[family] for family in active)
    # Absent families grant no free credit, but their fabricated claims still
    # count against precision. Otherwise a silent video could reward invented
    # speech without limit merely because dialogue has zero active weight.
    claim_precision = correct_claims / asserted_claims if asserted_claims else 0.0
    quality = 0.0 if invalid else min(weighted, coverage, claim_precision, scores["events"], *group_scores.values())
    quality = round(quality, 12)
    # The conserved credit mass bounds pre-sharing quality, including rounding.
    quality = min(quality, coverage)
    cost_input = {} if cost is None else cost
    if not isinstance(cost_input, Mapping):
        raise ValueError("cost must be a JSON object")
    values = {key: legacy._nonnegative_number(cost_input, key)
              for key in ("visual_tokens", "audio_seconds", "transcript_chars")}
    if not math.isfinite(cost_lambda) or not 0 <= cost_lambda <= 1:
        raise ValueError("cost_lambda must be in [0,1]")
    reference = math.ceil(duration) * visual_token_cost(640, 360)
    ratio = sum(values.values()) / reference
    efficiency = max(0., 1 - cost_lambda * ratio)
    metrics = reward_metrics(quality, efficiency, valid=invalid is None)
    if invalid:
        diagnostics.append({"family": "reconstruction", "reason": invalid})
    fingerprint = hashlib.sha256(json.dumps(credits, sort_keys=True).encode()).hexdigest()
    return {"schema_version": "1.0", "benchmark_version": SCORER_VERSION,
            "quality": quality, "score": metrics["score"], "metrics": metrics,
            "family_scores": scores, "active_weights": active,
            "grounding": {"question_groups": group_scores, "semantic_coverage": coverage,
                          "reconstruction_quality": weighted, "claim_precision": claim_precision},
            "semantic_credits": credits, "semantic_fingerprint": fingerprint,
            "gate": {"q_min": QUALITY_THRESHOLD, "passed": metrics["full_reward_eligible"],
                     "partial_floor": QUALITY_THRESHOLD, "reward_factor": metrics["reward_factor"]},
            "cost": {**values, "total": sum(values.values()), "cost_ref": reference,
                     "ratio": ratio, "lambda": cost_lambda, "efficiency_factor": efficiency},
            "diagnostics": diagnostics}


def apply_fact_sharing(records: list[dict[str, Any]]) -> None:
    """Conserve the sum of semantic credit across exact and partial clones.

    For fact f with total submitted credit C_f, each eligible miner receives
    c_if/max(1,C_f/w_f) of the fact's fixed weight w_f. Since pre-sharing score is
    bounded by its recovered mass, the total shared score cannot exceed 1 per
    scene, regardless of the number of partial/exact copies or false claims.
    """
    by_scene = defaultdict(list)
    for record in records:
        if record.get("score_version") == SCORER_VERSION:
            by_scene[str(record["scene_id"])].append(record)
    for scene_records in by_scene.values():
        eligible = [r for r in scene_records if r["responded"] and r["score_before_duplicates"] > 0]
        # A validator emits at most one record per UID/scene. Reject contradictory
        # local state instead of counting duplicate rows as independent identities.
        if len({int(r["uid"]) for r in eligible}) != len(eligible):
            raise ValueError("duplicate UID in grounded scene records")
        totals = defaultdict(float)
        weights = defaultdict(float)
        for r in eligible:
            for fact, value in r["semantic_credits"].items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid validator semantic credit")
                totals[fact] += value
                weights[fact] = max(weights[fact], value)
        for r in scene_records:
            mass = sum(r.get("semantic_credits", {}).values())
            if r not in eligible or mass <= 0:
                fraction = 0.0
                count = 0
            else:
                shared = sum(value * weights[fact] / totals[fact]
                             for fact, value in r["semantic_credits"].items() if totals[fact] > 0)
                fraction = min(1., shared / mass)
                count = sum(any(value > 0 and other["semantic_credits"].get(fact, 0) > 0
                                for fact, value in r["semantic_credits"].items()) for other in eligible)
            r["shared_credit_fraction"] = fraction
            r["duplicate_count"] = count
            r["score"] = round(float(r["score_before_duplicates"]) * fraction, 12)
            r["reconstruction_hash"] = r.get("semantic_fingerprint")
            if "metrics" in r:
                r["metrics"]["score"] = r["score"]
