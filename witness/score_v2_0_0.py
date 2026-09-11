"""Scoring for observed temporal histories; historical scoring is unchanged."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from witness import score_v18 as legacy
from witness.score_v1_1_0 import QUALITY_THRESHOLD, reward_metrics

SCORER_VERSION = "2.0.0"
MAX_RECONSTRUCTION_BYTES = 262_144
MAX_FAMILY_ITEMS = 512
FAMILIES = tuple(legacy.WEIGHTS)


def _bounded(reconstruction: Any) -> dict[str, Any]:
    if not isinstance(reconstruction, Mapping):
        raise ValueError("reconstruction must be an object")
    if len(json.dumps(reconstruction, allow_nan=False).encode()) > MAX_RECONSTRUCTION_BYTES:
        raise ValueError("reconstruction exceeds byte limit")
    for family in FAMILIES:
        rows = reconstruction.get(family)
        if isinstance(rows, (list, dict)) and len(rows) > MAX_FAMILY_ITEMS:
            raise ValueError("reconstruction exceeds item limit")
    return deepcopy(dict(reconstruction))


def _history_answer(value: Any, resolver: legacy._EntityResolver) -> list[str] | None:
    if not isinstance(value, str):
        return None
    parts = value.split(">")
    if not 1 <= len(parts) <= MAX_FAMILY_ITEMS:
        return None
    result = [resolver.resolve(part.strip()) for part in parts]
    return result if all(x is not None for x in result) else None


def _normalize_histories(scene: Mapping[str, Any], reconstruction: dict[str, Any]) -> float:
    questions = [q for q in scene.get("qa", []) if q.get("type") == "temporal_sequence"]
    if len(questions) < 3:
        raise ValueError("temporal scenes require at least three history questions")
    resolver = legacy._EntityResolver(scene["actors"])
    answers = reconstruction.get("qa", {})
    if not isinstance(answers, dict):
        answers = {}
    correct = 0
    for question in questions:
        sequence = _history_answer(answers.get(question["id"]), resolver)
        accepted = sequence == question["actor_sequence"]
        correct += accepted
        # The inherited lexical QA scorer sees one canonical answer, with no
        # credit from partial histories or a bag of alternative actor sequences.
        answers[question["id"]] = question["a"] if accepted else ""
    reconstruction["qa"] = answers
    return correct / len(questions)


def _fingerprint(scene: Mapping[str, Any], reconstruction: Mapping[str, Any]) -> str:
    """Hash recovered scored facts, not arbitrary JSON spelling or metadata.

    Extra unmatched claims do not create a fresh identity: they already lower
    precision. Equivalent claims within scoring tolerance map to the same fact.
    """
    fps, tier = int(scene["fps"]), int(scene["difficulty"])
    actors = legacy._EntityResolver(scene["actors"])
    objects = legacy._EntityResolver(scene["objects"])
    tolerance = legacy.EVENT_TOLERANCE_SECONDS[tier] * fps

    def point(e, a, family):
        if abs(legacy._frame(e, fps) - legacy._frame(a, fps)) > tolerance:
            return False
        if family == "audio_events":
            return e.get("kind") == a.get("kind")
        field = "action" if family == "events" else "type"
        return (e.get(field) == a.get(field)
                and legacy._same_entity(e.get("actor"), a.get("actor"), actors)
                and legacy._same_entity(e.get("object"), a.get("object"), objects)
                and (family != "intentional_errors" or e.get("audio") == a.get("audio")))

    def interval(e, a):
        return legacy._interval_accepted(
            legacy._seconds(e, fps, "start") * fps, legacy._seconds(e, fps, "end") * fps,
            legacy._seconds(a, fps, "start") * fps, legacy._seconds(a, fps, "end") * fps,
            tolerance)[0]

    fingerprint: dict[str, Any] = {}
    for family in FAMILIES:
        truth = scene.get(family, [])
        predictions = reconstruction.get(family, [] if family != "qa" else {})
        if family == "qa":
            predictions = predictions if isinstance(predictions, dict) else {}
            fingerprint[family] = [legacy.normalize_answer(predictions.get(q["id"], ""))
                                   for q in truth]
            continue
        predictions = predictions if isinstance(predictions, list) else []
        values = []
        for expected in truth:
            matches = []
            for actual in predictions:
                if not isinstance(actual, dict):
                    continue
                try:
                    if family in {"events", "audio_events", "intentional_errors"}:
                        credit = float(point(expected, actual, family))
                    elif family == "shots":
                        credit = float(abs(legacy._frame(expected, fps, "start_frame", "start")
                                           - legacy._frame(actual, fps, "start_frame", "start")) <= 2)
                    elif family == "dialogue":
                        credit = (legacy._lexical_credit(expected.get("text", ""), actual.get("text", ""))
                                  if interval(expected, actual) else 0.0)
                    else:
                        credit = float(interval(expected, actual) and
                            legacy.normalize_answer(expected.get("text", "")) ==
                            legacy.normalize_answer(actual.get("text", "")))
                    matches.append(credit)
                except (ArithmeticError, KeyError, TypeError, ValueError):
                    continue
            values.append(round(max(matches, default=0.0), 12))
        fingerprint[family] = values
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score_reconstruction(scene: Mapping[str, Any], reconstruction: Any,
                         cost: Mapping[str, Any] | None = None, *,
                         q_min: Mapping[int, float] | None = None,
                         cost_lambda: float = legacy.LAMBDA) -> dict[str, Any]:
    if scene.get("schema_version") != "2.0":
        raise ValueError("scorer2.0.0 requires temporal scene schema2.0")
    invalid_reason = None
    try:
        submitted = _bounded(reconstruction)
        temporal = _normalize_histories(scene, submitted)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        submitted, temporal = {}, 0.0
        invalid_reason = type(exc).__name__
    report = legacy.score_reconstruction(scene, submitted, cost,
        q_min={1: 0, 2: 0, 3: 0}, cost_lambda=cost_lambda)
    active = {family: weight for family, weight in legacy.WEIGHTS.items() if scene.get(family)}
    denominator = sum(active.values())
    reconstruction_quality = (sum(active[f] * report["family_scores"][f] for f in active) / denominator
                              if denominator else 0.0)
    event_quality = report["family_scores"]["events"]
    # A collection of plausible static facts cannot substitute for the history.
    temporal_quality = temporal * event_quality
    quality = 0.0 if invalid_reason else min(reconstruction_quality, temporal_quality)
    quality = round(quality, 12)
    metrics = reward_metrics(quality, report["cost"]["efficiency_factor"], valid=invalid_reason is None)
    report.update(benchmark_version=SCORER_VERSION, quality=quality, score=metrics["score"],
                  metrics=metrics, semantic_fingerprint=_fingerprint(scene, submitted),
                  active_weights={f: w / denominator for f, w in active.items()},
                  temporal={"history_accuracy": temporal, "event_quality": event_quality,
                            "quality": temporal_quality},
                  reconstruction_quality=round(reconstruction_quality, 12))
    report["gate"].update(q_min=QUALITY_THRESHOLD, passed=metrics["full_reward_eligible"],
                         partial_floor=QUALITY_THRESHOLD, reward_factor=metrics["reward_factor"])
    if invalid_reason:
        report["diagnostics"].append({"family": "reconstruction", "reason": invalid_reason})
    return report
