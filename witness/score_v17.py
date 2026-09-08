"""Candidate scoring v1.7: bounded timestamps and multiplicity-aware matching.

Independent snapshot derived from candidate v1.6; prior scoring versions
remain unchanged for reproducible historical reports. Corpus schema 1.5 remains readable for development rescoring.
This module does not certify a corpus or authorize a launch.
Frozen origin SHA256: ecdcfafa893b5621fe082bd98533385051375785c6ebabe08a02a7a9368e1a93"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from witness.tools.metering import visual_token_cost

DEFAULT_Q_MIN = {1: 0.70, 2: 0.65, 3: 0.60}
EVENT_TOLERANCE_SECONDS = {1: 0.50, 2: 0.375, 3: 0.25}
LAMBDA = 0.30
WEIGHTS = {
    "events": 0.25,
    "dialogue": 0.20,
    "shots": 0.10,
    "on_screen_text": 0.05,
    "audio_events": 0.05,
    "intentional_errors": 0.15,
    "qa": 0.20,
}


_ARTICLES = {"a", "an", "the"}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_answer(value: Any) -> str:
    """Apply deterministic SQuAD-style normalization to a scalar answer."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    words = [word for word in _TOKEN_RE.findall(text) if word not in _ARTICLES]
    return " ".join(words)


class _EntityResolver:
    """Resolve observable scene descriptions without guessing ambiguous aliases."""

    def __init__(self, entities: Sequence[Mapping[str, Any]]) -> None:
        self._ids = {str(entity["id"]) for entity in entities}
        aliases: dict[str, set[str]] = {}
        for entity in entities:
            entity_id = str(entity["id"])
            for field in ("visual_description", "color", "shape", "kind"):
                value = entity.get(field)
                if value is not None:
                    aliases.setdefault(normalize_answer(value), set()).add(entity_id)
        self._aliases = aliases

    def resolve(self, value: Any) -> str | None:
        if value is None:
            return None
        raw = str(value)
        if raw in self._ids:
            return raw
        candidates = self._aliases.get(normalize_answer(raw), set())
        return next(iter(candidates)) if len(candidates) == 1 else None


def _same_entity(expected: Any, actual: Any, resolver: _EntityResolver) -> bool:
    if expected is None or actual is None:
        return expected is actual
    expected_id = resolver.resolve(expected)
    return expected_id is not None and resolver.resolve(actual) == expected_id


def _prepare_items(
    value: Any,
    name: str,
    validator: Callable[[Mapping[str, Any]], str | None],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return scoreable items and the number of malformed false positives."""
    if value is None:
        return [], 0
    if not isinstance(value, list):
        diagnostics.append({"family": name, "index": None, "reason": "expected an array"})
        return [], 1
    valid: list[dict[str, Any]] = []
    invalid_count = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            reason = "expected an object"
        else:
            try:
                reason = validator(item)
            except (ArithmeticError, OverflowError, TypeError, ValueError):
                reason = "invalid field type or value"
        if reason is not None:
            diagnostics.append({"family": name, "index": index, "reason": reason})
            invalid_count += 1
        else:
            valid.append(item)
    return valid, invalid_count


def _frame(item: Mapping[str, Any], fps: int, frame_key: str = "frame", time_key: str = "t") -> int:
    if frame_key in item:
        value = item[frame_key]
    elif time_key in item:
        if isinstance(item[time_key], bool):
            raise ValueError("timestamps must be finite numbers")
        value = float(item[time_key]) * fps
    else:
        raise ValueError(f"item needs {frame_key!r} or {time_key!r}")
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("timestamps must be finite numbers")
    return int(round(float(value)))


def _point_reason(item: Mapping[str, Any], fps: int, duration_frames: int) -> str | None:
    try:
        frame = _frame(item, fps)
    except (ArithmeticError, OverflowError, TypeError, ValueError):
        return "missing or invalid frame/t"
    if not 0 <= frame < duration_frames:
        return "timestamp outside video"
    return None


def _interval_reason(
    item: Mapping[str, Any], fps: int, duration_frames: int, *, seconds: bool = False
) -> str | None:
    try:
        if seconds:
            start = _seconds(item, fps, "start") * fps
            end = _seconds(item, fps, "end") * fps
        else:
            start = _frame(item, fps, "start_frame", "start")
            end = _frame(item, fps, "end_frame", "end")
    except (ArithmeticError, OverflowError, TypeError, ValueError):
        return "missing or invalid interval"
    if start < 0 or end <= start:
        return "negative or non-increasing interval"
    if end > duration_frames:
        return "interval outside video"
    return None


def _string_fields_reason(item: Mapping[str, Any], required: Sequence[str]) -> str | None:
    if any(not isinstance(item.get(field), str) or not item[field].strip() for field in required):
        return f"required string fields: {', '.join(required)}"
    for field in ("actor", "object"):
        if field in item and item[field] is not None and not isinstance(item[field], str):
            return f"{field} must be a string"
    return None


def _seconds(item: Mapping[str, Any], fps: int, key: str) -> float:
    if key in item:
        value = item[key]
    elif f"{key}_frame" in item:
        if isinstance(item[f"{key}_frame"], bool):
            raise ValueError("timestamps must be finite numbers")
        value = float(item[f"{key}_frame"]) / fps
    else:
        raise ValueError(f"dialogue item needs {key!r} or {key + '_frame'!r}")
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("timestamps must be finite numbers")
    return float(value)


def _f_score(
    matches: float,
    truth_count: int,
    prediction_count: int,
    *,
    family: str,
    beta: float = 1.0,
) -> float:
    if truth_count == prediction_count == 0:
        return 1.0
    if matches == 0:
        return 0.0
    precision = matches / prediction_count
    recall = matches / truth_count
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def _maximum_matches(
    truth: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    candidate: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[bool, float]],
) -> int:
    """Maximum-cardinality bipartite matching with stable nearest-first edges."""
    edges: list[list[int]] = []
    for truth_index, truth_item in enumerate(truth):
        eligible: list[tuple[float, int]] = []
        for prediction_index, prediction_item in enumerate(predicted):
            accepted, distance = candidate(truth_item, prediction_item)
            if accepted:
                eligible.append((distance, prediction_index))
        edges.append([index for _, index in sorted(eligible, key=lambda pair: (pair[0], pair[1]))])

    prediction_owner: dict[int, int] = {}

    def augment(truth_index: int, seen: set[int]) -> bool:
        for prediction_index in edges[truth_index]:
            if prediction_index in seen:
                continue
            seen.add(prediction_index)
            owner = prediction_owner.get(prediction_index)
            if owner is None or augment(owner, seen):
                prediction_owner[prediction_index] = truth_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(truth)))


def _event_score(
    truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    fps: int,
    tier: int,
    actors: _EntityResolver,
    objects: _EntityResolver,
    prediction_count: int,
) -> float:
    tolerance = EVENT_TOLERANCE_SECONDS[tier] * fps

    def candidate(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> tuple[bool, float]:
        identity = (
            actual.get("action") == expected.get("action")
            and _same_entity(expected.get("actor"), actual.get("actor"), actors)
            and _same_entity(expected.get("object"), actual.get("object"), objects)
        )
        distance = abs(_frame(expected, fps) - _frame(actual, fps))
        return identity and distance <= tolerance, distance

    matches = _maximum_matches(truth, predicted, candidate)
    return _f_score(
        matches,
        len(truth),
        prediction_count,
        family="events",
    )


def _interval_accepted(
    expected_start: float,
    expected_end: float,
    actual_start: float,
    actual_end: float,
    tolerance: float,
) -> tuple[bool, float]:
    """Accept sufficient overlap or two independently close boundaries."""

    distance = max(abs(expected_start - actual_start), abs(expected_end - actual_end))
    overlap = max(0.0, min(expected_end, actual_end) - max(expected_start, actual_start))
    union = max(expected_end, actual_end) - min(expected_start, actual_start)
    iou = overlap / union if union > 0 else 0.0
    return iou >= 0.5 or distance <= tolerance, distance


def _dialogue_score(
    truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    fps: int,
    tier: int,
    prediction_count: int,
) -> float:
    """Interval-matched lines credited by lexical similarity (1 - WER); speaker labels are not scored.

    A miner that returns the validator's degraded transcript with the right timing earns partial
    credit proportional to the words it got right; exact text is not required to match a line.
    """
    tolerance = EVENT_TOLERANCE_SECONDS[tier] * fps
    credits = []
    for expected in truth:
        row = []
        for actual in predicted:
            accepted, distance = _interval_accepted(
                _seconds(expected, fps, "start") * fps,
                _seconds(expected, fps, "end") * fps,
                _seconds(actual, fps, "start") * fps,
                _seconds(actual, fps, "end") * fps,
                tolerance,
            )
            row.append(_lexical_credit(expected.get("text", ""), actual.get("text", "")) if accepted else 0.0)
        credits.append(row)
    credit = _maximum_credit(credits)
    return _f_score(credit, len(truth), prediction_count, family="dialogue")


def _maximum_credit(credits: Sequence[Sequence[float]]) -> float:
    """Maximum-weight one-to-one assignment with zero-credit unmatched rows.

    Rectangular Hungarian assignment uses dummy columns so unmatched lines never
    force an invalid pair. Unlike nearest-first greedy matching, this maximizes
    the lexical credit actually used by the metric, within accepted intervals.
    """
    if not credits or not credits[0]:
        return 0.0
    rows, real_columns = len(credits), len(credits[0])
    if any(len(row) != real_columns for row in credits):
        raise ValueError("credit matrix must be rectangular")
    columns = real_columns + rows
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    owner, previous = [0] * (columns + 1), [0] * (columns + 1)
    for row_index in range(1, rows + 1):
        owner[0] = row_index
        column = 0
        minimum, used = [float("inf")] * (columns + 1), [False] * (columns + 1)
        while True:
            used[column] = True
            current_row = owner[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                weight = credits[current_row - 1][candidate - 1] if candidate <= real_columns else 0.0
                reduced = -weight - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate], previous[candidate] = reduced, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            predecessor = previous[column]
            owner[column] = owner[predecessor]
            column = predecessor
    return sum(credits[owner[column] - 1][column - 1]
               for column in range(1, real_columns + 1) if owner[column])


def _lexical_credit(expected_text: Any, actual_text: Any) -> float:
    expected = normalize_answer(expected_text).split()
    actual = normalize_answer(actual_text).split()
    if not expected:
        return 1.0 if not actual else 0.0
    return max(0.0, 1.0 - _levenshtein(expected, actual) / len(expected))


def _levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (reference_token != hypothesis_token)))
        previous = current
    return previous[-1]


def _shot_score(
    truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    duration_frames: int,
    invalid_count: int,
    fps: int = 24,
) -> float:
    def boundaries(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if len(items) < 2:
            return []
        # A shot list defines cuts between adjacent entries: the start of each later shot
        # (frames or seconds); end of the previous shot is an accepted fallback.
        values: list[dict[str, Any]] = []
        for item in sorted(items, key=lambda x: _frame(x, fps, "start_frame", "start"))[1:]:
            try:
                frame = _frame(item, fps, "start_frame", "start")
                if 0 <= frame < duration_frames:
                    values.append({"frame": frame, "id": item.get("id")})
            except (TypeError, ValueError, KeyError):
                pass
        return values

    expected_items = boundaries(truth)
    actual_items = boundaries(predicted)
    matches = _maximum_matches(
        expected_items,
        actual_items,
        lambda expected, actual: (
            abs(int(expected["frame"]) - int(actual["frame"])) <= 2,
            abs(int(expected["frame"]) - int(actual["frame"])),
        ),
    )
    return _f_score(
        matches,
        len(expected_items),
        len(actual_items) + invalid_count,
        family="shots",
    )


def _interval_f1(
    truth: list[dict[str, Any]], predicted: list[dict[str, Any]], fps: int, tier: int, prediction_count: int
) -> float:
    tolerance = EVENT_TOLERANCE_SECONDS[tier] * fps

    def candidate(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> tuple[bool, float]:
        identity = (
            normalize_answer(actual.get("text", "")) == normalize_answer(expected.get("text", ""))
        )
        e0, e1 = _frame(expected, fps, "start_frame", "start"), _frame(expected, fps, "end_frame", "end")
        a0, a1 = _frame(actual, fps, "start_frame", "start"), _frame(actual, fps, "end_frame", "end")
        accepted, distance = _interval_accepted(e0, e1, a0, a1, tolerance)
        return identity and accepted, distance

    matches = _maximum_matches(truth, predicted, candidate)
    return _f_score(matches, len(truth), prediction_count, family="on_screen_text")


def _audio_score(
    truth: list[dict[str, Any]], predicted: list[dict[str, Any]], fps: int, tier: int, prediction_count: int
) -> float:
    tolerance = EVENT_TOLERANCE_SECONDS[tier] * fps

    def candidate(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> tuple[bool, float]:
        distance = abs(_frame(expected, fps) - _frame(actual, fps))
        return actual.get("kind") == expected.get("kind") and distance <= tolerance, distance

    matches = _maximum_matches(truth, predicted, candidate)
    return _f_score(matches, len(truth), prediction_count, family="audio_events")


def _error_score(
    truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    fps: int,
    tier: int,
    actors: _EntityResolver,
    objects: _EntityResolver,
    prediction_count: int,
) -> float:
    tolerance = EVENT_TOLERANCE_SECONDS[tier] * fps

    def candidate(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> tuple[bool, float]:
        identity = (
            actual.get("type") == expected.get("type")
            and _same_entity(expected.get("object"), actual.get("object"), objects)
            and _same_entity(expected.get("actor"), actual.get("actor"), actors)
            and actual.get("audio") == expected.get("audio")
        )
        distance = abs(_frame(expected, fps) - _frame(actual, fps))
        return identity and distance <= tolerance, distance

    matches = _maximum_matches(truth, predicted, candidate)
    return _f_score(
        matches,
        len(truth),
        prediction_count,
        family="intentional_errors",
        beta=0.5,
    )


def _qa_score(truth: list[dict[str, Any]], predicted: Any) -> float:
    if not truth:
        return 1.0
    if predicted is None:
        answers: Mapping[str, Any] = {}
    elif isinstance(predicted, dict):
        answers = predicted
    else:
        raise ValueError("reconstruction.qa must be an object keyed by qa id")
    correct = sum(_qa_match(answers.get(item["id"], ""), item["a"]) for item in truth)
    return correct / len(truth)


def _qa_match(answer: Any, truth: Any) -> float:
    """One exact normalized answer; bags of alternative answers get no credit."""
    return float(normalize_answer(answer) == normalize_answer(truth))


def _nonnegative_number(cost: Mapping[str, Any], key: str) -> float:
    value = cost.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"cost.{key} must be a finite non-negative number")
    return float(value)


def score_reconstruction(
    scene: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    cost: Mapping[str, Any] | None = None,
    *,
    q_min: Mapping[int, float] | None = None,
    cost_lambda: float = LAMBDA,
) -> dict[str, Any]:
    """Return a JSON-serializable deterministic score report."""
    tier = int(scene["difficulty"])
    if tier not in EVENT_TOLERANCE_SECONDS:
        raise ValueError(f"unsupported difficulty tier: {tier}")
    fps = int(scene["fps"])
    duration_frames = int(scene["duration_frames"])
    actors = _EntityResolver(list(scene.get("actors", [])))
    objects = _EntityResolver(list(scene.get("objects", [])))
    diagnostics: list[dict[str, Any]] = []
    invalid_reason: str | None = None
    if isinstance(reconstruction, Mapping):
        submitted: Mapping[str, Any] = reconstruction
    else:
        submitted = {}
        invalid_reason = "reconstruction must be a JSON object"
        diagnostics.append({"family": "reconstruction", "index": None, "reason": invalid_reason})

    def point_item(item: Mapping[str, Any], required: Sequence[str]) -> str | None:
        reason = _string_fields_reason(item, required)
        return reason or _point_reason(item, fps, duration_frames)

    def entity_item(item: Mapping[str, Any], required: Sequence[str]) -> str | None:
        reason = point_item(item, required)
        if reason:
            return reason
        for field, resolver in (("actor", actors), ("object", objects)):
            if item.get(field) is not None and resolver.resolve(item[field]) is None:
                return f"unresolvable or ambiguous {field}"
        return None

    events, invalid_events = _prepare_items(
        submitted.get("events"), "events", lambda item: entity_item(item, ("action",)), diagnostics
    )
    dialogue, invalid_dialogue = _prepare_items(
        submitted.get("dialogue"),
        "dialogue",
        lambda item: _string_fields_reason(item, ("speaker", "text"))
        or _interval_reason(item, fps, duration_frames, seconds=True),
        diagnostics,
    )
    shots, invalid_shots = _prepare_items(
        submitted.get("shots"),
        "shots",
        lambda item: _interval_reason(item, fps, duration_frames),
        diagnostics,
    )
    on_screen_text, invalid_text = _prepare_items(
        submitted.get("on_screen_text"),
        "on_screen_text",
        lambda item: _string_fields_reason(item, ("text",))
        or _interval_reason(item, fps, duration_frames),
        diagnostics,
    )
    audio_events, invalid_audio = _prepare_items(
        submitted.get("audio_events"),
        "audio_events",
        lambda item: point_item(item, ("kind",)),
        diagnostics,
    )
    intentional_errors, invalid_errors = _prepare_items(
        submitted.get("intentional_errors"),
        "intentional_errors",
        lambda item: entity_item(item, ("type",)),
        diagnostics,
    )

    raw_qa = submitted.get("qa", {})
    if not isinstance(raw_qa, dict):
        if invalid_reason is None:
            invalid_reason = "reconstruction.qa must be an object keyed by qa id"
        diagnostics.append({"family": "qa", "index": None, "reason": "expected an object keyed by qa id"})
        qa_answers: dict[str, Any] = {}
    else:
        qa_answers = {}
        for qa_id, answer in raw_qa.items():
            valid_scalar = isinstance(answer, (str, int, float, bool))
            valid_number = not isinstance(answer, float) or math.isfinite(answer)
            if not isinstance(qa_id, str) or not valid_scalar or not valid_number:
                diagnostics.append({"family": "qa", "index": str(qa_id), "reason": "answer must be a finite scalar"})
            else:
                qa_answers[qa_id] = answer

    family_scores = {
        "events": _event_score(list(scene.get("events", [])), events, fps, tier, actors, objects, len(events) + invalid_events),
        "dialogue": _dialogue_score(
            list(scene.get("dialogue", [])),
            dialogue,
            fps,
            tier,
            len(dialogue) + invalid_dialogue,
        ),
        "shots": _shot_score(list(scene.get("shots", [])), shots, duration_frames, invalid_shots, fps),
        "on_screen_text": _interval_f1(list(scene.get("on_screen_text", [])), on_screen_text, fps, tier, len(on_screen_text) + invalid_text),
        "audio_events": _audio_score(list(scene.get("audio_events", [])), audio_events, fps, tier, len(audio_events) + invalid_audio),
        "intentional_errors": _error_score(list(scene.get("intentional_errors", [])), intentional_errors, fps, tier, actors, objects, len(intentional_errors) + invalid_errors),
        "qa": _qa_score(list(scene.get("qa", [])), qa_answers),
    }
    quality = sum(WEIGHTS[name] * family_scores[name] for name in WEIGHTS)
    if invalid_reason is not None:
        quality = 0.0
    thresholds = DEFAULT_Q_MIN if q_min is None else q_min
    threshold = float(thresholds[tier])
    passed = quality >= threshold

    cost_input = {} if cost is None else cost
    if not isinstance(cost_input, Mapping):
        raise ValueError("cost must be a JSON object")
    cost_values = {key: _nonnegative_number(cost_input, key) for key in ("visual_tokens", "audio_seconds", "transcript_chars")}
    total_cost = sum(cost_values.values())
    duration_seconds = duration_frames / fps
    tokens_per_frame = visual_token_cost(640, 360)
    cost_ref = math.ceil(duration_seconds) * tokens_per_frame
    cost_ratio = total_cost / cost_ref if cost_ref else 0.0
    efficiency_factor = max(0.0, 1.0 - cost_lambda * cost_ratio)
    final_score = quality * efficiency_factor if passed else 0.0

    report = {
        "schema_version": "1.0",
        "tier": tier,
        "family_scores": {key: round(value, 12) for key, value in family_scores.items()},
        "weights": WEIGHTS.copy(),
        "benchmark_version": "1.7-candidate",
        "false_positive_policy": "all submitted unmatched predictions count",
        "qa_policy": "exact_normalized",
        "quality": round(quality, 12),
        "gate": {"q_min": threshold, "passed": passed},
        "cost": {
            **cost_values,
            "total": round(total_cost, 12),
            "cost_ref": round(cost_ref, 12),
            "ratio": round(cost_ratio, 12),
            "lambda": cost_lambda,
            "efficiency_factor": round(efficiency_factor, 12),
        },
        "score": round(final_score, 12),
        "diagnostics": diagnostics,
    }
    if invalid_reason is not None:
        report["invalid_reason"] = invalid_reason
    return report
