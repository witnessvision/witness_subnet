"""Annotation concordance with deterministic one-to-one temporal matching.

Semantic decisions are explicit frozen inputs, never inferred by string overlap.
Missing annotations are not evidence that an event did not physically occur.
"""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from witness.events import (SCORER_VERSION, Seconds, StrictModel, Text,
                            content_hash, scored_text, validate_events)

TOLERANCE_S = 5.0
Relation = Literal["supported", "contradiction", "unbacked", "uncertain"]


class ReferenceEvent(StrictModel):
    timestamp: Seconds
    text: Text


class ReferenceInterval(StrictModel):
    start: Seconds
    end: Seconds
    text: Text

    @model_validator(mode="after")
    def valid_interval(self):
        if self.end <= self.start:
            raise ValueError("reference_interval_must_increase")
        return self


class Reference(StrictModel):
    duration: float = Field(ge=60, le=120)
    events: list[ReferenceEvent | ReferenceInterval] = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def valid_times(self):
        if len({type(e) for e in self.events}) != 1:
            raise ValueError("mixed_reference_time_semantics")
        for e in self.events:
            if isinstance(e, ReferenceEvent) and e.timestamp >= self.duration:
                raise ValueError("reference_outside_clip")
            if isinstance(e, ReferenceInterval) and e.end > self.duration:
                raise ValueError("reference_outside_clip")
        return self


def temporal_error(timestamp: float, reference: ReferenceEvent | ReferenceInterval) -> float:
    """Distance to annotated support; zero inside a human interval, not onset error."""
    if isinstance(reference, ReferenceEvent):
        return abs(timestamp - reference.timestamp)
    return max(reference.start-timestamp, timestamp-reference.end, 0.)


class PairDecision(StrictModel):
    prediction: int = Field(ge=0)
    reference: int = Field(ge=0)
    relation: Relation
    fields: dict[str, Relation]


def candidate_pairs(reference: Reference, predictions) -> list[tuple[int, int]]:
    return [(i, j) for i, p in enumerate(predictions.events)
            for j, r in enumerate(reference.events) if temporal_error(p.timestamp, r) <= TOLERANCE_S]


def score_events(reference: dict, response: dict, decisions: list[dict], *,
                 evaluator_id: str, calibrated: bool = False) -> dict:
    ref = Reference.model_validate(reference)
    pred = validate_events(response, ref.duration)
    expected = set(candidate_pairs(ref, pred))
    pairs = {}
    for raw in decisions:
        d = PairDecision.model_validate(raw)
        key = (d.prediction, d.reference)
        if key not in expected or key in pairs:
            raise ValueError("unexpected_or_duplicate_pair_decision")
        if set(d.fields) != set(scored_text(pred.events[d.prediction])):
            raise ValueError("every_text_field_requires_a_decision")
        # A supported headline cannot conceal an extra contradictory/unbacked detail.
        relations = set(d.fields.values())
        derived = next(r for r in ("contradiction", "uncertain", "unbacked", "supported") if r in relations)
        if d.relation != derived:
            raise ValueError("overall_relation_must_cover_every_field")
        pairs[key] = d
    if set(pairs) != expected:
        raise ValueError("incomplete_semantic_decisions")

    # Iterative augmenting paths give maximum cardinality, independent of response
    # order, without recursion on adversarially large submissions. Temporal order
    # and indices break ties deterministically; temporal error is reported separately.
    edges = {i: sorted([j for (p, j), d in pairs.items() if p == i and d.relation == "supported"],
                       key=lambda j: (temporal_error(pred.events[i].timestamp, ref.events[j]), j))
             for i in range(len(pred.events))}
    seen_events = set()
    for i, event in enumerate(pred.events):
        digest = content_hash(event.model_dump())
        if digest in seen_events:
            edges[i] = []  # Repeating the same event cannot acquire another label.
        seen_events.add(digest)
    matched_ref, matched_pred = {}, {}
    for start in edges:
        queue, seen_p, seen_r, parent = [start], {start}, set(), {}
        free = None
        for p in queue:
            for r in edges[p]:
                if r in seen_r:
                    continue
                seen_r.add(r)
                parent[r] = p
                if r not in matched_ref:
                    free = r
                    break
                next_p = matched_ref[r]
                if next_p not in seen_p:
                    seen_p.add(next_p)
                    queue.append(next_p)
            if free is not None:
                break
        while free is not None:
            p = parent[free]
            previous = matched_pred.get(p)
            matched_pred[p], matched_ref[free] = free, p
            free = previous

    matches = [{"prediction": p, "reference": r,
                "error_s": temporal_error(pred.events[p].timestamp, ref.events[r])}
               for p, r in sorted(matched_pred.items())]
    tp, npred, nref = len(matches), len(pred.events), len(ref.events)
    precision, recall = tp / npred if npred else 0., tp / nref
    diagnostics = {"contradictions": 0, "unbacked": 0, "uncertain": 0,
                   "duplicate_or_competing": 0, "no_temporal_reference": 0}
    for p in edges:
        if p in matched_pred:
            continue
        relations = {d.relation for (i, _), d in pairs.items() if i == p}
        if "supported" in relations:
            key = "duplicate_or_competing"
        elif "contradiction" in relations:
            key = "contradictions"
        elif "uncertain" in relations:
            key = "uncertain"
        elif relations:
            key = "unbacked"
        else:
            key = "no_temporal_reference"
        diagnostics[key] += 1
    # Uncertainty makes quality provisional even if the judge passed calibration.
    unresolved = any(d.relation == "uncertain" for d in pairs.values())
    return {"scorer_version": SCORER_VERSION, "evaluator_id": evaluator_id,
            "reference_hash": content_hash(reference), "response_hash": content_hash(response),
            "decisions_hash": content_hash(decisions), "tolerance_s": TOLERANCE_S,
            "temporal_metric": ("distance_to_human_interval" if isinstance(ref.events[0], ReferenceInterval)
                                else "absolute_point_error"),
            "f1": 2*tp/(npred+nref), "precision": precision, "recall": recall,
            "matched": tp, "predictions": npred, "references": nref, "matches": matches,
            "temporal_error_mean_s": sum(m["error_s"] for m in matches)/tp if tp else None,
            "temporal_errors_s": [m["error_s"] for m in matches], **diagnostics,
            "provisional": not calibrated or unresolved, "weights_enabled": False}
