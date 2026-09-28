"""Claim precision against reference facts and weighted per-modality coverage (reward lives in reward.py).

Quality is the F1 of two numbers:

* precision over all original claims. The judge splits each claim into its
  atomic propositions; a claim earns (supported - penalty * contradicted) /
  propositions, so a true claim with one unverifiable detail keeps most of its
  credit. The total is floored at zero, and supported credit is capped at one
  per distinct fact covered, so repeating or paraphrasing a truth adds nothing;
* recall averaged over the modalities the annotators labeled, each fact
  weighted by salience. Claims in another modality only lower precision, and a
  reviewed fact never adds a modality to the average.

Time is a gate, not a multiplier: a claim may cite a fact of the same modality
when most of the claim lies inside the fact's interval (widened by the policy
tolerance). A whole-clip claim therefore cannot cite a short event.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from witness.events import StrictModel
from .contract import MODALITIES, Claim, Policy, Reference, Response


Status = Literal["supported", "contradicted", "unresolved"]


class Part(StrictModel):
    """One atomic proposition of a claim and its verdict."""
    text: str = Field(min_length=1, max_length=400)
    status: Status
    fact_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def cited(self):
        if self.status == "supported" and not self.fact_ids:
            raise ValueError("supported_part_without_fact")
        if self.status == "unresolved" and self.fact_ids:
            raise ValueError("unresolved_part_cites_fact")
        return self


class Decision(StrictModel):
    prediction_id: str
    parts: list[Part] = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1, max_length=1000)

    @classmethod
    def whole(cls, claim: Claim, status: Status, fact_ids: list[str], reason: str) -> "Decision":
        """A verdict on the claim as one proposition."""
        return cls(prediction_id=claim.id, reason=reason,
                   parts=[Part(text=claim.description[:400], status=status, fact_ids=fact_ids)])

    def share(self, status: Status) -> float:
        return sum(part.status == status for part in self.parts) / len(self.parts)

    @property
    def status(self) -> Status:
        """Contradicted if any part is, supported only if every part is."""
        if any(part.status == "contradicted" for part in self.parts):
            return "contradicted"
        return "supported" if all(part.status == "supported" for part in self.parts) else "unresolved"

    @property
    def fact_ids(self) -> list[str]:
        return list(dict.fromkeys(fact_id for part in self.parts for fact_id in part.fact_ids))


class Assessment(StrictModel):
    judge_id: str
    decisions: list[Decision]
    review_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def compatible(prediction: Claim, fact: Claim, policy: Policy) -> bool:
    tolerance = policy.temporal_tolerance_s
    inside = min(prediction.end, fact.end + tolerance) - max(prediction.start, fact.start - tolerance)
    return (prediction.modality == fact.modality
            and inside >= policy.min_claim_overlap * (prediction.end - prediction.start))


def iou(a: Claim, b: Claim) -> float:
    overlap = max(0., min(a.end, b.end) - max(a.start, b.start))
    return overlap / (max(a.end, b.end) - min(a.start, b.start))


def validate_assessment(reference: Reference, response: Response,
                        assessment: Assessment, policy: Policy) -> None:
    predictions = {claim.id: claim for claim in response.claims}
    if (len(assessment.decisions) != len(predictions)
            or {item.prediction_id for item in assessment.decisions} != set(predictions)):
        raise ValueError("missing_or_duplicate_claim_decision")
    facts = {fact.claim.id: fact.claim for fact in reference.facts}
    for item in assessment.decisions:
        for part in item.parts:
            if len(part.fact_ids) != len(set(part.fact_ids)) or len(part.fact_ids) > policy.max_facts_per_claim:
                raise ValueError("invalid_fact_citation_count")
            for fact_id in part.fact_ids:
                if fact_id not in facts:
                    raise ValueError("unknown_reference_fact")
                if not compatible(predictions[item.prediction_id], facts[fact_id], policy):
                    raise ValueError("fact_wrong_interval_or_modality")
            if part.status == "contradicted" and not part.fact_ids and not assessment.review_hash:
                raise ValueError("uncited_contradiction_requires_review")


def score(reference: Reference, response: Response, assessment: Assessment, policy: Policy) -> dict:
    validate_assessment(reference, response, assessment, policy)
    decisions = {item.prediction_id: item for item in assessment.decisions}
    facts = {fact.claim.id: fact for fact in reference.facts}
    weight = {fact_id: 1. if fact.salience == "core" else policy.detail_weight
              for fact_id, fact in facts.items()}
    covered = {fact_id for item in decisions.values() for part in item.parts
               if part.status == "supported" for fact_id in part.fact_ids}
    counts = {status: sum(item.share(status) for item in decisions.values())
              for status in ("supported", "contradicted", "unresolved")}
    credit = min(counts["supported"], len(covered)) - policy.contradiction_penalty * counts["contradicted"]
    precision = max(0., credit) / len(response.claims) if response.claims else 0.
    labeled = {fact.claim.modality for fact in reference.facts if fact.origin == "annotator"}
    modalities = {}
    for modality in MODALITIES:
        own = [fact_id for fact_id, fact in facts.items() if fact.claim.modality == modality]
        claims = [claim.id for claim in response.claims if claim.modality == modality]
        modalities[modality] = {
            "in_reference": modality in labeled, "reference_facts": len(own), "predicted": len(claims),
            "recall": (sum(weight[f] for f in own if f in covered) / sum(weight[f] for f in own))
            if modality in labeled else None,
            **{status: sum(decisions[c].status == status for c in claims)
               for status in ("supported", "contradicted", "unresolved")}}
    recalls = [part["recall"] for part in modalities.values() if part["in_reference"]]
    recall = sum(recalls) / len(recalls)
    quality = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    predictions = {claim.id: claim for claim in response.claims}
    overlaps = [iou(predictions[item.prediction_id], facts[fact_id].claim)
                for item in decisions.values() for part in item.parts if part.status == "supported"
                for fact_id in part.fact_ids]
    return {"quality": quality, "precision": precision, "recall": recall, "modalities": modalities,
            "claims": len(response.claims), **counts,
            "missed_core": sum(fact.salience == "core" and fact_id not in covered
                               for fact_id, fact in facts.items()),
            "temporal_iou": sum(overlaps) / len(overlaps) if overlaps else None}
