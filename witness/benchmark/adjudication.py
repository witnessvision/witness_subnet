"""Judge every model against the same reference; optionally review novel propositions blind.

A proposition the reference neither supports nor contradicts is ``unresolved``.
When the policy names a reviewer, one bounded batch of such propositions from
all models is checked against the clip without model identity. Verified ones
become shared ``review`` facts and every model is rescored on that reference.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

from pydantic import Field

from witness.events import StrictModel, content_hash
from witness.storage import store_immutable
from .contract import Case, Claim, Fact, Policy, Reference, Response
from .scoring import Assessment, validate_assessment

REVIEW_TIME_LIMIT_S = 300.  # one Codex call over contact sheets


class JudgeProvider(Protocol):
    identity: str

    def assess(self, reference: Reference, response: Response) -> Assessment: ...


class MediaReview(StrictModel):
    reviewer_id: str = Field(min_length=1)
    accessed_media: bool
    heard_audio: bool
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: dict[str, str]


class MediaReviewer(Protocol):
    identity: str

    def resolve(self, clip: Path, claims: list[Claim]) -> MediaReview: ...


def _cached(path: Path, binding: dict, key: str, compute):
    """Reuse a stored provider answer only if it was produced for this exact input."""
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt.get("binding") != binding or receipt.get("output_hash") != content_hash(receipt.get(key)):
            raise ValueError("provider_cache_changed")
        return receipt[key]
    value = compute()
    store_immutable(path, {"binding": binding, key: value, "output_hash": content_hash(value)})
    return value


def assess(judge: JudgeProvider, reference: Reference, response: Response,
            policy: Policy, root: Path) -> Assessment:
    binding = {"judge": judge.identity, "reference": reference.model_dump(),
               "response": response.model_dump()}
    value = _cached(root / "judgments" / (content_hash(binding) + ".json"), binding, "assessment",
                    lambda: judge.assess(reference, response).model_dump())
    result = Assessment.model_validate(value)
    if result.judge_id != judge.identity or result.review_hash is not None:
        raise ValueError("invalid_judge_assessment")
    validate_assessment(reference, response, result, policy)
    return result


def _review(reviewer: MediaReviewer, clip: Path, claims: list[Claim]) -> dict:
    started = time.monotonic()
    try:
        review = reviewer.resolve(clip, claims)
        if time.monotonic() - started > REVIEW_TIME_LIMIT_S:
            raise TimeoutError("media_review_time_limit")
    except (TimeoutError, ValueError, OSError):
        # A failed review grants nothing and is never retried automatically.
        review = MediaReview(reviewer_id=reviewer.identity, accessed_media=False, heard_audio=False,
                             prompt_hash="0" * 64,
                             decisions={claim.id: "unresolved" for claim in claims})
    return review.model_dump()


def _proposition(claim: Claim, text: str) -> Claim:
    """An unresolved part as a reviewable claim with its parent's interval and modality."""
    return claim.model_copy(update={"id": "part", "description": text})


def _claim_key(claim: Claim) -> str:
    return content_hash(claim.model_dump(exclude={"id"}))


def adjudicate(case: Case, responses: list[Response], *, policy: Policy, judge: JudgeProvider,
               reviewer: MediaReviewer | None, private_root: Path) -> tuple[Reference, list[Assessment], dict]:
    if judge.identity != policy.judge_id or (reviewer.identity if reviewer else None) != policy.reviewer_id:
        raise ValueError("judge_or_reviewer_identity_mismatch")
    initial = [assess(judge, case.reference, response, policy, private_root) for response in responses]
    summary = {"reviewed": 0, "unreviewed": 0, "added_facts": 0, "review_hash": None}
    unresolved: dict[str, Claim] = {}
    per_model: list[list[str]] = []
    for response, assessment in zip(responses, initial):
        claims = {claim.id: claim for claim in response.claims}
        own = []
        for decision in assessment.decisions:
            for part in decision.parts:
                if part.status == "unresolved":
                    proposition = _proposition(claims[decision.prediction_id], part.text)
                    key = _claim_key(proposition)
                    unresolved.setdefault(key, proposition)
                    if key not in own:
                        own.append(key)
        per_model.append(own)
    if reviewer is None or not unresolved:
        summary["unreviewed"] = len(unresolved)
        return case.reference, initial, summary
    # Take turns across models in their own claim order, so no answer can
    # crowd the others out of the bounded review batch.
    chosen: list[str] = []
    for turn in range(max(map(len, per_model))):
        for own in per_model:
            if turn < len(own) and own[turn] not in chosen and len(chosen) < policy.novel_review_limit:
                chosen.append(own[turn])
    blind = [unresolved[key].model_copy(update={"id": f"blind_{index}"}) for index, key in enumerate(chosen)]
    binding = {"clip_sha256": case.task.clip_sha256, "policy_hash": policy.identity,
               "reviewer_id": reviewer.identity, "claims": [claim.model_dump() for claim in blind]}
    review_hash = content_hash(binding)
    review = MediaReview.model_validate(_cached(
        private_root / "reviews" / (review_hash + ".json"), binding, "review",
        lambda: _review(reviewer, Path(case.media_path), blind)))
    if review.reviewer_id != reviewer.identity or set(review.decisions) != {claim.id for claim in blind}:
        raise ValueError("invalid_media_review")
    statuses = {}
    for key, claim in zip(chosen, blind):
        status = review.decisions[claim.id]
        if status not in ("supported", "contradicted", "unresolved"):
            raise ValueError("unknown_media_review_status")
        if not review.accessed_media or claim.modality in ("speech", "sound") and not review.heard_audio:
            status = "unresolved"
        statuses[key] = status
    additions = [Fact(claim=unresolved[key].model_copy(update={"id": "review_" + key[:20]}),
                      salience="detail", origin="review")
                 for key in chosen if statuses[key] == "supported"]
    reference = Reference.model_validate({**case.reference.model_dump(),
                                          "facts": [fact.model_dump() for fact in case.reference.facts + additions]})
    final = [assess(judge, reference, response, policy, private_root) for response in responses] if additions else initial
    repaired = []
    for response, assessment in zip(responses, final):
        claims = {claim.id: claim for claim in response.claims}
        rows = []
        for decision in assessment.decisions:
            parts = [part.model_copy(update={"status": "contradicted"})
                     if part.status == "unresolved" and statuses.get(
                         _claim_key(_proposition(claims[decision.prediction_id], part.text))) == "contradicted"
                     else part for part in decision.parts]
            rows.append(decision.model_copy(update={"parts": parts}))
        value = Assessment(judge_id=judge.identity, decisions=rows, review_hash=review_hash)
        validate_assessment(reference, response, value, policy)
        repaired.append(value)
    summary.update(reviewed=len(chosen), unreviewed=len(unresolved) - len(chosen), review_hash=review_hash,
                   added_facts=len(additions))
    return reference, repaired, summary
