"""One evaluation round: the champion and every challenger answer the same clips.

Each round consumes one fresh batch of human-labeled source groups; the
statistical alpha is split across challengers up front. A challenger is promoted when its paired reward gain is credibly above
the superiority margin and its quality is not credibly worse. Fixed controls
(empty and generic answers) are scored every case; if they earn quality the
judge or references are broken and nobody is promoted.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from witness.events import content_hash
from witness.storage import sha256_file, store_immutable
from .adjudication import JudgeProvider, MediaReviewer, adjudicate, assess
from .contract import Case, Claim, Execution, InfrastructureError, Policy, Reference, Response, parse_response
from .lineage import Catalog, Draw
from .media import PREPROCESSOR_ID, render_clip
from .reward import reward
from .scoring import Assessment, score


def controls(duration: float) -> dict[str, Response]:
    """Answers that know nothing about the clip; any real benchmark scores them near zero."""
    generic = [("visual", "Something", "Something is visible on screen."),
               ("speech", "A voice", "Someone speaks."),
               ("text", "Text", "Some text is shown."),
               ("sound", "Audio", "Sound is audible.")]
    return {"empty": Response(claims=[]),
            "generic": Response(claims=[Claim(id=f"generic_{modality}", start=0., end=duration,
                                              subject=subject, description=description, modality=modality)
                                        for modality, subject, description in generic])}


def bootstrap_lower(deltas: list[float], alpha: float, seed: str) -> float:
    if not deltas:
        raise ValueError("empty_paired_deltas")
    rng = np.random.default_rng(int(seed[:16], 16))
    values = np.asarray(deltas, dtype=np.float64)
    samples = np.empty(100_000, dtype=np.float64)
    for start in range(0, len(samples), 1000):
        count = min(1000, len(samples) - start)
        picks = rng.integers(0, len(values), size=(count, len(values)))
        samples[start:start + count] = values[picks].mean(axis=1)
    return float(np.quantile(samples, alpha, method="lower"))


def decide(pairs: list[tuple[dict, dict]], *, policy: Policy, alpha: float, seed: str) -> dict:
    """Paired, group-level decision for one challenger against the champion."""
    quality_deltas = [b["quality"] - a["quality"] for a, b in pairs]
    reward_deltas = [b["reward"] - a["reward"] for a, b in pairs]
    quality_lower = bootstrap_lower(quality_deltas, alpha, content_hash({"quality": seed}))
    reward_lower = bootstrap_lower(reward_deltas, alpha, content_hash({"reward": seed}))
    challenger_quality = sum(b["quality"] for _, b in pairs) / len(pairs)
    promoted = (challenger_quality >= policy.quality_floor
                and quality_lower >= -policy.quality_margin
                and reward_lower > policy.superiority_margin)
    return {"promoted": promoted, "alpha": alpha, "groups": len(pairs),
            "quality_lower": quality_lower, "reward_lower": reward_lower,
            "challenger_quality": challenger_quality,
            "challenger_reward": sum(b["reward"] for _, b in pairs) / len(pairs)}


def _validate_case(case: Case, catalog: Catalog, private_root: Path, policy: Policy) -> None:
    if sha256_file(Path(case.media_path)) != case.task.clip_sha256:
        raise InfrastructureError("selected_media_changed")
    source = {item.id: item for item in catalog.sources}.get(case.source_id)
    if (source is None or case.source_id not in dict(catalog.eval_groups()).get(case.source_group, [])
            or source.original_sha256 != case.source_sha256
            or sha256_file(Path(case.source_path)) != case.source_sha256):
        raise InfrastructureError("source_lineage_binding_failed")
    if policy.preprocessing_hash != PREPROCESSOR_ID:
        raise InfrastructureError("wrong_preprocessing_profile")
    with tempfile.TemporaryDirectory(dir=private_root) as temporary:
        expected = Path(temporary) / "reconstructed.mp4"
        if render_clip(Path(case.source_path), expected, start=case.source_start,
                       duration=case.task.duration) != case.task.clip_sha256:
            raise InfrastructureError("clip_not_derived_from_committed_source")


def _grade(execution: Execution, model_id: str, case: Case, policy: Policy) -> tuple[dict, Response]:
    """Bind an execution receipt to this case and parse it; invalid answers score zero."""
    if (execution.model_id != model_id or execution.checkpoint_hash != model_id
            or execution.task_id != case.task.id or execution.clip_sha256 != case.task.clip_sha256
            or execution.runtime_hash != policy.runtime_hash
            or execution.preprocessing_hash != policy.preprocessing_hash):
        raise InfrastructureError("execution_binding_failure")
    if execution.status == "infra_error":
        raise InfrastructureError("runner_infrastructure_failure")
    valid = (execution.status == "ok" and execution.elapsed_s <= policy.deadline_s(case.task.duration)
             and execution.audio_tokens > 0 and execution.video_tokens > 0)
    response, error = Response(claims=[]), None
    if valid:
        try:
            response = parse_response(execution.raw, case.task.duration)
        except (ValueError, TypeError) as exc:
            valid, error = False, type(exc).__name__
    grade = {"valid": valid, "error": error, "execution": execution.model_dump()}
    grade.update(reward(0., execution.elapsed_s, case.task.duration, policy, valid=False))
    return grade, response


def _finish(grade: dict, reference: Reference, response: Response, assessment: Assessment,
            case: Case, policy: Policy) -> dict:
    if not grade["valid"]:
        return grade
    measured = score(reference, response, assessment, policy)
    return {**grade, "score": measured,
            **reward(measured["quality"], grade["execution"]["elapsed_s"], case.task.duration, policy, valid=True)}


def _execute(case: Case, models: list[str], runner, ordinal: int) -> dict[str, Execution]:
    # The runner receives only model identity, task and MP4 path. Rotate call order.
    shift = ordinal % len(models)
    executions = {}
    for model_id in models[shift:] + models[:shift]:
        execution = runner(model_id, case.task, Path(case.media_path))
        if not isinstance(execution, Execution):
            raise InfrastructureError("invalid_execution_receipt")
        executions[model_id] = execution
    return executions


def _receipt(case: Case, executions: dict[str, Execution], models: list[str], *, policy: Policy,
             judge: JudgeProvider, reviewer: MediaReviewer | None, private_root: Path, ordinal: int) -> dict:
    """Everything after inference; provider answers come from the private cache when present."""
    graded = {model_id: _grade(executions[model_id], model_id, case, policy) for model_id in models}
    reference, assessments, review = adjudicate(
        case, [graded[model_id][1] for model_id in models], policy=policy, judge=judge,
        reviewer=reviewer, private_root=private_root)
    control_scores = {name: score(reference, response, assess(judge, reference, response, policy, private_root),
                                  policy)["quality"]
                      for name, response in controls(case.task.duration).items()}
    return {"ordinal": ordinal, "source_group": case.source_group, "task": case.task.model_dump(),
            "original_reference_hash": content_hash(case.reference.model_dump()),
            "final_reference": reference.model_dump(), "review": review, "controls": control_scores,
            "profile": {"policy": policy.identity, "judge": judge.identity},
            "assessments": {model_id: assessment.model_dump() for model_id, assessment in zip(models, assessments)},
            "grades": {model_id: _finish(graded[model_id][0], reference, graded[model_id][1], assessment, case, policy)
                       for model_id, assessment in zip(models, assessments)}}


def run_round(*, cases: list[Case], catalog: Catalog, draw: Draw, policy: Policy,
              hotkeys: dict[str, str], runner, judge: JudgeProvider,
              reviewer: MediaReviewer | None, private_root: Path) -> dict:
    commit = draw.commit
    champion, challengers = commit.champion_hash, list(commit.challenger_hashes)
    if (len(cases) != policy.confirmation_size or [case.source_group for case in cases] != draw.groups
            or len({case.task.clip_sha256 for case in cases}) != len(cases)
            or commit.policy_hash != policy.identity or commit.catalog_hash != catalog.identity
            or set(hotkeys) != {champion, *challengers}):
        raise ValueError("round_draw_or_model_mismatch")
    draw_hash = content_hash(draw.model_dump())
    active = list(challengers)
    screened: dict[str, dict] = {}
    receipts = []
    for ordinal, case in enumerate(cases):
        # Screened-out challengers stop; the champion always runs every group
        # because the quality floor and controls need the full sample.
        _validate_case(case, catalog, private_root, policy)
        models = [champion, *active]
        result_path = private_root / "results" / f"{ordinal:03d}.json"
        start_path = private_root / "started" / f"{ordinal:03d}.json"
        options = {"policy": policy, "judge": judge, "reviewer": reviewer,
                   "private_root": private_root, "ordinal": ordinal}
        if result_path.exists():
            # Resume re-derives the whole receipt from saved executions and caches.
            receipt = json.loads(result_path.read_text())
            if sorted(receipt.get("grades", {})) != sorted(models):
                raise InfrastructureError("saved_case_models_changed")
            executions = {model_id: Execution.model_validate(row["execution"])
                          for model_id, row in receipt["grades"].items()}
            if _receipt(case, executions, models, **options) != receipt:
                raise InfrastructureError("saved_case_receipt_changed")
        else:
            if start_path.exists():
                raise InfrastructureError("unknown_inference_outcome_requires_reconciliation")
            store_immutable(start_path, {"task": case.task.model_dump(), "models": models,
                                         "policy_hash": policy.identity})
            receipt = _receipt(case, _execute(case, models, runner, ordinal), models, **options)
            store_immutable(result_path, receipt)
        receipts.append(receipt)
        if len(receipts) == policy.screen_size:
            for model_id in list(active):
                rows = [row["grades"][model_id] for row in receipts]
                invalid = sum(not row["valid"] for row in rows)
                quality = sum(row["quality"] for row in rows) / len(rows)
                if invalid > policy.screen_max_invalid or quality < policy.screen_min_quality:
                    screened[model_id] = {"invalid": invalid, "quality": quality}
                    active.remove(model_id)

    def mean(model_id: str, key: str) -> float:
        rows = [row["grades"][model_id][key] for row in receipts if model_id in row["grades"]]
        return sum(rows) / len(rows)

    control_quality = {name: sum(row["controls"][name] for row in receipts) / len(receipts)
                       for name in receipts[0]["controls"]}
    controls_pass = all(value <= policy.control_max_quality for value in control_quality.values())
    alpha = policy.alpha / len(challengers)
    decisions = {model_id: {"status": "screen_reject", "promoted": False, **screened[model_id]}
                 for model_id in screened}
    for model_id in active:
        pairs = [(row["grades"][champion], row["grades"][model_id]) for row in receipts]
        decisions[model_id] = {"status": "finished", **decide(
            pairs, policy=policy, alpha=alpha, seed=content_hash({"draw": draw_hash, "model": model_id}))}
        decisions[model_id]["promoted"] &= controls_pass
    promoted = [model_id for model_id in challengers if decisions[model_id]["promoted"]]
    champion_quality = mean(champion, "quality")
    if promoted:
        winner = max(promoted, key=lambda model_id: decisions[model_id]["challenger_reward"])
    elif controls_pass and champion_quality >= policy.quality_floor:
        winner = champion
    else:
        winner = None  # nobody qualifies
    report = {"schema_version": "witness-round-report-3", "draw_hash": draw_hash,
              "policy_hash": policy.identity, "runtime_hash": policy.runtime_hash,
              "judge_id": policy.judge_id, "reviewer_id": policy.reviewer_id,
              "preprocessing_hash": policy.preprocessing_hash, "hotkeys": hotkeys,
              "champion": champion, "groups_evaluated": len(receipts),
              "summary": {model_id: {"quality": mean(model_id, "quality"), "reward": mean(model_id, "reward"),
                                     "invalid": sum(not row["grades"][model_id]["valid"]
                                                    for row in receipts if model_id in row["grades"])}
                          for model_id in [champion, *challengers]},
              "controls": {"quality": control_quality, "pass": controls_pass},
              "decisions": decisions, "winner": winner,
              "evidence_hash": content_hash(receipts)}
    store_immutable(private_root / "round-report.json", report)  # a rerun must reproduce it exactly
    return report
