"""One king-versus-challenger evaluation on videos drawn from the private pool.

Both models answer the same clips on the same hardware. Sampling, the clip
reward and the video/eval aggregation come from ``reward.py`` (``EVAL``): the
seed picks ``EVAL.videos`` videos and every pool window of those videos is
answered; each answer is judged against all of the window's Luna references.
The challenger takes the crown when its eval reward beats the king's by more
than ``superiority_margin``, its eval quality meets ``quality_floor`` and
exceeds ``control_max_quality``, and the fixed controls earn nothing. Without a
king, the first two submissions duel and the better one that clears those bars
is crowned (``first_king``).
"""
from __future__ import annotations

from pathlib import Path
import random

from witness.events import content_hash
from witness.storage import sha256_file, write_private
from .adjudication import JudgeProvider
from .contract import Case, Execution, InfrastructureError, Policy, Reference, Task
from .reward import EVAL, eval_score, judge_clip, reward, video_scores
from .round import _grade, controls


def draw(pool: list[dict], seed: str, videos: int = EVAL.videos) -> list[dict]:
    """Every pool window of ``videos`` videos chosen by the seed."""
    names = sorted({row["video"] for row in pool})
    chosen = set(random.Random(seed).sample(names, videos))
    return sorted((row for row in pool if row["video"] in chosen), key=lambda row: (row["video"], row["index"]))


def cases(rows: list[dict], policy: Policy, seed: str) -> list[tuple[Case, list[Reference]]]:
    result = []
    for row in rows:
        references = [Reference.model_validate_json(Path(path).read_text()) for path in row["reference_paths"]]
        if (sha256_file(Path(row["media_path"])) != row["clip_sha256"] or
                any(ref.clip_sha256 != row["clip_sha256"] or ref.kind != policy.reference_kind
                    or ref.duration != references[0].duration for ref in references)):
            raise InfrastructureError("pool_reference_binding_failed")
        task = Task(id="task_" + content_hash({"seed": seed, "clip": row["clip_sha256"]})[:32],
                    clip_sha256=row["clip_sha256"], duration=references[0].duration)
        case = Case(task=task, source_group=row["video"], source_id=row["file"], source_sha256=row["clip_sha256"],
                    source_path=row["media_path"], source_start=0., media_path=row["media_path"],
                    reference=references[0])
        result.append((case, references))
    return result


def grade(case: Case, references: list[Reference], execution: Execution, model_id: str, *,
          policy: Policy, judge: JudgeProvider, root: Path) -> dict:
    """One model's answer to one clip: validity, clip score against every reference and reward."""
    if (execution.checkpoint_hash != model_id or execution.runtime_hash != policy.runtime_hash
            or execution.preprocessing_hash != policy.preprocessing_hash or execution.clip_sha256 != case.task.clip_sha256):
        raise InfrastructureError("execution_binding_failed")
    graded, response = _grade(execution, model_id, case, policy)
    if not graded["valid"]:
        return {**graded, "video": case.source_group, "response": execution.raw, "latency_s": execution.elapsed_s}
    measured, _ = judge_clip(references, response, judge=judge, policy=policy, root=root)
    return {**graded, "video": case.source_group, "score": measured,
            "response": response.model_dump(), "latency_s": execution.elapsed_s,
            **reward(measured["quality"], execution.elapsed_s, case.task.duration, policy, valid=True)}


def decide(receipts: list[dict], king: str | None, challenger: str, policy: Policy) -> dict:
    models = ([king] if king else []) + [challenger]
    summary = {}
    for model_id in models:
        clips = [row["grades"][model_id] for row in receipts]
        videos = video_scores(clips)
        summary[model_id] = {**eval_score(videos), "per_video": videos,
                             "invalid": sum(not clip["valid"] for clip in clips)}
    controls_quality = {name: eval_score(video_scores([{"video": row["video"], "quality": row["controls"][name],
                                                        "reward": 0.} for row in receipts]))["quality"]
                        for name in receipts[0]["controls"]}
    controls_pass = all(value <= policy.control_max_quality for value in controls_quality.values())
    # A king must at least beat what the empty and generic controls may earn.
    challenger_ok = (controls_pass and summary[challenger]["quality"] >= policy.quality_floor
                     and summary[challenger]["quality"] > policy.control_max_quality)
    delta = summary[challenger]["reward"] - summary[king]["reward"] if king else None
    crowned = challenger_ok and (king is None or delta > policy.superiority_margin)
    return {"summary": summary, "controls": {"quality": controls_quality, "pass": controls_pass},
            "reward_delta": delta, "crowned": crowned}


def first_king(report: dict, first: str, second: str, policy: Policy) -> str | None:
    """The first two submissions: the better eval reward among those that clear the quality bars.

    A tie goes to the earlier submission. With clean controls required and
    neither above the bars, nobody is crowned.
    """
    if not report["controls"]["pass"]:
        return None
    summary = report["summary"]
    eligible = [model_id for model_id in (first, second)
                if summary[model_id]["quality"] >= policy.quality_floor
                and summary[model_id]["quality"] > policy.control_max_quality]
    return max(eligible, key=lambda model_id: (summary[model_id]["reward"], model_id == first)) if eligible else None


def run_duel(*, pool: list[dict], king: str | None, challenger: str, seed: str, policy: Policy,
             runner, judge: JudgeProvider, root: Path) -> dict:
    """``runner(models, tasks, paths) -> {model_id: {task_id: Execution}}`` runs every model on every clip."""
    selected = cases(draw(pool, seed), policy, seed)
    models = ([king] if king else []) + [challenger]
    executions = runner(models, [case.task for case, _ in selected], [Path(case.media_path) for case, _ in selected])
    receipts = []
    for case, references in selected:
        grades = {}
        for model_id in models:
            execution = executions.get(model_id, {}).get(case.task.id)
            if not isinstance(execution, Execution):
                raise InfrastructureError("missing_execution_receipt")
            grades[model_id] = grade(case, references, execution, model_id, policy=policy, judge=judge, root=root)
        control = {name: judge_clip(references, response, judge=judge, policy=policy, root=root)[0]["quality"]
                   for name, response in controls(case.task.duration).items()}
        receipts.append({"video": case.source_group, "task": case.task.model_dump(), "grades": grades,
                         "controls": control})
    report = {"schema_version": "witness-duel-report-2", "seed": seed, "policy_hash": policy.identity,
              "eval": EVAL.identity, "judge_id": policy.judge_id, "king": king, "challenger": challenger,
              "videos": sorted({case.source_group for case, _ in selected}),
              "clips": [case.task.clip_sha256 for case, _ in selected],
              **decide(receipts, king, challenger, policy), "evidence_hash": content_hash(receipts)}
    write_private(root / "receipts.json", receipts)
    write_private(root / "duel-report.json", report)
    return report
