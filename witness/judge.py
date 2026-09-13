"""Frozen semantic evaluator entrypoint; all inputs and decisions stay private."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from witness.events import content_hash, scored_text, validate_events
from witness.events_evaluation import JUDGE_PROMPT, PROMPT_HASH, judge_response
from witness.providers import ApiText, ApiJudge, load_key
from witness.score_v5_0_0 import Reference, candidate_pairs


@dataclass(frozen=True)
class JudgeConfig:
    judge_provider: str = "saygm"
    judge_model: str = "gpt-5.6-luna"
    judge_effort: str = "low"
    max_tokens: int = 2048

    def build(self, cache, *, budget_path, cache_only=False):
        return ApiJudge(ApiText(self.judge_model, Path(cache), effort=self.judge_effort,
                               max_tokens=self.max_tokens, provider=self.judge_provider,
                               budget_path=budget_path, budget_role="validator", cache_only=cache_only))

    def identity(self, cache, *, budget_path):
        judge = self.build(cache, budget_path=budget_path)
        return {**asdict(self), "evaluator_id": judge.identity, "prompt_hash": PROMPT_HASH,
                "score_version": "5.0.0", "reward_version": "5.1.0"}


def parallel_score(value, judge, calibration_report=None):
    ref = Reference.model_validate(value["reference"])
    pred = validate_events(value["response"], ref.duration)
    inputs = {}
    for i, j in candidate_pairs(ref, pred):
        item = {"narration": ref.events[j].text, "event_fields": scored_text(pred.events[i])}
        inputs[content_hash(item)] = item
    with ThreadPoolExecutor(max_workers=4) as pool:
        decisions = dict(pool.map(lambda item: (item[0], judge(JUDGE_PROMPT, item[1])), inputs.items()))
    result = judge_response(value["reference"], value["response"],
                            lambda prompt, item: decisions[content_hash(item)],
                            evaluator_id=judge.identity, calibration=calibration_report)
    result["execution_policy"] = "identical-independent-pairs-parallel-4"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--provider", choices=["saygm", "openai"], default="saygm")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    load_key(args.env, args.provider)
    judge = JudgeConfig(args.provider, args.model, args.effort).build(args.cache, budget_path=args.budget)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    print(json.dumps(parallel_score(json.loads(args.input.read_text()), judge, calibration)))


if __name__ == "__main__":
    main()
