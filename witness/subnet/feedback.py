"""Public round report projection and model-independent miner reception."""

import asyncio
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Tuple

from .protocol import RoundFeedback, WitnessFeedback


def round_feedback(artifact: dict[str, Any]) -> RoundFeedback:
    """Copy only allowlisted scores; never serialize the private round wholesale."""
    weights = dict(zip(artifact["weights"]["uids"], artifact["weights"]["values"]))
    miners = []
    for uid, miner in artifact["miners"].items():
        scenes = []
        for row in miner["scenes"]:
            status = (row.get("trace_summary") or {}).get("status")
            status = "ok" if row["responded"] else (
                status if status in {"busy", "deadline_exceeded", "error", "degraded"} else "missing")
            scenes.append({
                **{key: row[key] for key in (
                    "scene_id", "tier", "responded", "quality", "efficiency_factor",
                    "family_scores", "cost", "score_before_duplicates", "duplicate_count", "score")},
                "status": status,
                "gate": {"threshold": row["gate"]["threshold"], "passed": row["gate"]["passed"],
                         "partial_floor": row["metrics"]["partial_floor"],
                         "reward_factor": row["metrics"]["reward_factor"]},
            })
        miners.append({
            "uid": int(uid), "hotkey": miner["scenes"][0]["hotkey"],
            **{key: miner[key] for key in ("round_score", "window_score", "window_rounds_observed",
                                          "ema_score", "responded")},
            "weight": weights[int(uid)], "scenes": scenes,
        })
    policy = artifact["weight_policy"]
    return RoundFeedback(
        round_id=artifact["round_id"], validator_hotkey=artifact["validator_hotkey"],
        completed_at=artifact["completed_at"], scorer_version=artifact["scoring_identity"]["version"],
        aggregation={key: artifact["aggregation_identity"][key] for key in (
            "version", "algorithm", "window_rounds", "alpha", "bootstrap")},
        weight_policy=policy["name"], burn_uid=policy["burn_uid"], burn_rate=policy["burn_rate"],
        winner_uid=policy.get("winner_uid"), submission_status=artifact["weight_submission"]["status"],
        miners=miners,
    )


class FeedbackReceiver:
    """Retain one latest report per signed, registered validator, optionally on disk."""

    def __init__(self, *, chain=None, feedback_dir: Path | None = None):
        self.chain = chain
        self.feedback_dir = feedback_dir
        self.latest: dict[str, RoundFeedback] = {}

    async def blacklist(self, synapse: WitnessFeedback) -> Tuple[bool, str]:
        hotkey = str(getattr(synapse.dendrite, "hotkey", "") or "")
        if not hotkey:
            return True, "missing caller hotkey"
        # Header preprocessing does not carry the body; identity binding is
        # checked again in forward, after SDK body/signature verification.
        if self.chain is not None and not self.chain.is_validator(hotkey):
            return True, "caller lacks a registered validator permit"
        return False, "validator caller"

    async def priority(self, synapse: WitnessFeedback) -> float:
        return 0.0

    async def forward(self, synapse: WitnessFeedback) -> WitnessFeedback:
        synapse.accepted = False
        blocked, _ = await self.blacklist(synapse)
        hotkey = str(getattr(synapse.dendrite, "hotkey", "") or "")
        if blocked or synapse.report.validator_hotkey != hotkey:
            return synapse
        report = synapse.report.model_copy(deep=True)
        if self.feedback_dir is not None:
            await asyncio.to_thread(self._save, hotkey, report)
        self.latest[hotkey] = report
        synapse.accepted = True
        return synapse

    def _save(self, hotkey: str, report: RoundFeedback) -> None:
        directory = Path(self.feedback_dir)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination = directory / (hashlib.sha256(hotkey.encode()).hexdigest() + ".json")
        fd, temporary = tempfile.mkstemp(prefix=".feedback-", dir=directory)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(report.model_dump_json() + "\n")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
