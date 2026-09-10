"""Explicit full-burn operation without scenes, inference or scoring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .chain import BittensorChainAdapter, WeightSubmission


def burn_state(chain: BittensorChainAdapter, burn_uid: int | None) -> dict[str, Any]:
    substrate = chain.subtensor.substrate
    block_hash = substrate.get_chain_finalised_head()
    block = substrate.get_block_number(block_hash)

    def query(name: str, *params: Any) -> Any:
        return substrate.query("SubtensorModule", name, list(params), block_hash=block_hash).value

    netuid = chain.netuid
    owner = query("SubnetOwner", netuid)
    owner_hotkey = query("SubnetOwnerHotkey", netuid)
    target = query("Uids", netuid, owner_hotkey) if burn_uid is None else burn_uid
    if target is None or query("Owner", query("Keys", netuid, target)) != owner:
        raise ValueError("Burn target must be a registered subnet-owner hotkey")
    if query("RecycleOrBurn", netuid) != "Burn":
        raise ValueError("Subnet is configured to recycle, not burn")
    uid = query("Uids", netuid, chain.validator_hotkey)
    permits = query("ValidatorPermit", netuid)
    if uid is None or uid >= len(permits) or not permits[uid]:
        raise ValueError("Hotkey has no validator permit")
    last_update = query("LastUpdate", netuid)[uid]
    rate_limit = query("WeightsSetRateLimit", netuid)
    active = query("Weights", netuid, uid)
    return {
        "mode": "full_burn", "netuid": netuid, "block": block,
        "block_hash": block_hash, "validator_uid": uid, "burn_uid": target,
        "burn_rate": 1.0, "recycle_or_burn": "Burn",
        "weights": {"uids": [target], "values": [1.0]},
        "weights_applied": len(active) == 1 and list(active[0]) == [target, 65535],
        "blocks_until_submission": max(0, last_update + rate_limit + 1 - block),
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


async def run_full_burn(
    chain: BittensorChainAdapter, *, state_path: Path, burn_uid: int | None = None,
    interval_s: float = 60, once: bool = False,
    set_weights_enabled: bool = True,
) -> None:
    if interval_s <= 0:
        raise ValueError("Full-burn polling interval must be positive")
    if set_weights_enabled and state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("weight_submission", {}).get("status") in {"prepared", "unknown"}:
            raise RuntimeError("Unresolved burn submission; reconcile before restarting")
    while True:
        state = burn_state(chain, burn_uid)
        if not set_weights_enabled:
            state["weight_submission"] = {"status": "disabled"}
            # Preserve any prior ambiguous submission artifact for reconciliation.
            write_state(state_path.with_name("burn-observation.json"), state)
        elif state["blocks_until_submission"]:
            state["weight_submission"] = {"status": "rate_limited"}
            write_state(state_path, state)
        else:
            state["weight_submission"] = WeightSubmission("prepared").as_dict()
            write_state(state_path, state)
            try:
                submission = chain.set_weights(**{
                    "uids": state["weights"]["uids"],
                    "weights": state["weights"]["values"],
                })
                if not isinstance(submission, WeightSubmission):
                    raise TypeError("Missing submission evidence")
            except Exception as exc:
                submission = WeightSubmission("unknown", error_type=type(exc).__name__)
            state["weight_submission"] = submission.as_dict()
            write_state(state_path, state)
            if submission.status in {"unknown", "prepared"}:
                raise RuntimeError("Ambiguous burn submission; stopped for reconciliation")
        print(json.dumps(state), flush=True)
        if once:
            return
        await asyncio.sleep(interval_s)
