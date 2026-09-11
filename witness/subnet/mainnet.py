"""CPU-only SN20 preset and finalized-chain checks for scored weights."""
from __future__ import annotations

import math
from typing import Any

from witness.score_v1_1_0 import SCORER_VERSION

from .burn import burn_state
from .chain import BittensorChainAdapter, MinerEndpoint, WeightSubmission

FINNEY_GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"


class MainnetChainAdapter(BittensorChainAdapter):
    """Discover all serving non-owner miners and verify destinations before writes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.netuid != 20 or self.subtensor.substrate.get_block_hash(0) != FINNEY_GENESIS:
            self.close()
            raise ValueError("The mainnet preset requires Witness SN20 on Finney")
        self.burn_uid = int(burn_state(self, None)["burn_uid"])
        self.selected: dict[int, str] = {}

    def miner_endpoints(self) -> list[MinerEndpoint]:
        # Sync before pinning so an advancing metagraph is checked against the
        # newest finalized registry, never a stale owner/UID snapshot.
        endpoints = super().miner_endpoints()
        state = burn_state(self, self.burn_uid)
        p = self.subtensor.substrate
        def q(name: str, *args: Any) -> Any:
            return p.query("SubtensorModule", name, list(args), block_hash=state["block_hash"]).value
        owner = q("SubnetOwner", self.netuid)
        selected = []
        for endpoint in endpoints:
            if endpoint.uid != self.burn_uid and not (endpoint.axon is not None and endpoint.axon.is_serving):
                continue
            if q("Keys", self.netuid, endpoint.uid) != endpoint.hotkey:
                raise RuntimeError("Miner registration changed during endpoint discovery; retry next epoch")
            if endpoint.uid == self.burn_uid or q("Owner", endpoint.hotkey) != owner:
                selected.append(endpoint)
        if self.burn_uid not in {e.uid for e in selected}:
            raise RuntimeError("The registered owner burn endpoint is missing")
        self.selected = {e.uid: e.hotkey for e in selected}
        return selected

    async def query(self, endpoint, task, timeout):
        if endpoint.uid == self.burn_uid:
            return None
        return await super().query(endpoint, task, timeout)

    def set_weights(self, uids: list[int], weights: list[float]) -> WeightSubmission:
        if len(uids) != len(weights) or len(set(uids)) != len(uids):
            raise ValueError("Invalid weight destinations")
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Invalid weights")
        positive = {u: w for u, w in zip(uids, weights) if w > 0}
        burn_weight = positive.get(self.burn_uid, 0)
        if not (len(positive) == 1 and math.isclose(burn_weight, 1.0) or
                len(positive) == 2 and math.isclose(burn_weight, .7)
                and math.isclose(sum(positive.values()), 1.0)):
            raise ValueError("Mainnet requires 70% burn / 30% one winner, or full burn")
        state = burn_state(self, self.burn_uid)
        if state["blocks_until_submission"]:
            return WeightSubmission("rate_limited")  # definitely no network write
        p = self.subtensor.substrate
        def q(name: str, *args: Any) -> Any:
            return p.query("SubtensorModule", name, list(args), block_hash=state["block_hash"]).value
        owner = q("SubnetOwner", self.netuid)
        for uid in positive:
            hotkey = q("Keys", self.netuid, uid)
            if self.selected.get(uid) != hotkey:
                raise RuntimeError("Weight destination hotkey changed since evaluation")
            if uid != self.burn_uid and q("Owner", hotkey) == owner:
                raise RuntimeError("Winning hotkey became an owner burn destination")
        if q("MinAllowedWeights", self.netuid) > len(positive):
            raise RuntimeError("Chain minimum weight count rejects this winner policy")
        if q("MaxWeightsLimit", self.netuid) != 65535:
            raise RuntimeError("Chain weight cap changed; verify the policy before submitting")
        return super().set_weights(uids, weights)


def mainnet_config(*, round_root, burn_uid, tool_host, tool_port, tool_public_url,
                   set_weights_enabled, score_window=5, ema_alpha=0.2, score_version=SCORER_VERSION,
                   pool_manifest=None):
    from .validator import ValidatorConfig
    grounded = score_version == "3.0.0"
    if grounded and pool_manifest is None:
        raise ValueError("Grounded mainnet rounds require a reviewed natural annotation pool")
    return ValidatorConfig(
        round_root=round_root, scene_count=5, programmatic_share=.6 if grounded else 1.0,
        pool_manifest=pool_manifest if grounded else None,
        tool_host=tool_host, tool_port=tool_port, tool_public_url=tool_public_url,
        burn_uid=burn_uid, burn_rate=.7, weight_policy="winner-takes-all",
        transcript_source="none", allow_unlocked=True, epoch_aligned=True,
        query_concurrency=4, set_weights_enabled=set_weights_enabled,
        score_window=score_window, ema_alpha=ema_alpha, score_version=score_version,
    )
