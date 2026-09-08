"""Thin Bittensor adapter plus a deterministic in-memory test implementation."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Protocol

from .protocol import WitnessTask


@dataclass(frozen=True, slots=True)
class MinerEndpoint:
    uid: int
    hotkey: str
    axon: Any = None


@dataclass(frozen=True, slots=True)
class WeightSubmission:
    """Submission evidence, never a claim that commit/reveal weights are active."""
    status: str
    extrinsic_hash: str | None = None
    block_hash: str | None = None
    operation: str | None = None
    error_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "weights_applied": None}


class ChainAdapter(Protocol):
    validator_hotkey: str

    def current_block_hash(self) -> str: ...

    def miner_endpoints(self) -> list[MinerEndpoint]: ...

    async def query(
        self, endpoint: MinerEndpoint, task: WitnessTask, timeout: float
    ) -> WitnessTask | None: ...

    def set_weights(self, uids: list[int], weights: list[float]) -> WeightSubmission: ...


def _bt() -> Any:
    import bittensor as bt

    return bt


def _constructor(module: Any, *names: str) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise AttributeError(f"bittensor does not expose any of: {', '.join(names)}")


class BittensorChainAdapter:
    """Own all SDK objects so the validator and miner core remain chain-free."""

    def __init__(
        self,
        *,
        netuid: int,
        network: str,
        wallet_name: str,
        wallet_hotkey: str,
        wallet_path: str | None = None,
        with_dendrite: bool = True,
    ) -> None:
        bt = _bt()
        wallet_kwargs = {"name": wallet_name, "hotkey": wallet_hotkey}
        if wallet_path:
            wallet_kwargs["path"] = wallet_path
        self.netuid = int(netuid)
        self.network = network
        self.wallet = _constructor(bt, "Wallet", "wallet")(**wallet_kwargs)
        self.subtensor = _constructor(bt, "Subtensor", "subtensor")(network=network)
        self.dendrite = (
            _constructor(bt, "Dendrite", "dendrite")(wallet=self.wallet)
            if with_dendrite
            else None
        )
        self.metagraph = self.subtensor.metagraph(self.netuid)
        self.validator_hotkey = str(self.wallet.hotkey.ss58_address)

    def current_block_hash(self) -> str:
        block = int(self.subtensor.get_current_block())
        substrate = getattr(self.subtensor, "substrate", None)
        if substrate is None or not hasattr(substrate, "get_block_hash"):
            raise RuntimeError("subtensor substrate does not expose get_block_hash")
        return str(substrate.get_block_hash(block))

    def miner_endpoints(self) -> list[MinerEndpoint]:
        self.metagraph.sync(subtensor=self.subtensor)
        uids = [int(value) for value in self.metagraph.uids.tolist()]
        return [
            MinerEndpoint(uid=uid, hotkey=str(self.metagraph.hotkeys[index]), axon=self.metagraph.axons[index])
            for index, uid in enumerate(uids)
        ]

    async def query(
        self, endpoint: MinerEndpoint, task: WitnessTask, timeout: float
    ) -> WitnessTask | None:
        if self.dendrite is None:
            raise RuntimeError("this adapter was created without a dendrite")
        responses = await self.dendrite(
            axons=[endpoint.axon],
            synapse=task,
            timeout=timeout,
            deserialize=False,
        )
        if not responses:
            return None
        response = responses[0]
        status = getattr(getattr(response, "dendrite", None), "status_code", None)
        return response if status in (None, 200) else None

    def set_weights(self, uids: list[int], weights: list[float]) -> WeightSubmission:
        if len(uids) != len(weights):
            raise ValueError("uids and weights must have the same length")
        response = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            max_attempts=1,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            wait_for_revealed_execution=False,
        )
        # Pinned SDK returns ExtrinsicResponse. Persist only safe receipt fields,
        # not raw extrinsics, arbitrary provider messages or signing payloads.
        if response.success is not True:
            return WeightSubmission("unknown" if response.error else "rejected", operation=response.extrinsic_function,
                                    error_type=type(response.error).__name__ if response.error else None)
        receipt = response.extrinsic_receipt
        block_hash = getattr(receipt, "block_hash", None)
        extrinsic_hash = getattr(receipt, "extrinsic_hash", None)
        return WeightSubmission(
            ("finalized" if getattr(receipt, "finalized", False) is True else "included") if block_hash else "submitted",
            extrinsic_hash=extrinsic_hash, block_hash=block_hash,
            operation=response.extrinsic_function,
        )

    def serve_axon(
        self,
        *,
        forward_fn: Callable[..., Any],
        blacklist_fn: Callable[..., Any],
        priority_fn: Callable[..., Any],
        port: int,
        host: str,
        external_ip: str | None = None,
        external_port: int | None = None,
        max_workers: int = 4,
    ) -> Any:
        bt = _bt()
        axon = _constructor(bt, "Axon", "axon")(
            wallet=self.wallet,
            port=port,
            ip=host,
            external_ip=external_ip,
            external_port=external_port,
            max_workers=max_workers,
        )
        axon.attach(
            forward_fn=forward_fn,
            blacklist_fn=blacklist_fn,
            priority_fn=priority_fn,
        )
        return axon.serve(netuid=self.netuid, subtensor=self.subtensor).start()

    def stake_for_hotkey(self, hotkey: str) -> float:
        try:
            index = self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return 0.0
        return float(self.metagraph.S[index])

    def is_registered(self, hotkey: str) -> bool:
        return hotkey in self.metagraph.hotkeys

    def close(self) -> None:
        for value in (self.dendrite, self.subtensor):
            close = getattr(value, "close", None)
            if callable(close):
                close()


Handler = Callable[[WitnessTask], WitnessTask | Awaitable[WitnessTask]]


class InMemoryChainAdapter:
    """Fake metagraph, dendrite, and weight sink used by dry runs and tests."""

    def __init__(
        self,
        handlers: dict[int, Handler] | None = None,
        *,
        block_hash: str = "0x" + "42" * 32,
        validator_hotkey: str = "validator-test-hotkey",
    ) -> None:
        self.handlers = dict(handlers or {})
        self.hotkeys = {uid: f"fake-hotkey-{uid}" for uid in self.handlers}
        self.block_hash = block_hash
        self.validator_hotkey = validator_hotkey
        self.weight_history: list[dict[str, list[int] | list[float]]] = []

    def add_miner(self, uid: int, handler: Handler, hotkey: str | None = None) -> None:
        self.handlers[int(uid)] = handler
        self.hotkeys[int(uid)] = hotkey or f"fake-hotkey-{int(uid)}"

    def current_block_hash(self) -> str:
        return self.block_hash

    def miner_endpoints(self) -> list[MinerEndpoint]:
        return [
            MinerEndpoint(uid=uid, hotkey=self.hotkeys[uid], axon=self.handlers[uid])
            for uid in sorted(self.handlers)
        ]

    async def query(
        self, endpoint: MinerEndpoint, task: WitnessTask, timeout: float
    ) -> WitnessTask | None:
        request = task.model_copy(deep=True)

        async def invoke() -> WitnessTask:
            result = endpoint.axon(request)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, WitnessTask):
                raise TypeError("fake miner handler must return WitnessTask")
            return result

        try:
            return await asyncio.wait_for(invoke(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return None

    def set_weights(self, uids: list[int], weights: list[float]) -> WeightSubmission:
        if len(uids) != len(weights):
            raise ValueError("uids and weights must have the same length")
        record: dict[str, Any] = {"uids": list(uids), "weights": list(weights)}
        self.weight_history.append(record)
        return WeightSubmission("simulated")
