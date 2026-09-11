"""Minimal asynchronous Witness miner template; no model or inference policy."""

import asyncio
import math
from pathlib import Path
from typing import Any, Tuple

import typer

from .chain import BittensorChainAdapter
from .protocol import WitnessTask
from .feedback import FeedbackReceiver


class WitnessMiner:
    """Subclass reconstruct() to implement a miner using the assigned tools."""

    def __init__(self, *, chain=None, concurrency: int = 1,
                 max_deadline_s: float = 300.0, allow_unregistered: bool = False,
                 feedback_dir: Path | None = None):
        if concurrency < 1:
            raise ValueError("concurrency must be at least one")
        if not math.isfinite(max_deadline_s) or max_deadline_s <= 0:
            raise ValueError("max_deadline_s must be positive and finite")
        self.chain = chain
        self.concurrency = concurrency
        self.max_deadline_s = max_deadline_s
        self.allow_unregistered = allow_unregistered
        self._active = 0
        self.feedback = FeedbackReceiver(chain=chain, feedback_dir=feedback_dir)

    async def reconstruct(self, task: WitnessTask) -> dict[str, Any]:
        """Return an empty valid response. Replace with nonblocking inference.

        Extension workflow: read task.task_spec, plan within task.budget, query
        the assigned observation tools, and fill this schema from your evidence.
        Keep inference in reconstruct(); forward() owns admission and deadlines.

        Use task.tool_base_url and task.session_id for observations. Never create
        a second session or access validator labels. Propagate cancellation and
        bound external requests by the task deadline.
        """
        return {"schema_version": "1.0", "events": [], "dialogue": [],
                "shots": [], "on_screen_text": [], "audio_events": [],
                "intentional_errors": [], "qa": {}}

    async def forward(self, synapse: WitnessTask) -> WitnessTask:
        synapse.reconstruction = {}
        if self._active >= self.concurrency:
            synapse.trace_summary = {"status": "busy"}
            return synapse
        self._active += 1
        try:
            result = await asyncio.wait_for(
                self.reconstruct(synapse),
                timeout=min(synapse.deadline_s, self.max_deadline_s),
            )
            if not isinstance(result, dict):
                raise TypeError("reconstruction must be an object")
            synapse.reconstruction = result
            synapse.trace_summary = {"status": "ok"}
        except asyncio.TimeoutError:
            synapse.trace_summary = {"status": "deadline_exceeded"}
        except Exception as exc:
            synapse.trace_summary = {"status": "error", "error_type": type(exc).__name__}
        finally:
            self._active -= 1
        return synapse

    async def blacklist(self, synapse: WitnessTask) -> Tuple[bool, str]:
        hotkey = str(getattr(synapse.dendrite, "hotkey", "") or "")
        if not hotkey:
            return True, "missing caller hotkey"
        if self.chain is not None and not self.allow_unregistered and not self.chain.is_registered(hotkey):
            return True, "caller hotkey is not registered"
        return False, "registered caller"

    async def priority(self, synapse: WitnessTask) -> float:
        hotkey = str(getattr(synapse.dendrite, "hotkey", "") or "")
        return self.chain.stake_for_hotkey(hotkey) if self.chain is not None else 0.0


app = typer.Typer(add_completion=False, help="Serve the Witness base miner axon")


@app.command()
def run(
    netuid: int = typer.Option(1, envvar="WITNESS_NETUID"),
    network: str = typer.Option("finney", envvar="WITNESS_NETWORK"),
    wallet_name: str = typer.Option("default", "--wallet", envvar="WITNESS_WALLET"),
    wallet_hotkey: str = typer.Option("default", "--hotkey", envvar="WITNESS_HOTKEY"),
    wallet_path: str | None = typer.Option(None, envvar="WITNESS_WALLET_PATH"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8091, min=1, max=65535),
    external_ip: str | None = typer.Option(None, envvar="WITNESS_EXTERNAL_IP"),
    external_port: int | None = typer.Option(None, envvar="WITNESS_EXTERNAL_PORT"),
    concurrency: int = typer.Option(1, min=1),
    max_deadline_s: float = typer.Option(300.0, min=1),
    allow_unregistered: bool = typer.Option(False),
    feedback_dir: Path = typer.Option(Path("feedback"), envvar="WITNESS_FEEDBACK_DIR"),
) -> None:
    """Serve empty responses until reconstruct() is implemented."""
    chain = BittensorChainAdapter(netuid=netuid, network=network,
        wallet_name=wallet_name, wallet_hotkey=wallet_hotkey,
        wallet_path=wallet_path, with_dendrite=False)
    miner = WitnessMiner(chain=chain, concurrency=concurrency,
        max_deadline_s=max_deadline_s, allow_unregistered=allow_unregistered,
        feedback_dir=feedback_dir)
    axon = chain.serve_axon(forward_fn=miner.forward, blacklist_fn=miner.blacklist,
        priority_fn=miner.priority, port=port, host=host, external_ip=external_ip,
        external_port=external_port, max_workers=concurrency, feedback_receiver=miner.feedback)
    typer.echo(f"Witness base miner serving on port {port}; no inference configured")
    async def wait():
        while True:
            await asyncio.sleep(1)
    try:
        asyncio.run(wait())
    except KeyboardInterrupt:
        pass
    finally:
        axon.stop()
        chain.close()


if __name__ == "__main__":
    app()
