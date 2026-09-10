"""One round per observed finalized subnet epoch, with restart-safe ownership."""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Any

log = logging.getLogger(__name__)


async def run_epoch_rounds(chain: Any, run_round: Callable[[], Awaitable[Any]],
                           state_path: Path, *, poll_interval_s: float = 12.0) -> None:
    """Start once in the current epoch on first launch; never replay missed epochs."""
    if poll_interval_s <= 0:
        raise ValueError("Epoch poll interval must be positive")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        last_epoch = int(state.get('last_attempted_epoch', -1))

        def save(**updates: Any) -> None:
            state.update(updates, updated_at=time.time())
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.chmod(0o600)
            temporary.replace(state_path)

        while True:
            try:
                snapshot = chain.epoch_state()
                epoch = int(snapshot['epoch_index'])
                if state.get('netuid', snapshot['netuid']) != snapshot['netuid']:
                    raise ValueError('Scheduler state belongs to a different subnet')
                save(chain=snapshot, netuid=snapshot['netuid'])
                if epoch > last_epoch:
                    last_epoch = epoch
                    save(status='running', last_attempted_epoch=epoch, started_at=time.time())
                    try:
                        artifact = await run_round()
                        save(last_round_id=artifact.get('round_id') if isinstance(artifact, dict) else None,
                             last_round_status='completed', finished_at=time.time())
                    except Exception as exc:
                        log.exception('Epoch round failed; waiting for a new epoch')
                        save(last_round_status='failed', error_type=type(exc).__name__, finished_at=time.time())
                    # Consume epochs crossed during a long round instead of catching up.
                    finished = chain.epoch_state()
                    last_epoch = max(last_epoch, int(finished['epoch_index']))
                    save(last_attempted_epoch=last_epoch, chain=finished)
                save(status='waiting_for_epoch')
            except Exception as exc:
                log.exception('Epoch state unavailable; no round started')
                save(status='chain_error', error_type=type(exc).__name__)
            await asyncio.sleep(poll_interval_s)
