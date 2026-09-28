"""Independent RunPod stop watchdog. Read-only unless --stop is supplied.

Never starts, replaces or deletes a pod. A stop latch prevents the evaluator
from restarting compute until the operator reconciles provider state.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

from witness.storage import write_private
from .gpu import RunPodGpu


def inspect(gpu, root, *, now=None, stale_s=300, stop=False):
    now = time.time() if now is None else now
    heartbeat = root / 'heartbeat.json'
    pulse = json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
    state = gpu.state
    if state.get('owner') != 'witness-mainnet-2' or not state.get('pod_id'):
        return {'action': 'none', 'reason': 'no_owned_pod'}
    reason = None
    if now - pulse.get('unix', 0) > stale_s:
        reason = 'validator_heartbeat_stale'
    elif state.get('running_since') and gpu.spend_today() >= gpu.config.get('daily_cap_usd', 10.):
        reason = 'daily_cost_cap'
    elif not state.get('running_since') and now - state.get('created_unix', now) > gpu.config.get('ready_timeout_s', 900):
        reason = 'startup_deadline'
    if reason is None:
        return {'action': 'none', 'reason': 'within_limits'}
    result = {'action': 'would_stop', 'reason': reason, 'pod_id': state['pod_id'], 'unix': now}
    if stop:
        # Latch BEFORE the request. A failed stop must remain visible and retryable.
        write_private(root / 'gpu-watchdog-stop.json', result)
        pod = gpu._rest('GET', f"/pods/{state['pod_id']}")
        if pod.get('desiredStatus') != 'EXITED':
            gpu._rest('POST', f"/pods/{state['pod_id']}/stop")
        result['action'] = 'stop_requested'
        write_private(root / 'gpu-watchdog-stop.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--env', type=Path)
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--stale-seconds', type=int, default=300)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.env:
        from .validator import load_env
        load_env(args.env)
    config = json.loads(args.config.read_text())['gpu']
    if config.get('backend', 'runpod') != 'runpod':
        parser.error('watchdog only stops evaluator-owned RunPod compute')
    args.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (args.root / 'watchdog.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            result = inspect(RunPodGpu(config, args.root), args.root,
                             stale_s=args.stale_seconds, stop=args.stop)
        except Exception as error:
            result = {'action': 'error', 'error': type(error).__name__}
        write_private(args.root / 'watchdog-status.json', {**result, 'unix': time.time()})
        if args.once:
            print(json.dumps(result))
            return
        time.sleep(max(1, args.interval))


if __name__ == '__main__':
    main()
