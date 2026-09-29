"""CPU follower by default. Chain replay/weights never wait for GPU inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time

from witness.storage import write_private
from .chain import Chain
from .commitments import ChainCommitments
from .ledger import Ledger
from .protocol import policy_identity, weight_vector, weights_match


class Validator:
    def __init__(self, *, hotkey, root, chain, store, ledger, set_weights=None, evaluator=None,
                 publication_store=None, telemetry=None):
        self.hotkey, self.root, self.chain, self.store, self.ledger = hotkey, Path(root), chain, store, ledger
        self.set_weights, self.evaluator = set_weights, evaluator
        self.telemetry = telemetry
        self.publication_store = publication_store or store
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='witness-chain-writer')
        self.future = None
        self.operation = None
        self.snapshot_cache = None
        interrupted = self._load('weight-send.json', {})
        if interrupted.get('status') == 'submitting':
            write_private(self.root / 'weight-send.json', {**interrupted, 'status': 'unknown'})

    @property
    def mode(self):
        return 'evaluator' if self.evaluator else 'follower'

    def _complete_write(self):
        if self.future is None or not self.future.done():
            return
        operation = self.operation
        try:
            receipt = self.future.result()
            if operation['kind'] == 'weights':
                state = {**operation, 'receipt': receipt,
                         'status': 'submitted' if receipt.get('success') is True else 'rejected'}
                write_private(self.root / 'weight-send.json', state)
                if receipt.get('success') is True:
                    write_private(self.root / 'last-weights.json', {**operation, 'receipt': receipt})
        except Exception as error:
            # Unknown submission outcome must be reconciled before signing again.
            if operation['kind'] == 'weights':
                write_private(self.root / 'weight-send.json', {**operation, 'status': 'unknown',
                                                              'error': type(error).__name__})
            else:
                write_private(self.root / 'commitment-error.json', {'block': operation['block'],
                                                                   'error': type(error).__name__})
        finally:
            self.future = self.operation = None

    def _load(self, name, default=None):
        path = self.root / name
        return json.loads(path.read_text()) if path.exists() else default

    def _due(self, vector, snapshot):
        state = self._load('weight-send.json', {})
        if state.get('status') in ('unknown', 'submitted'):
            # Reconcile observable finalization; an operator must resolve an ambiguous,
            # non-observable pending commit instead of potentially duplicating it.
            if (weights_match(self.chain.applied_weights(self.hotkey, snapshot), state)
                    and snapshot.get('last_update', {}).get(self.hotkey, snapshot['block']) >= state['block']):
                write_private(self.root / 'weight-send.json', {**state, 'status': 'applied', 'block': snapshot['block']})
                write_private(self.root / 'last-weights.json', {**state, 'block': snapshot['block']})
            return False
        if state.get('status') == 'rejected' and snapshot['block'] - state['block'] < 5:
            return False
        last = self._load('last-weights.json', {})
        last_chain = snapshot.get('last_update', {}).get(self.hotkey, 0)
        if snapshot['block'] - max(last.get('block', 0), last_chain) < snapshot.get('weights_rate_limit', 100):
            return False
        return ((last.get('uids'), last.get('weights')) != (vector['uids'], vector['weights'])
                or snapshot['block'] - last.get('block', 0) >= snapshot.get('tempo', 360))

    def step(self):
        write_private(self.root / 'heartbeat.json', {'unix': time.time(), 'pid': os.getpid()})
        self._complete_write()
        head = self.chain.head()
        # Every block is read: polling just the newest CommitmentOf loses intermediate scores.
        for block in range(self.ledger.cursor + 1, min(head, self.ledger.cursor + 64) + 1):
            frame = self.chain.frame(block)
            rows = self.store.at(block, frame['block_hash'])
            active = self.ledger.active
            boundary = active is None or frame['epoch_index'] >= active['end_epoch']
            if self.snapshot_cache is None or rows or boundary:
                self.snapshot_cache = self.chain.snapshot(block)
            snapshot = {**self.snapshot_cache, **frame}
            self.ledger.ingest(snapshot, rows)
        snapshot = self.chain.snapshot(head)
        caught_up = self.ledger.cursor == head
        decision = self.ledger.get('decision', {})
        king = decision.get('king')
        if king and king['hotkey'] not in snapshot['uids']:
            king = None
        vector = weight_vector(king, snapshot)
        intent = {**vector, 'block': head, 'block_hash': snapshot['block_hash'], 'mode': self.mode,
                  'validator': self.hotkey, 'window': self.ledger.active, 'caught_up': caught_up,
                  'set_weights': self.set_weights is not None, 'decision': decision}
        if caught_up and self.evaluator:
            self.evaluator.update(snapshot)
        if caught_up and self.future is None:
            if (self.set_weights is not None and vector['uids'] and self._due(vector, snapshot)
                    and self.hotkey in snapshot['permits']):
                if (snapshot.get('min_weights', 1) > len(vector['uids'])
                        or max(vector['weights']) > snapshot.get('max_weight', 65535) / 65535 + 1 / 65535):
                    raise RuntimeError('chain_disallows_burn_allocation')
                operation = {'kind': 'weights', 'block': head, 'uids': vector['uids'], 'weights': vector['weights'],
                             'hotkey': king['hotkey'] if king else None}
                write_private(self.root / 'weight-send.json', {**operation, 'status': 'submitting'})
                self.operation = operation
                self.future = self.executor.submit(self.set_weights, vector['uids'], vector['weights'])
            elif self.evaluator:
                last_error = self._load('commitment-error.json', {})
                if head - last_error.get('block', -100) >= 5:
                    self.operation = {'kind': 'commitment', 'block': head}
                    self.future = self.executor.submit(self.evaluator.flush_one, self.publication_store, snapshot)
        observed = self.chain.applied_weights(self.hotkey, snapshot) if self.hotkey in snapshot['uids'] else None
        intent['weights_applied'] = ({'hotkey': king['hotkey'] if king else None,
                                      'block': snapshot.get('last_update', {}).get(self.hotkey),
                                      'uids': vector['uids'], 'weights': vector['weights']}
                                     if weights_match(observed, vector) else None)
        write_private(self.root / 'intended-weights.json', intent)
        self.publish_status(snapshot, intent)
        return intent

    def publish_status(self, snapshot, intent):
        from .protocol import BASELINE, EARLY_STOP, Result
        active = self.ledger.active
        triggers = self.evaluator.triggers.rows() if self.evaluator else []
        if self.evaluator and active:
            order = self.evaluator.triggers.pending(10000, before_block=active['start_block'], current_block=snapshot['block'])
            positions = {r['hotkey']: i + 1 for i, r in enumerate(order) if r['model_id'] in active['candidates']}
            for row in triggers:
                row['position'] = positions.get(row['hotkey'])
                if row['usage'] == 'reserved' and row['position'] is None and not row.get('reason'):
                    row['reason'] = 'eligible_in_next_window'
        evaluations = []
        for hotkey, usage in self.ledger.usage(self.hotkey).items():
            value = usage['result']
            partial = bool(value['flags'] & EARLY_STOP)
            evaluations.append({'validator': self.hotkey, 'window_id': usage['window'], 'hotkey': hotkey,
                                'model_id': value['model_id'], 'quality': None if partial else value['quality'],
                                'reward': None if partial else value['reward'],
                                'quality_upper': value['quality'] if partial else None,
                                'reward_upper': value['reward'] if partial else None,
                                'status': 'early_stop' if partial else 'rejected' if value['flags'] & 2 else 'done',
                                'report_hash': value['evidence'],
                                'available': self.ledger.is_closed(usage['window'])})
        own_block = self.ledger.last_commitment_block(self.hotkey)
        window = {key: active[key] for key in ('id', 'start_block', 'start_epoch', 'end_epoch', 'state')} if active else None
        if window:
            window['end_block'] = None
        decision = intent['decision']
        write_private(self.root / 'queue.json', {'schema_version': 'witness-evaluator-status-2',
            'validator': self.hotkey, 'mode': self.mode, 'block': snapshot['block'],
            'updated_unix': time.time(), 'policy_hash': self.ledger.policy,
            'stake': snapshot['validators'].get(self.hotkey, 0) / 1e9, 'commitment_block': own_block,
            'window': window, 'king': intent['king'], 'triggers': triggers, 'evaluations': evaluations,
            'caught_up': intent['caught_up'],
            'progress': self._load('progress.json'),
            'weights': {'intended': {k: intent[k] for k in ('uids', 'weights', 'source', 'burn_uid',
                        'king_uid', 'burn_fraction', 'king_fraction')},
                        'decision': {'hotkey': (decision.get('king') or {}).get('hotkey'), 'block': decision['block'],
                                     'inconclusive': decision.get('inconclusive', [])}
                         if decision else None, 'submitted': self._load('weight-send.json'),
                        'applied': intent['weights_applied']}})
        write_private(self.root / 'telemetry-peers.json', {'self': self.hotkey, 'updated_unix': time.time(),
                                                         'validators': snapshot['validators']})
        if self.telemetry:
            self.telemetry.submit(self._load('queue.json'))

    def close(self):
        if self.evaluator:
            self.evaluator.stop()
        # Keep the writer connection and process lock alive until its final receipt.
        self.executor.shutdown(wait=True, cancel_futures=True)
        self._complete_write()
        if self.telemetry:
            self.telemetry.close()


def load_env(path):
    if path.stat().st_mode & 0o077:
        raise ValueError('unsafe_env_permissions')
    for line in path.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            name, value = line.split('=', 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"\''))


def submit_weights(chain, wallet, netuid, uids, weights):
    """Use the current chain-required version, not the SDK's default zero."""
    substrate = chain.subtensor.substrate
    head = substrate.get_chain_finalised_head()
    version = int(substrate.query('SubtensorModule', 'WeightsVersionKey', [netuid], block_hash=head).value)
    response = chain.subtensor.set_weights(wallet=wallet, netuid=netuid, uids=uids, weights=weights,
                                          version_key=version, wait_for_inclusion=True, wait_for_finalization=True)
    receipt = getattr(response, 'extrinsic_receipt', None)
    return {'success': response.success is True, 'version_key': version,
            'extrinsic_hash': getattr(receipt, 'extrinsic_hash', None),
            'block_hash': getattr(receipt, 'block_hash', None),
            'error': type(response.error).__name__ if response.error else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['follower', 'evaluator'], default='follower')
    parser.add_argument('--network', default='finney')
    parser.add_argument('--netuid', type=int, default=20)
    parser.add_argument('--activation-block', type=int, required=True)
    parser.add_argument('--activation-epoch', type=int, required=True)
    parser.add_argument('--root', type=Path, default=Path.home() / '.witness' / 'validator-v2')
    parser.add_argument('--wallet-name', default='default')
    parser.add_argument('--wallet-hotkey', default='default')
    parser.add_argument('--wallet-path')
    parser.add_argument('--hotkey', help='Public identity for a CPU read-only follower; no wallet is opened')
    parser.add_argument('--set-weights', action='store_true')
    parser.add_argument('--publish-results', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--env', type=Path)
    parser.add_argument('--interval', type=float, default=12.)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--status-port', type=int)
    parser.add_argument('--status-host', default='127.0.0.1')
    parser.add_argument('--web-port', type=int, help='Serve the public web/API in this process (requires a separately installed web application)')
    parser.add_argument('--web-host', default='127.0.0.1')
    parser.add_argument('--telemetry-url', help='Optional HTTPS /api/telemetry for signed display status')
    args = parser.parse_args()
    if args.hotkey and (args.set_weights or args.publish_results or args.mode == 'evaluator' or args.telemetry_url):
        parser.error('--hotkey is only for a read-only CPU follower; evaluator needs a signing hotkey for P2P')
    if args.mode == 'evaluator' and not args.config:
        parser.error('evaluator requires --config')
    if args.publish_results and args.mode != 'evaluator':
        parser.error('followers do not publish independent evaluations')
    if args.env:
        load_env(args.env)
    root = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (root / 'validator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    wallet = None
    if not args.hotkey:
        import bittensor as bt
        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey,
                           **({'path': args.wallet_path} if args.wallet_path else {}))
    hotkey = args.hotkey or str(wallet.hotkey.ss58_address)
    chain, writer_chain = Chain(args.network, args.netuid), None
    ledger = Ledger(root / 'chain.sqlite3', activation_block=args.activation_block,
                    activation_epoch=args.activation_epoch, policy=policy_identity())
    store = ChainCommitments(chain.subtensor, args.netuid)
    publish_store = None
    set_weights = None
    if args.publish_results or args.set_weights:
        writer_chain = Chain(args.network, args.netuid)
        publish_store = ChainCommitments(writer_chain.subtensor, args.netuid, wallet, writable=args.publish_results)
    if args.set_weights:
        def set_weights(uids, weights):
            return submit_weights(writer_chain, wallet, args.netuid, uids, weights)
    evaluator = None
    if args.mode == 'evaluator':
        from .managed_evaluator import ManagedEvaluator
        evaluator = ManagedEvaluator.from_config(
            json.loads(args.config.read_text()), root, hotkey, ledger, wallet.hotkey)
    telemetry = None
    if args.telemetry_url:
        from .telemetry import TelemetrySender
        telemetry = TelemetrySender(args.telemetry_url, wallet.hotkey)
    validator = Validator(hotkey=hotkey, root=root, chain=chain, store=store, ledger=ledger,
                          evaluator=evaluator, set_weights=set_weights, publication_store=publish_store,
                          telemetry=telemetry)
    server = None
    if args.status_port:
        from .status import serve_status
        server = serve_status(root, args.status_host, args.status_port)
    web, web_thread = None, None
    if args.web_port:
        import uvicorn
        from witness_web.app import create_app
        web = uvicorn.Server(uvicorn.Config(create_app(sources=[str(root)], telemetry_root=root),
                              host=args.web_host, port=args.web_port, access_log=False))
        web_thread = threading.Thread(target=web.run, name='witness-web', daemon=True)
        web_thread.start()
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    try:
        while not stopped.is_set():
            if web_thread and not web_thread.is_alive():
                raise RuntimeError('web_server_stopped')
            try:
                result = validator.step()
                print(json.dumps({k: result[k] for k in ('block', 'mode', 'caught_up', 'uids', 'weights')}), flush=True)
            except Exception as error:
                write_private(root / 'last-error.json', {'type': type(error).__name__, 'unix': time.time()})
                print(json.dumps({'error': type(error).__name__}), flush=True)
            if args.once:
                break
            stopped.wait(args.interval)
    finally:
        validator.close()
        if server:
            server.shutdown()
        if web:
            web.should_exit = True
            web_thread.join(timeout=15)
        chain.close()
        if writer_chain and validator.future is None:
            writer_chain.close()
        lock.close()
        ledger.close()


if __name__ == '__main__':
    main()
