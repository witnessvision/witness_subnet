"""Empty miner template and immutable, authenticated P2P model submissions.

Witness charges no challenge fee. Each hotkey reserves one model permanently.
Serving/inspection never publishes; --publish is required for chain mutations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

from witness.storage import write_private
from .contract import Response
from .submission import ARCHITECTURES, Submission, challenge_id, prepare_manifest, validate_manifest


def respond(_mp4):
    return Response(claims=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    template = commands.add_parser('respond')
    template.add_argument('--mp4', type=Path, required=True)
    for name in ('prepare', 'serve', 'commit', 'status'):
        command = commands.add_parser(name)
        command.add_argument('--network', default='finney')
        command.add_argument('--netuid', type=int, default=20)
        command.add_argument('--wallet-name', default='default')
        command.add_argument('--wallet-hotkey', default='default')
        command.add_argument('--wallet-path')
        command.add_argument('--state', type=Path, default=Path.home() / '.witness' / 'miner-v2')
        if name != 'status':
            command.add_argument('--model-dir', type=Path, required=True)
            command.add_argument('--arch', choices=sorted(ARCHITECTURES), required=True)
        if name in ('serve', 'commit'):
            command.add_argument('--publish', action='store_true', help='Explicitly publish endpoint or submission to chain')
    serve = commands.choices['serve']
    serve.add_argument('--host', default='0.0.0.0')
    serve.add_argument('--port', type=int, default=8091)
    serve.add_argument('--external-ip', help='Required to publish a served endpoint')
    serve.add_argument('--min-stake-alpha', type=int, default=100000)
    args = parser.parse_args()
    if args.command == 'respond':
        print(respond(args.mp4).model_dump_json())
        return
    from .p2p import Auth, ModelServer, certificate, endpoint
    root = args.state.expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.command != 'status':
        manifest = prepare_manifest(args.model_dir, args.arch)
        info = validate_manifest(manifest)
        certfile, keyfile, fingerprint = certificate(root)
        submission = Submission(info['model_id'], fingerprint)
        write_private(root / 'manifest.json', manifest)
        if args.command == 'prepare' or (args.command == 'commit' and not args.publish):
            print(json.dumps({'manifest_sha256': info['model_id'], 'bytes': info['bytes'],
                              'commitment': submission.commitment, 'published': False}))
            return
    import bittensor as bt
    from .chain import Chain
    from .commitments import ChainCommitments
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey,
                       **({'path': args.wallet_path} if args.wallet_path else {}))
    hotkey = str(wallet.hotkey.ss58_address)
    chain = Chain(args.network, args.netuid)
    try:
        store = ChainCommitments(chain.subtensor, args.netuid, wallet, writable=getattr(args, 'publish', False))
        if args.command == 'status':
            print(json.dumps({'hotkey': hotkey, 'commitment': store.read_all().get(hotkey)}))
        elif args.command == 'commit':
            receipt = store.publish(hotkey, submission.commitment, chain.head())
            print(json.dumps({'hotkey': hotkey, 'model_id': challenge_id(hotkey, submission.model_id),
                              'manifest_sha256': submission.model_id, 'receipt': receipt}))
        elif args.command == 'serve':
            if args.min_stake_alpha < 0:
                parser.error('--min-stake-alpha must be nonnegative')
            if args.publish:
                if not args.external_ip:
                    parser.error('--publish requires --external-ip')
                ip, port = endpoint(args.external_ip, args.port)
                from bittensor.core.extrinsics.serving import serve_extrinsic
                receipt = serve_extrinsic(chain.subtensor, wallet, ip=ip, port=port,
                                         protocol=4, netuid=args.netuid, mev_protection=False,
                                         wait_for_inclusion=True, wait_for_finalization=True)
                if receipt.success is not True:
                    raise RuntimeError('endpoint_not_finalized')
            current = {'snapshot': chain.snapshot()}
            stop = threading.Event()
            def refresh():
                while not stop.wait(30):
                    try:
                        current['snapshot'] = chain.snapshot()
                    except Exception:
                        pass  # existing data ages out; Auth fails closed after 120 seconds
            thread = threading.Thread(target=refresh, daemon=True, name='witness-miner-chain')
            thread.start()
            server = ModelServer((args.host, args.port), model_root=args.model_dir, manifest=manifest,
                                  auth=Auth(hotkey, lambda: current['snapshot'], min_stake_alpha=args.min_stake_alpha),
                                  certfile=certfile, keyfile=keyfile)
            print(json.dumps({'model_id': challenge_id(hotkey, submission.model_id), 'port': server.server_port,
                              'min_stake_alpha': args.min_stake_alpha}), flush=True)
            try:
                server.serve_forever()
            finally:
                stop.set()
                server.server_close()
                thread.join(timeout=35)
    finally:
        chain.close()


if __name__ == '__main__':
    main()
