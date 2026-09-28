"""Finalized chain commitments plus an append-only local rehearsal transport."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .protocol import MAX_COMMITMENT_BYTES
from .submission import SS58


class FileCommitments:
    def __init__(self, directory: Path, *, writable=False):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.writable = writable
        self.db = sqlite3.connect(directory / 'commitments.sqlite3', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS records (hotkey TEXT,value TEXT,block INTEGER, '
                        'UNIQUE(hotkey,value,block))')

    def read_all(self, block=None, block_hash=None):
        rows = self.db.execute('SELECT hotkey,value,block FROM records WHERE block<=? ORDER BY block,hotkey',
                               (block if block is not None else 2**63-1,))
        return {h: {'value': value, 'block': height} for h, value, height in rows}

    def at(self, block, block_hash=None):
        return [{'hotkey': h, 'value': v, 'block': b} for h, v, b in self.db.execute(
            'SELECT hotkey,value,block FROM records WHERE block=? ORDER BY hotkey', (block,))]

    def publish(self, hotkey, value, block):
        if not self.writable:
            raise PermissionError('commitment_writes_disabled')
        if not SS58.fullmatch(hotkey) or len(value.encode()) > MAX_COMMITMENT_BYTES:
            raise ValueError('invalid_commitment')
        self.db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?)', (hotkey, value, block))
        self.db.commit()
        return {'success': True, 'finalized': True, 'block': block}


class ChainCommitments:
    def __init__(self, subtensor, netuid, wallet=None, *, writable=False):
        self.subtensor, self.netuid, self.wallet = subtensor, netuid, wallet
        self.writable = bool(writable and wallet is not None)

    def read_all(self, block=None, block_hash=None):
        from bittensor.core.chain_data.utils import decode_metadata
        substrate = self.subtensor.substrate
        if block_hash is None:
            block_hash = substrate.get_block_hash(block) if block is not None else substrate.get_chain_finalised_head()
        rows = {}
        for hotkey, entry in substrate.query_map('Commitments', 'CommitmentOf', [self.netuid], block_hash=block_hash):
            value = getattr(entry, 'value', entry)
            try:
                rows[str(hotkey)] = {'value': decode_metadata(value), 'block': int(value['block'])}
            except (KeyError, TypeError, ValueError, UnicodeDecodeError):
                continue
        return rows

    def at(self, block, block_hash):
        return [{'hotkey': hotkey, **row} for hotkey, row in self.read_all(block, block_hash).items()
                if row['block'] == block]

    def publish(self, hotkey, value, block):
        if not self.writable or str(self.wallet.hotkey.ss58_address) != hotkey:
            raise PermissionError('commitment_writes_disabled')
        if len(value.encode()) > MAX_COMMITMENT_BYTES:
            raise ValueError('commitment_too_large')
        response = self.subtensor.set_commitment(wallet=self.wallet, netuid=self.netuid, data=value,
                                                 mev_protection=False, wait_for_inclusion=True,
                                                 wait_for_finalization=True)
        if response.success is not True:
            raise RuntimeError('commitment_not_finalized')
        receipt = getattr(response, 'extrinsic_receipt', None)
        return {'success': True, 'finalized': True, 'block_hash': getattr(receipt, 'block_hash', None),
                'extrinsic_hash': getattr(receipt, 'extrinsic_hash', None)}
