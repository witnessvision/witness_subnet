"""Pinned finalized SN state, using integer alpha balances for access control."""
from __future__ import annotations
import ipaddress
import time


class Chain:
    def __init__(self, network='finney', netuid=20):
        import bittensor as bt
        self.subtensor, self.netuid = bt.Subtensor(network=network), netuid

    def head(self):
        h = self.subtensor.substrate.get_chain_finalised_head()
        return self.subtensor.substrate.get_block_number(h)

    def frame(self, block):
        substrate = self.subtensor.substrate
        h = substrate.get_block_hash(block)
        epoch = int(substrate.query('SubtensorModule', 'SubnetEpochIndex', [self.netuid], block_hash=h).value)
        return {'block': block, 'block_hash': h, 'epoch_index': epoch}

    def snapshot(self, block=None):
        frame = self.frame(self.head() if block is None else block)
        s, h, block = self.subtensor.substrate, frame['block_hash'], frame['block']
        info = self.subtensor.get_metagraph_info(self.netuid, block=block)
        if info is None:
            raise RuntimeError('subnet_state_unavailable')
        def query(name, args=None):
            return s.query('SubtensorModule', name, [self.netuid] if args is None else args, block_hash=h).value
        keys = list(map(str, info.hotkeys))
        def axon_value(axon, key):
            return axon.get(key) if isinstance(axon, dict) else getattr(axon, key)
        permits = [key for key, allowed in zip(keys, info.validator_permit) if allowed]
        owner = str(query('SubnetOwnerHotkey'))
        return {**frame, 'uids': {key: i for i, key in enumerate(keys)},
                'coldkeys': dict(zip(keys, map(str, info.coldkeys))),
                'alpha_rao': {key: int(stake.rao) for key, stake in zip(keys, info.alpha_stake)},
                'validators': {key: int(stake.rao) for key, stake in zip(keys, info.total_stake)
                               if key in permits and stake.rao > 0},
                'permits': permits, 'observed_unix': time.time(),
                'endpoints': {key: {'host': str(ipaddress.ip_address(axon_value(axon, 'ip'))), 'port': int(axon_value(axon, 'port'))}
                              for key, axon in zip(keys, info.axons)
                              if axon_value(axon, 'port') and axon_value(axon, 'ip') != '0.0.0.0'},
                'last_update': dict(zip(keys, map(int, info.last_update))),
                'burn_uid': keys.index(owner) if owner in keys else None,
                'tempo': int(query('Tempo')), 'weights_rate_limit': int(query('WeightsSetRateLimit')),
                'commit_reveal': bool(query('CommitRevealWeightsEnabled')),
                'reveal_epochs': int(query('RevealPeriodEpochs')),
                'min_weights': int(query('MinAllowedWeights')), 'max_weight': int(query('MaxWeightsLimit'))}

    def applied_weights(self, hotkey, snapshot):
        uid = snapshot['uids'].get(hotkey)
        if uid is None:
            return None
        rows = self.subtensor.substrate.query('SubtensorModule', 'Weights', [self.netuid, uid],
                                              block_hash=snapshot['block_hash']).value
        return [(int(u), int(w)) for u, w in rows if int(w) > 0]

    def close(self):
        self.subtensor.close()
