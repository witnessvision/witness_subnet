"""Pinned Finney identity reads; the evaluator cannot submit weights."""
import threading

from witness.subnet.chain import BittensorChainAdapter
from witness.subnet.mainnet import FINNEY_GENESIS


class ProductionChain(BittensorChainAdapter):
    def __init__(self, **kwargs):
        self.registry_lock = threading.RLock()
        super().__init__(**kwargs, with_dendrite=False)
        if self.netuid != 20 or self.subtensor.substrate.get_block_hash(0) != FINNEY_GENESIS:
            self.close()
            raise ValueError('wrong_network')

    def registered_hotkey(self, uid):
        with self.registry_lock:
            s = self.subtensor.substrate
            h = s.get_chain_finalised_head()
            return s.query('SubtensorModule', 'Keys', [self.netuid, uid], block_hash=h).value

    def miner_endpoints(self):
        with self.registry_lock:
            return super().miner_endpoints()

    def epoch_state(self):
        with self.registry_lock:
            return super().epoch_state()

    def set_weights(self, *args, **kwargs):
        raise RuntimeError('evaluation_service_cannot_write_weights')
