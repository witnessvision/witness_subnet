import asyncio
import json
from types import SimpleNamespace

import pytest

from witness.subnet.burn import burn_state, run_full_burn
from witness.subnet.chain import WeightSubmission


class Substrate:
    def get_chain_finalised_head(self):
        return 'finalized'

    def get_block_number(self, block_hash):
        return 200

    def query(self, module, name, params, block_hash):
        assert block_hash == 'finalized'
        values = {
            'SubnetOwner': 'owner', 'SubnetOwnerHotkey': 'owner-hotkey',
            'Uids': 1, 'Keys': 'owner-hotkey', 'Owner': 'owner',
            'RecycleOrBurn': 'Burn', 'ValidatorPermit': [False, True],
            'LastUpdate': [0, 0], 'WeightsSetRateLimit': 100,
            'Weights': [(1, 65535)],
        }
        return SimpleNamespace(value=values[name])


def chain():
    return SimpleNamespace(netuid=20, validator_hotkey='validator',
                           subtensor=SimpleNamespace(substrate=Substrate()))


def test_owner_target_and_applied_weight_are_read_at_one_finalized_block():
    state = burn_state(chain(), None)
    assert state['burn_uid'] == 1
    assert state['weights_applied'] is True
    assert state['weights'] == {'uids': [1], 'values': [1.0]}


@pytest.mark.parametrize('storage,value', [('Owner', 'someone-else'),
                                         ('RecycleOrBurn', 'Recycle'),
                                         ('ValidatorPermit', [False, False])])
def test_unsafe_destination_or_missing_permit_fails(storage, value):
    adapter = chain()
    original = adapter.subtensor.substrate.query
    adapter.subtensor.substrate.query = lambda module, name, params, block_hash: (
        SimpleNamespace(value=value) if name == storage else original(module, name, params, block_hash))
    with pytest.raises(ValueError):
        burn_state(adapter, None)


def test_submission_intent_persisted_before_network_and_unknown_blocks_restart(tmp_path):
    path = tmp_path / 'burn-state.json'
    adapter = chain()

    def uncertain(**kwargs):
        assert json.loads(path.read_text())['weight_submission']['status'] == 'prepared'
        assert kwargs == {'uids': [1], 'weights': [1.0]}
        raise TimeoutError('transport interrupted')

    adapter.set_weights = uncertain
    with pytest.raises(RuntimeError, match='Ambiguous'):
        asyncio.run(run_full_burn(adapter, state_path=path, once=True))
    assert json.loads(path.read_text())['weight_submission']['status'] == 'unknown'
    with pytest.raises(RuntimeError, match='Unresolved'):
        asyncio.run(run_full_burn(adapter, state_path=path, once=True))


def test_rate_limit_waits_without_submitting(tmp_path):
    adapter = chain()
    original = adapter.subtensor.substrate.query
    adapter.subtensor.substrate.query = lambda module, name, params, block_hash: (
        SimpleNamespace(value=[0, 180]) if name == 'LastUpdate' else original(module, name, params, block_hash))
    adapter.set_weights = lambda **kwargs: pytest.fail('rate-limited submission')
    path = tmp_path / 'burn-state.json'
    asyncio.run(run_full_burn(adapter, state_path=path, once=True))
    assert json.loads(path.read_text())['blocks_until_submission'] == 81


def test_disabled_weights_observes_without_overwriting_unresolved_submission(tmp_path):
    adapter = chain()
    adapter.set_weights = lambda **kwargs: pytest.fail('disabled submission called')
    path = tmp_path / 'burn-state.json'
    original = {'weight_submission': {'status': 'unknown'}}
    path.write_text(json.dumps(original))
    asyncio.run(run_full_burn(adapter, state_path=path, once=True, set_weights_enabled=False))
    assert json.loads(path.read_text()) == original
    observation = json.loads(path.with_name('burn-observation.json').read_text())
    assert observation['weight_submission']['status'] == 'disabled'
    assert observation['weights_applied'] is True
