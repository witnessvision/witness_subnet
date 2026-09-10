import asyncio
import json
from pathlib import Path
import pytest
from witness.subnet.scheduling import run_epoch_rounds


class StopPolling(BaseException):
    pass


class Chain:
    epoch = 10
    def epoch_state(self):
        return {'netuid': 20, 'epoch_index': self.epoch, 'tempo': 360}


def test_once_per_epoch_and_restart_does_not_repeat(tmp_path, monkeypatch):
    chain = Chain()
    calls = []
    sleeps = 0
    async def run():
        calls.append(chain.epoch)
        return {'round_id': str(chain.epoch)}
    async def sleep(_):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            chain.epoch += 1
        if sleeps == 3:
            raise StopPolling
    monkeypatch.setattr('witness.subnet.scheduling.asyncio.sleep', sleep)
    path = tmp_path/'state.json'
    with pytest.raises(StopPolling):
        asyncio.run(run_epoch_rounds(chain, run, path))
    assert calls == [10, 11]
    async def stop(_):
        raise StopPolling
    monkeypatch.setattr('witness.subnet.scheduling.asyncio.sleep', stop)
    with pytest.raises(StopPolling):
        asyncio.run(run_epoch_rounds(chain, run, path))
    assert calls == [10, 11]
    assert json.loads(path.read_text())['status'] == 'waiting_for_epoch'


def test_long_round_skips_crossed_epochs(tmp_path, monkeypatch):
    chain = Chain()
    calls = []
    polls = 0
    async def run():
        calls.append(chain.epoch)
        chain.epoch += 2
    async def sleep(_):
        nonlocal polls
        polls += 1
        if polls == 2:
            raise StopPolling
    monkeypatch.setattr('witness.subnet.scheduling.asyncio.sleep', sleep)
    path=tmp_path/'state.json'
    with pytest.raises(StopPolling):
        asyncio.run(run_epoch_rounds(chain, run, path))
    assert calls == [10]
    assert json.loads(path.read_text())['last_attempted_epoch'] == 12


def test_failed_round_waits_instead_of_retrying_same_epoch(tmp_path, monkeypatch):
    chain=Chain();calls=[];polls=0
    async def run():
        calls.append(chain.epoch)
        raise RuntimeError('failed round')
    async def sleep(_):
        nonlocal polls
        polls+=1
        if polls==2:raise StopPolling
    monkeypatch.setattr('witness.subnet.scheduling.asyncio.sleep',sleep)
    path=tmp_path/'state.json'
    with pytest.raises(StopPolling):asyncio.run(run_epoch_rounds(chain,run,path))
    assert calls==[10]
    assert json.loads(path.read_text())['last_round_status']=='failed'


def test_epoch_storage_is_pinned_to_finalized_block():
    from types import SimpleNamespace
    from witness.subnet.chain import BittensorChainAdapter
    seen=[]
    values={'Tempo':360,'LastEpochBlock':1000,'PendingEpochAt':1200,'SubnetEpochIndex':9}
    class Substrate:
        def get_chain_finalised_head(self):return 'finalized-hash'
        def get_block_number(self,h):
            assert h=='finalized-hash'
            return 1100
        def query(self,module,name,params,block_hash):
            seen.append((module,params,block_hash))
            return SimpleNamespace(value=values[name])
    chain=object.__new__(BittensorChainAdapter)
    chain.netuid=20
    chain.subtensor=SimpleNamespace(substrate=Substrate())
    state=chain.epoch_state()
    assert state['epoch_index']==9 and state['next_epoch_block']==1200
    assert seen==[('SubtensorModule',[20],'finalized-hash')]*4
