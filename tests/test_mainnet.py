import asyncio
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from witness.subnet.chain import BittensorChainAdapter, InMemoryChainAdapter, MinerEndpoint, WeightSubmission
from witness.subnet.mainnet import MainnetChainAdapter, mainnet_config
from witness.subnet.validator import WitnessValidator, app


def adapter(monkeypatch):
    chain = object.__new__(MainnetChainAdapter)
    chain.netuid, chain.burn_uid = 20, 240
    chain.selected = {7: "miner", 240: "burn"}
    state = {"block_hash": "finalized", "blocks_until_submission": 0}
    monkeypatch.setattr("witness.subnet.mainnet.burn_state", lambda *a: state)

    def query(module, name, args, block_hash):
        assert block_hash == "finalized"
        value = {"SubnetOwner": "owner", "MinAllowedWeights": 1, "MaxWeightsLimit": 65535}.get(name)
        if name == "Keys":
            value = {7: "miner", 8: "other-owner", 240: "burn"}[args[-1]]
        if name == "Owner":
            value = "someone" if args[0] == "miner" else "owner"
        return SimpleNamespace(value=value)

    chain.subtensor = SimpleNamespace(substrate=SimpleNamespace(query=query))
    sent = []
    monkeypatch.setattr(BittensorChainAdapter, "set_weights", lambda self, uids, weights:
                        sent.append((uids, weights)) or WeightSubmission("simulated"))
    return chain, state, sent


def test_population_includes_all_serving_miners_and_excludes_other_owner_hotkeys(monkeypatch):
    chain, _, _ = adapter(monkeypatch)
    endpoints = [MinerEndpoint(7, "miner", SimpleNamespace(is_serving=True)),
                 MinerEndpoint(8, "other-owner", SimpleNamespace(is_serving=True)),
                 MinerEndpoint(9, "offline", SimpleNamespace(is_serving=False)),
                 MinerEndpoint(240, "burn")]
    monkeypatch.setattr(BittensorChainAdapter, "miner_endpoints", lambda self: endpoints)
    assert [e.uid for e in chain.miner_endpoints()] == [7, 240]


def test_submit_exact_split_and_full_burn(monkeypatch):
    chain, _, sent = adapter(monkeypatch)
    assert chain.set_weights([7, 240], [.3, .7]).status == "simulated"
    assert chain.set_weights([7, 240], [0, 1]).status == "simulated"
    assert len(sent) == 2


@pytest.mark.parametrize("uids,weights", [([7, 240], [.5, .5]), ([7], [1]),
                                          ([7, 7, 240], [.1, .2, .7]),
                                          ([7, 240], [float("nan"), .7])])
def test_invalid_mainnet_vector_never_reaches_sdk(monkeypatch, uids, weights):
    chain, _, sent = adapter(monkeypatch)
    with pytest.raises(ValueError):
        chain.set_weights(uids, weights)
    assert sent == []


def test_rate_limit_records_definite_non_submission(monkeypatch):
    chain, state, sent = adapter(monkeypatch)
    state["blocks_until_submission"] = 80
    assert chain.set_weights([7, 240], [.3, .7]).status == "rate_limited"
    assert sent == []


def test_changed_hotkey_never_receives_previous_miner_weight(monkeypatch):
    chain, _, sent = adapter(monkeypatch)
    chain.selected[7] = "previous-miner"
    with pytest.raises(RuntimeError, match="hotkey changed"):
        chain.set_weights([7, 240], [.3, .7])
    assert sent == []


def test_preset_is_cpu_only_without_asr_or_hosted_generation(tmp_path):
    config = mainnet_config(round_root=tmp_path, burn_uid=240, tool_host="127.0.0.1", tool_port=0,
                            tool_public_url=None, set_weights_enabled=False)
    config.validate()
    assert config.scene_count == 5 and config.programmatic_share == 1
    assert config.pool_manifest is None and config.source_scenes == ()
    assert config.transcript_source == "none" and config.score_version == "1.0.0"
    assert config.weight_policy == "winner-takes-all" and config.burn_rate == .7
    assert config.epoch_aligned and not config.set_weights_enabled


@pytest.mark.parametrize("args", [["--mainnet"], ["--mainnet", "--burn-only"], ["--mainnet", "--dry-run"]])
def test_bad_mainnet_invocation_fails_before_wallet_or_rpc(monkeypatch, args):
    monkeypatch.setattr(MainnetChainAdapter, "__init__", lambda *a, **kw: pytest.fail("opened chain"))
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2


def test_parallel_queries_have_separate_sessions_and_never_overlap_same_miner(tmp_path, generated_scenes):
    chain = InMemoryChainAdapter()
    active, peak, sessions, running_uids = 0, 0, set(), set()

    async def respond(uid, task):
        nonlocal active, peak
        assert uid not in running_uids and task.session_id not in sessions
        running_uids.add(uid)
        sessions.add(task.session_id)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.03)
        active -= 1
        running_uids.remove(uid)
        task.reconstruction = {}
        task.trace_summary = {"status": "ok"}
        return task

    for uid in (7, 8, 9, 10, 240):
        chain.add_miner(uid, lambda task, uid=uid: respond(uid, task))
    config = mainnet_config(round_root=tmp_path, burn_uid=240, tool_host="127.0.0.1", tool_port=0,
                            tool_public_url=None, set_weights_enabled=False)
    config.scene_count = 2
    config.source_scenes = (generated_scenes / "scene_101",)
    result = asyncio.run(WitnessValidator(chain, config).run_round())
    assert peak == 4 and len(sessions) == 10
    assert result["weights"]["values"][-1] == 1  # empty responses cannot win
    assert chain.weight_history == []
