import asyncio
import json

import pytest

from witness.score.oracle import perfect_reconstruction
from witness.subnet.chain import InMemoryChainAdapter
from witness.subnet.validator import ValidatorConfig, WitnessValidator, build_weight_vector


def allocate(uids=(7, 3, 240), ema=None, current=None, responders=None, **kwargs):
    return build_weight_vector(
        list(uids), {7: 0.8, 3: 0.2, 240: 9} if ema is None else ema,
        {7, 3, 240} if responders is None else responders,
        burn_uid=240, burn_rate=0.7, weight_policy="winner-takes-all",
        round_scores={7: 0.1, 3: 0.5, 240: 1} if current is None else current,
        **kwargs,
    )


def test_winner_uses_ema_and_owner_never_competes():
    assert allocate() == pytest.approx([0.3, 0, 0.7])


def test_zero_reward_or_offline_former_winner_cannot_win():
    assert allocate(current={7: 0, 3: 0.1}) == pytest.approx([0, 0.3, 0.7])
    assert allocate(responders={3}) == pytest.approx([0, 0.3, 0.7])


def test_tie_is_deterministic_independent_of_endpoint_order():
    assert allocate(ema={7: 0.4, 3: 0.4}) == pytest.approx([0, 0.3, 0.7])
    assert allocate(uids=(240, 3, 7), ema={7: 0.4, 3: 0.4}) == pytest.approx([0.7, 0.3, 0])


@pytest.mark.parametrize("kwargs", [{"current": {}}, {"responders": set()}, {"ema": {}}])
def test_no_eligible_miner_burns_everything(kwargs):
    assert allocate(**kwargs) == [0, 0, 1]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_reward_history_fails_closed(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        allocate(ema={7: value})


@pytest.mark.parametrize("uids", [(7, 3), (7, 7, 240), (-1, 3, 240)])
def test_missing_burn_or_invalid_uid_vector_fails_closed(uids):
    with pytest.raises(ValueError):
        allocate(uids=uids)


def test_current_round_is_required():
    with pytest.raises(ValueError, match="current round"):
        build_weight_vector([3, 240], {3: 1}, {3}, burn_uid=240, burn_rate=0.7,
                            weight_policy="winner-takes-all")


def test_ema_is_bound_to_hotkey_and_never_inherited_after_reregistration(tmp_path):
    v = WitnessValidator(InMemoryChainAdapter(), ValidatorConfig(
        round_root=tmp_path, weight_policy="winner-takes-all", burn_uid=240, burn_rate=0.7))
    v._save_ema({7: 0.9, 3: 0.1}, {"7": "old-key", "3": "stable-key"})
    assert v._load_ema({"7": "new-key", "3": "stable-key"}) == {"3": 0.1}
    raw = json.loads(v._ema_path.read_text())
    del raw["hotkeys"]
    v._ema_path.write_text(json.dumps(raw))
    assert v._load_ema({"7": "old-key"}) == {}


def test_missing_burn_target_fails_before_preparing_scenes(tmp_path, monkeypatch):
    v = WitnessValidator(InMemoryChainAdapter(), ValidatorConfig(
        round_root=tmp_path, weight_policy="winner-takes-all", burn_uid=240))
    monkeypatch.setattr(v, "_prepare_scenes", lambda *a: pytest.fail("prepared scenes"))
    with pytest.raises(ValueError, match="burn UID is missing"):
        asyncio.run(v.run_round())


def test_ambiguous_submission_is_not_retried_or_overwritten(tmp_path, generated_scenes):
    chain = InMemoryChainAdapter()
    chain.add_miner(240, lambda task: None)
    v = WitnessValidator(chain, ValidatorConfig(
        round_root=tmp_path, scene_count=1, source_scenes=(generated_scenes / "scene_101",),
        tool_host="127.0.0.1", tool_port=0, weight_policy="winner-takes-all", burn_uid=240,
        burn_rate=0.7))

    def interrupted(uids, weights):
        intent = json.loads((tmp_path / "weight-submission.json").read_text())
        assert intent["weight_submission"]["status"] == "prepared"
        raise TimeoutError("do not expose this transport message")

    chain.set_weights = interrupted
    first = asyncio.run(v.run_round())
    assert first["weight_submission"]["status"] == "unknown"
    with pytest.raises(RuntimeError, match="Unresolved weight submission"):
        asyncio.run(v.run_round())
    assert len(list(tmp_path.glob("round_*/round.json"))) == 1
    assert "do not expose" not in (tmp_path / "weight-submission.json").read_text()


def test_full_round_sends_70_30_to_one_winner_and_preserves_scorer(tmp_path, generated_scenes):
    chain = InMemoryChainAdapter()
    lookup = {}

    async def oracle(task):
        truth = json.loads((lookup[task.scene_id] / "scene.json").read_text())
        task.reconstruction = perfect_reconstruction(truth)
        task.trace_summary = {"status": "ok"}
        return task

    async def no_response(task):
        return None

    chain.add_miner(7, oracle, hotkey="seven")
    chain.add_miner(3, oracle, hotkey="three")
    chain.add_miner(240, no_response, hotkey="burn")
    v = WitnessValidator(chain, ValidatorConfig(
        round_root=tmp_path, scene_count=1, source_scenes=(generated_scenes / "scene_101",),
        tool_host="127.0.0.1", tool_port=0, weight_policy="winner-takes-all", burn_uid=240,
        burn_rate=0.7, set_weights_enabled=False), dry_scene_lookup=lookup)
    result = asyncio.run(v.run_round())
    weights = dict(zip(result["weights"]["uids"], result["weights"]["values"]))
    assert weights == pytest.approx({3: 0.3, 7: 0, 240: 0.7})
    assert result["weight_policy"]["winner_uid"] == 3
    assert result["scoring_identity"]["version"] == "1.0.0"
    assert result["weight_submission"]["status"] == "disabled"
    assert chain.weight_history == []
    assert json.loads(v._ema_path.read_text())["hotkeys"] == {"3": "three", "7": "seven", "240": "burn"}
