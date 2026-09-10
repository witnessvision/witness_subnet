import asyncio
import json

import pytest
from typer.testing import CliRunner

from witness.subnet.chain import InMemoryChainAdapter
from witness.subnet.history import rolling_mean_ema
from witness.subnet.mainnet import mainnet_config
from witness.subnet.validator import ValidatorConfig, WitnessValidator, app, build_weight_vector


def advance(samples, *, window=5, alpha=.1):
    prior, history = {}, {}
    results = []
    for values in samples:
        means, scores, history = rolling_mean_ema(values, prior, history, window=window, alpha=alpha)
        prior = {str(uid): score for uid, score in scores.items()}
        results.append((means, scores, history))
    return results


def test_first_positive_mean_bootstraps_after_zero_rounds():
    results = advance([{7: 0}, {7: 0}, {7: .9}, {7: .9}], window=3)
    assert results[1][1][7] == 0
    assert results[2][0][7] == pytest.approx(.3)
    assert results[2][1][7] == pytest.approx(.3)  # not .03, and not discarded
    assert results[3][0][7] == pytest.approx(.6)
    assert results[3][1][7] == pytest.approx(.33)
    assert results[3][2] == {"7": [0, .9, .9]}


def test_window_expires_raw_samples_while_ema_retains_longer_memory():
    means, ema, history = advance([{7: 1}, {7: 0}, {7: 0}], window=2)[-1]
    assert history == {"7": [0, 0]}
    assert means[7] == 0
    assert ema[7] == pytest.approx(.855)


def test_one_spike_does_not_replace_consistently_better_winner():
    _, ema, _ = advance([{7: .6, 3: .5}, {7: .55, 3: 1}])[-1]
    assert ema == pytest.approx({7: .5975, 3: .525})
    weights = build_weight_vector([7, 3, 240], ema, {7, 3}, burn_uid=240,
        burn_rate=.7, weight_policy="winner-takes-all", round_scores={7: .55, 3: 1})
    assert weights == pytest.approx([.3, 0, .7])
    # Current all-zero rounds must never pay an old winner.
    assert build_weight_vector([7, 3, 240], ema, {7, 3}, burn_uid=240,
        burn_rate=.7, weight_policy="winner-takes-all", round_scores={7: 0, 3: 0}) == [0, 0, 1]


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError, match="positive integer"):
        rolling_mean_ema({7: .5}, {}, {}, window=window, alpha=.1)
    with pytest.raises(ValueError, match="positive integer"):
        ValidatorConfig(score_window=window).validate()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_sample_fails_closed(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rolling_mean_ema({7: value}, {}, {}, window=5, alpha=.1)


def validator(root, **kwargs):
    return WitnessValidator(InMemoryChainAdapter(), ValidatorConfig(
        round_root=root, score_window=5, ema_alpha=.1, **kwargs))


def test_restart_preserves_window_and_hotkey_replacement_resets_it(tmp_path):
    v = validator(tmp_path)
    v._save_ema({7: .4, 3: .2}, {"7": "old", "3": "stable"}, {"7": [.3, .5], "3": [.2]})
    loaded = validator(tmp_path)._load_score_state({"7": "old", "3": "stable"})
    assert loaded == ({"7": .4, "3": .2}, {"7": [.3, .5], "3": [.2]}, {"7": "old", "3": "stable"})
    assert validator(tmp_path)._load_score_state({"7": "new", "3": "stable"}) == (
        {"3": .2}, {"3": [.2]}, {"3": "stable"})


@pytest.mark.parametrize("changed", [{"score_window": 3}, {"ema_alpha": .2}])
def test_aggregation_change_is_rejected_before_a_round(tmp_path, changed):
    validator(tmp_path)._save_ema({7: .4}, {"7": "seven"}, {"7": [.4]})
    config = ValidatorConfig(round_root=tmp_path, score_window=5, ema_alpha=.1)
    for name, value in changed.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match="aggregation identity"):
        asyncio.run(WitnessValidator(InMemoryChainAdapter(), config).run_round())
    assert not list(tmp_path.glob("round_*"))


def test_old_single_ema_cannot_be_silently_reused_as_window_history(tmp_path):
    v = validator(tmp_path)
    (tmp_path / "ema.json").write_text(json.dumps({
        "scores": {"7": .4}, "scoring_identity": v.scoring_identity}))
    with pytest.raises(ValueError, match="aggregation identity"):
        v._load_score_state()


def test_offline_miner_gets_zero_round_without_erasing_history(tmp_path, generated_scenes):
    chain = InMemoryChainAdapter()
    chain.add_miner(240, lambda task: None, hotkey="burn")
    config = mainnet_config(round_root=tmp_path, burn_uid=240, tool_host="127.0.0.1",
        tool_port=0, tool_public_url=None, set_weights_enabled=False)
    config.scene_count = 1
    config.source_scenes = (generated_scenes / "scene_101",)
    v = WitnessValidator(chain, config)
    v._save_ema({7: .8}, {"7": "seven"}, {"7": [.8]})
    result = asyncio.run(v.run_round())
    assert result["weights"] == {"uids": [240], "values": [1]}
    assert result["aggregation_identity"]["window_rounds"] == 5
    assert result["scoring_identity"]["version"] == "1.0.0"
    scores, history, keys = WitnessValidator(chain, config)._load_score_state({"7": "seven", "240": "burn"})
    assert scores["7"] == pytest.approx(.76)
    assert history["7"] == [.8, 0]
    assert keys["7"] == "seven"
    assert chain.weight_history == []


def test_mainnet_cli_exposes_window_and_alpha(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from witness.subnet import mainnet, validator as module
    captured = []
    monkeypatch.setattr(mainnet, "MainnetChainAdapter", lambda **kw:
        SimpleNamespace(burn_uid=240, close=lambda: None))
    class FakeValidator:
        def __init__(self, chain, config):
            captured.append(config)
        async def run_round(self):
            return {"round_id": "test", "weights": {}, "weight_submission": {"status": "disabled"}}
    monkeypatch.setattr(module, "WitnessValidator", FakeValidator)
    result = CliRunner().invoke(app, ["--mainnet", "--once", "--no-set-weights",
        "--tool-public-url", "http://127.0.0.1:8765", "--score-window", "3", "--ema-alpha", "0.2"])
    assert result.exit_code == 0, result.output
    assert captured[0].score_window == 3 and captured[0].ema_alpha == .2
    assert not captured[0].set_weights_enabled
