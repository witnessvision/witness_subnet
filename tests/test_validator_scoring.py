import asyncio
import json
from pathlib import Path

import pytest

from witness.score.oracle import perfect_reconstruction
from witness.subnet.chain import InMemoryChainAdapter
from witness.subnet.validator import ValidatorConfig, WitnessValidator
from witness.tools.client import WitnessClient

ROOT = Path(__file__).parents[1]


def test_candidate_requires_explicit_diagnostic_mode():
    with pytest.raises(ValueError, match="diagnostic"):
        ValidatorConfig(score_version="1.7-candidate").validate()


@pytest.mark.parametrize("version", ["1.7-candidate", "1.9-candidate", "1.0.0", "1.1.0"])
@pytest.mark.parametrize("set_weights_enabled", [True, False])
def test_round_uses_candidate_and_observed_asr_without_label_transcript(tmp_path, version, set_weights_enabled):
    from test_subnet import _tiny_scene
    from witness.tools.transcripts import sha256
    source = _tiny_scene(tmp_path / "source", 101)
    (source / "observations").mkdir()
    (source / "observations/transcript.json").write_text(json.dumps({
        "schema_version": "1", "source": "decoded_audio",
        "video_sha256": sha256(source / "video.mp4"),
        "model_sha256": {"fixture": "abc"},
        "entries": [{"start": .2, "end": .6, "text": "Observed speech"}],
    }))
    chain = InMemoryChainAdapter()
    observed = []
    truth = json.loads((source / "scene.json").read_text())
    asr = json.loads((source / "observations/transcript.json").read_text())["entries"]

    async def oracle(task):
        client = WitnessClient(task.tool_base_url, session_id=task.session_id)
        transcript = client.get_transcript(0, client.get_meta().data["duration"])
        observed.extend(transcript.data["entries"])
        task.reconstruction = perfect_reconstruction(truth)
        # The public schema needs neither shot IDs nor generator text IDs/roles.
        for item in task.reconstruction["shots"]:
            item.pop("id", None)
        for item in task.reconstruction["intentional_errors"]:
            item.pop("text_id", None)
        for item in task.reconstruction["on_screen_text"]:
            item.pop("role", None)
        task.trace_summary = {"status": "ok"}
        return task

    chain.add_miner(1, oracle)
    validator = WitnessValidator(chain, ValidatorConfig(
        round_root=tmp_path, source_scenes=(source,), scene_count=1,
        benchmark_lock=None, allow_unlocked=version not in {"1.0.0", "1.1.0"}, score_version=version,
        transcript_source="asr", tool_host="127.0.0.1", tool_port=0,
        set_weights_enabled=set_weights_enabled,
    ))
    result = asyncio.run(validator.run_round())
    assert [r["text"] for r in observed] == [r["text"] for r in asr]
    row = result["miners"]["1"]["scenes"][0]
    assert row["score_version"] == version
    assert row["quality"] == 1
    assert row["gate"]["passed"] is True
    assert result["miners"]["1"]["metrics"]["quality"] == 1
    assert result["miners"]["1"]["metrics"]["valid_rate"] == 1
    if version in {"1.9-candidate", "1.0.0"}:
        assert row["gate"]["reward_factor"] == 1
        assert {"quality_v18", "reward"} <= result["scoring_identity"]["code_sha256"].keys()
    assert result["scoring_identity"]["mode"] == ("production" if version in {"1.0.0", "1.1.0"} else "diagnostic")
    assert row["cost"]["transcript_chars"] > 0
    assert result["scoring_identity"]["transcript_source"] == "asr"
    assert set(result["scene_seeds_revealed"][0]["input_sha256"]) == {"scene.json", "video.mp4", "observations/transcript.json"}
    assert result["weight_submission"]["status"] == ("simulated" if set_weights_enabled else "disabled")
    assert bool(chain.weight_history) is set_weights_enabled
    assert json.loads((tmp_path / "ema.json").read_text())["scoring_identity"] == result["scoring_identity"]


def test_mismatched_ema_is_rejected_before_query_or_round_creation(tmp_path):
    config = ValidatorConfig(round_root=tmp_path, benchmark_lock=None, allow_unlocked=True,
                             score_version="1.7-candidate", transcript_source="none")
    chain = InMemoryChainAdapter()
    validator = WitnessValidator(chain, config)
    (tmp_path / "ema.json").write_text(json.dumps({"scores": {"1": .9}, "scoring_identity": {"version": "1.6-candidate"}}))
    with pytest.raises(ValueError, match="EMA scoring identity"):
        asyncio.run(validator.run_round())
    assert not list(tmp_path.glob("round_*"))
    assert not chain.weight_history


def test_production_is_default_and_does_not_require_candidate_mode():
    config = ValidatorConfig()
    assert config.score_version == "1.1.0"
    assert config.benchmark_lock is None
    assert not config.allow_unlocked
    config.validate()


def test_promotion_does_not_silently_relabel_or_mix_candidate_ema(tmp_path):
    config = ValidatorConfig(round_root=tmp_path, transcript_source="none")
    prior = {"scores": {"1": .12}, "scoring_identity": {"version": "1.9-candidate"}}
    path = tmp_path / "ema.json"
    path.write_text(json.dumps(prior))
    chain = InMemoryChainAdapter()
    with pytest.raises(ValueError, match="EMA scoring identity"):
        asyncio.run(WitnessValidator(chain, config).run_round())
    assert json.loads(path.read_text()) == prior
    assert not list(tmp_path.glob("round_*"))
    assert not chain.weight_history
