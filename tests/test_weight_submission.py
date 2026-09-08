import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from witness.subnet.chain import BittensorChainAdapter, InMemoryChainAdapter, WeightSubmission
from witness.subnet.validator import ValidatorConfig, WitnessValidator


@pytest.mark.parametrize("success,block,finalized,expected", [
    (False, None, False, "rejected"), (True, None, False, "submitted"),
    (True, "block", False, "included"), (True, "block", True, "finalized"),
])
def test_sdk_receipt_evidence_controls_reported_confirmation(success, block, finalized, expected):
    response = SimpleNamespace(success=success, error=None, extrinsic_function="commit_weights",
                               extrinsic_receipt=SimpleNamespace(block_hash=block, extrinsic_hash="tx", finalized=finalized))
    calls = []
    adapter = object.__new__(BittensorChainAdapter)
    adapter.wallet = object()
    adapter.netuid = 123
    def submit(**kwargs):
        calls.append(kwargs)
        return response
    adapter.subtensor = SimpleNamespace(set_weights=submit)
    result = adapter.set_weights([0], [1.0])
    assert result.status == expected
    assert result.as_dict()["weights_applied"] is None  # commitment inclusion is not reveal
    assert calls[0]["wait_for_finalization"] is True
    assert calls[0]["max_attempts"] == 1


@pytest.mark.parametrize("fail", [False, True])
def test_round_persists_intent_and_failure_without_claiming_chain_success(tmp_path, fail, generated_scenes):
    chain = InMemoryChainAdapter()
    source = generated_scenes / "scene_101"
    validator = WitnessValidator(chain, ValidatorConfig(
        round_root=tmp_path, benchmark_lock=None, source_scenes=(source,),
        scene_count=1, tool_host="127.0.0.1", tool_port=0,
    ))
    def submit(uids, weights):
        paths = list(tmp_path.glob("round_*/round.json"))
        assert len(paths) == 1
        before = json.loads(paths[0].read_text())
        assert before["weight_submission"]["status"] == "prepared"
        assert before["ema_updated"] is False
        if fail:
            raise TimeoutError("do not persist arbitrary exception text")
        return WeightSubmission("rejected")
    chain.set_weights = submit
    artifact = asyncio.run(validator.run_round())
    assert artifact["weight_submission"]["status"] == ("unknown" if fail else "rejected")
    saved = json.loads(next(tmp_path.glob("round_*/round.json")).read_text())
    assert saved == artifact
    assert "do not persist" not in json.dumps(saved)
    assert artifact["ema_updated"] is True
