import asyncio
import json
from types import SimpleNamespace

import bittensor as bt
import pytest
from pydantic import ValidationError

from test_protocol_transport import request
from test_subnet import _tiny_scene
from witness.score.oracle import perfect_reconstruction
from witness.subnet.chain import InMemoryChainAdapter
from witness.subnet.feedback import FeedbackReceiver
from witness.subnet.miner import WitnessMiner
from witness.subnet.protocol import RoundFeedback, WitnessFeedback
from witness.subnet.validator import ValidatorConfig, WitnessValidator


def feedback():
    return WitnessFeedback(report=RoundFeedback(
        round_id="test-round", validator_hotkey="validator", completed_at="2026-09-11T00:00:00Z",
        scorer_version="1.1.0", aggregation={"version": "1.1.0", "algorithm": "rolling_mean_ema",
            "window_rounds": 5, "alpha": .2, "bootstrap": "first_positive_mean"},
        weight_policy="winner-takes-all", burn_uid=240, burn_rate=.7, winner_uid=None,
        submission_status="disabled", miners=[]))


def test_feedback_signed_body_and_sdk_headers():
    synapse = feedback()
    digest = synapse.body_hash
    synapse.accepted = True
    assert synapse.body_hash == digest
    headers = synapse.to_headers()
    parsed = WitnessFeedback.from_headers(headers)
    assert parsed.computed_body_hash == digest
    with pytest.raises(ValidationError):
        WitnessFeedback.model_validate(parsed.model_dump())
    axon = SimpleNamespace(forward_class_types={"WitnessFeedback": WitnessFeedback})
    payload = synapse.model_dump()
    req = request(payload, headers)
    req.scope["path"] = "/WitnessFeedback"
    asyncio.run(bt.Axon.verify_body_integrity(axon, req))
    payload["report"]["burn_rate"] = .5
    req = request(payload, headers)
    req.scope["path"] = "/WitnessFeedback"
    with pytest.raises(ValueError, match="Hash mismatch"):
        asyncio.run(bt.Axon.verify_body_integrity(axon, req))
    payload["report"]["scene_seeds_revealed"] = ["private"]
    with pytest.raises(ValidationError, match="Extra inputs"):
        WitnessFeedback.model_validate(payload)


def test_base_and_feedback_callbacks_attach_to_real_sdk():
    axon = bt.Axon(wallet=SimpleNamespace(), ip="127.0.0.1", external_ip="127.0.0.1", port=8091)
    miner = WitnessMiner()
    try:
        axon.attach(forward_fn=miner.forward, blacklist_fn=miner.blacklist, priority_fn=miner.priority)
        axon.attach(forward_fn=miner.feedback.forward, blacklist_fn=miner.feedback.blacklist,
                    priority_fn=miner.feedback.priority)
        assert {"WitnessTask", "WitnessFeedback"} <= axon.forward_class_types.keys()
    finally:
        axon.thread_pool.shutdown(wait=False)


def test_signed_feedback_roundtrip_over_real_sdk_http(tmp_path, monkeypatch):
    import socket
    from bittensor_wallet import Keypair
    from witness.subnet.chain import BittensorChainAdapter, MinerEndpoint
    monkeypatch.setattr("bittensor.core.dendrite.networking.get_external_ip", lambda: "127.0.0.1")
    # Public development identities, held in memory only; no wallet files.
    caller, server = Keypair.create_from_uri("//Alice"), Keypair.create_from_uri("//Bob")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    axon = bt.Axon(wallet=SimpleNamespace(hotkey=server, coldkeypub=server),
                   ip="127.0.0.1", external_ip="127.0.0.1", port=port)
    receiver = FeedbackReceiver(chain=SimpleNamespace(is_validator=lambda h: h == caller.ss58_address),
                                feedback_dir=tmp_path)
    axon.attach(forward_fn=receiver.forward, blacklist_fn=receiver.blacklist, priority_fn=receiver.priority)
    axon.start()
    async def check():
        for _ in range(200):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(.01)
        else:
            pytest.fail("test axon did not start")
        chain = object.__new__(BittensorChainAdapter)
        async with bt.Dendrite(wallet=caller) as dendrite:
            chain.dendrite = dendrite
            synapse = feedback()
            synapse.report.validator_hotkey = caller.ss58_address
            result = await chain.send_feedback(MinerEndpoint(7, server.ss58_address, axon.info()),
                                                synapse, timeout=5)
            assert result == {"status": "accepted", "status_code": 200}
    try:
        asyncio.run(check())
        assert receiver.latest[caller.ss58_address].round_id == "test-round"
        assert len(list(tmp_path.glob("*.json"))) == 1
    finally:
        axon.stop()
        axon.thread_pool.shutdown(wait=False)


def test_receiver_binds_signed_identity_and_retains_per_validator(tmp_path):
    chain = SimpleNamespace(is_validator=lambda h: h in {"validator", "second"})
    receiver = FeedbackReceiver(chain=chain, feedback_dir=tmp_path)
    async def check():
        synapse = feedback()
        synapse.dendrite.hotkey = "other"
        assert not (await receiver.forward(synapse)).accepted
        synapse.dendrite.hotkey = "second"
        assert not (await receiver.forward(synapse)).accepted  # claimed identity is different
        synapse.dendrite.hotkey = "validator"
        assert (await receiver.forward(synapse)).accepted
        synapse.report.validator_hotkey = "second"
        synapse.dendrite.hotkey = "second"
        assert (await receiver.forward(synapse)).accepted
        assert receiver.latest["validator"].validator_hotkey == "validator"
        synapse.report.round_id = "newer"
        await receiver.forward(synapse)
    asyncio.run(check())
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in files)
    assert {json.loads(p.read_text())["round_id"] for p in files} == {"test-round", "newer"}


@pytest.mark.parametrize("set_weights", [True, False])
def test_completed_round_feedback_is_complete_private_safe_and_delivery_is_independent(tmp_path, set_weights):
    source = _tiny_scene(tmp_path / "scene", 101)
    truth = json.loads((source / "scene.json").read_text())
    chain = InMemoryChainAdapter(validator_hotkey="validator")
    async def respond(task):
        task.reconstruction = perfect_reconstruction(truth)
        task.trace_summary = {"status": "ok", "private_payload": "DO_NOT_SEND"}
        return task
    for uid in (7, 8, 9, 240):
        chain.add_miner(uid, respond)
    receiver = FeedbackReceiver(feedback_dir=tmp_path / "received")
    async def receive(synapse):
        assert len(chain.weight_history) == int(set_weights)
        assert (tmp_path / "rounds/ema.json").is_file()
        return await receiver.forward(synapse)
    async def fail(synapse):
        raise RuntimeError("untrusted response")
    chain.feedback_handlers = {7: receive, 8: fail}
    config = ValidatorConfig(round_root=tmp_path / "rounds", scene_count=1, source_scenes=(source,),
        score_window=5, ema_alpha=.2, weight_policy="winner-takes-all", burn_uid=240, burn_rate=.7,
        set_weights_enabled=set_weights, tool_host="127.0.0.1", tool_port=0, transcript_source="none")
    artifact = asyncio.run(WitnessValidator(chain, config).run_round())
    assert artifact["ema_updated"]
    assert artifact["weight_submission"]["status"] == ("simulated" if set_weights else "disabled")
    assert artifact["feedback"]["status"] == "completed"
    deliveries = artifact["feedback"]["deliveries"]
    assert deliveries["7"]["status"] == "accepted"
    assert deliveries["8"]["status"] == "failed"
    assert deliveries["9"]["status_code"] == 404
    assert deliveries["240"]["status"] == "skipped_burn"
    public = receiver.latest["validator"]
    assert len(public.miners) == 4
    assert public.aggregation.alpha == .2 and public.weights_applied is None
    for miner in public.miners:
        assert miner.round_score == artifact["miners"][str(miner.uid)]["round_score"]
        assert miner.scenes[0].gate.threshold == .4
        assert miner.scenes[0].duplicate_count == 4
    encoded = public.model_dump_json()
    for private in ("reconstruction", "seed", "session_id", "DO_NOT_SEND", "trace_summary", "diagnostics"):
        assert private not in encoded
    assert sum(m.weight for m in public.miners) == pytest.approx(1)
