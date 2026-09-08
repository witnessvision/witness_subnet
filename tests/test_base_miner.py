import asyncio
import pytest
from test_subnet import _task
from witness.subnet.miner import WitnessMiner


def test_base_miner_has_no_inference_or_answers():
    response = asyncio.run(WitnessMiner().forward(_task()))
    assert response.trace_summary == {"status": "ok"}
    assert response.reconstruction["qa"] == {}
    assert all(response.reconstruction[key] == [] for key in
               ("events", "dialogue", "shots", "on_screen_text", "audio_events", "intentional_errors"))


def test_capacity_timeout_and_recovery():
    class SlowMiner(WitnessMiner):
        async def reconstruct(self, task):
            await asyncio.sleep(60)
    async def check():
        miner = SlowMiner(max_deadline_s=.05)
        first = asyncio.create_task(miner.forward(_task()))
        await asyncio.sleep(.01)
        busy = await miner.forward(_task())
        assert busy.trace_summary["status"] == "busy"
        assert busy.reconstruction == {}
        assert (await first).trace_summary["status"] == "deadline_exceeded"
        assert miner._active == 0
        async def ready(task): return {"qa": {}}
        miner.reconstruct = ready
        assert (await miner.forward(_task())).trace_summary["status"] == "ok"
    asyncio.run(check())


def test_error_and_cancellation_release_capacity():
    class FailingMiner(WitnessMiner):
        async def reconstruct(self, task):
            raise RuntimeError("private diagnostic must not be returned")
    async def check():
        miner = FailingMiner()
        response = await miner.forward(_task())
        assert response.reconstruction == {}
        assert response.trace_summary == {"status": "error", "error_type": "RuntimeError"}
        async def wait(task): await asyncio.sleep(60)
        miner.reconstruct = wait
        running = asyncio.create_task(miner.forward(_task()))
        await asyncio.sleep(.01)
        running.cancel()
        with pytest.raises(asyncio.CancelledError): await running
        assert miner._active == 0
    asyncio.run(check())
