import asyncio
import copy
import json
import socket
import sys
from types import SimpleNamespace

import bittensor as bt
from bittensor_wallet import Keypair
import pytest

from witness.events import (Event, EventsTaskSpec, canonical_bytes, content_hash,
                            observation_budget, response_hash, scored_text, validate_events)
from witness.score_v5_0_0 import score_events
from witness.subnet.events_state import EventsState
from witness.subnet.events_transport import parse_response, validate_response, ResponseRejected
from witness.subnet.chain import BittensorChainAdapter, MinerEndpoint
from witness.subnet.miner import WitnessMiner
from witness.subnet.processes import run_process
from witness.subnet.protocol import EventsFeedback, WitnessFeedback, WitnessTask


def task():
    return WitnessTask(task_id="a"*32, tool_base_url="http://127.0.0.1:12345", session_id="b"*32,
        scene_id="c"*32, seed_commitment="d"*64, budget=observation_budget(60),
        task_spec=EventsTaskSpec(duration=60., fps=30., has_audio=True).model_dump(), deadline_s=180.)


def event(timestamp=20., action="abre", actor="la persona", objects=None, details=None):
    return {"timestamp": timestamp, "actor": actor, "action": action,
            "objects": ["la puerta"] if objects is None else objects,
            "details": [] if details is None else details}


def response(events=None):
    return {"schema_version": "5.0", "events": [event()] if events is None else events}


def decision(pred=0, ref=0, relation="supported", value=None):
    return {"prediction": pred, "reference": ref, "relation": relation,
            "fields": {k: relation for k in scored_text(Event.model_validate(value or event()))}}


REFERENCE = {"duration": 60., "events": [{"timestamp": 20., "text": "#C opens the door"}]}


@pytest.mark.parametrize("field,value", [("timestamp",True),("timestamp",float("nan")),
    ("timestamp",60.),("actor",""),("action"," "),("action",123),("objects",[1]),("details","text")])
def test_strict_events(field,value):
    row=event();row[field]=value
    with pytest.raises(ValueError):validate_events(response([row]),60.)


def test_no_hidden_spec_no_extra_prose_response_limit_and_hash():
    t=task()
    with pytest.raises(ValueError):
        WitnessTask.model_validate({**t.model_dump(), "task_spec": {**t.task_spec,"narrations":["canary"]}})
    with pytest.raises(ValueError):validate_events({**response(),"overview":"unscored claim"},60)
    with pytest.raises(ValueError):validate_events(response([event(details=["x"*2097152])]),60)
    assert response_hash(response()) != response_hash(response(), {"status":"ok"})
    for field in t.required_hash_fields:
        changed=t.model_copy(deep=True)
        changed.__dict__[field] = str(getattr(t,field))+"tampered"
        assert changed.body_hash != t.body_hash


def test_scoring_omission_duplicate_and_time_controls():
    good=score_events(REFERENCE,response(),[decision()],evaluator_id="test")
    assert good["f1"]==good["recall"]==good["precision"]==1
    assert good["provisional"] is True and good["temporal_error_mean_s"]==0
    assert score_events(REFERENCE,response([]),[],evaluator_id="test")["f1"]==0
    dup=score_events(REFERENCE,response([event(),event()]),[decision(),decision(1)],evaluator_id="test")
    assert dup["f1"]==pytest.approx(2/3) and dup["matched"]==1
    late=score_events(REFERENCE,response([event(timestamp=26.)]),[],evaluator_id="test")
    assert late["f1"]==0 and late["no_temporal_reference"]==1
    boundary=score_events(REFERENCE,response([event(timestamp=25.)]),[decision()],evaluator_id="test")
    assert boundary["f1"]==1 and boundary["temporal_error_mean_s"]==5
    assert good==score_events(REFERENCE,response(),[decision()],evaluator_id="test")


def test_human_interval_support_does_not_manufacture_an_onset():
    from witness.score_v5_0_0 import Reference
    ref={"duration":60.,"events":[{"start":10.,"end":40.,"text":"The person opens the door."}]}
    for instant, error in ((10.,0.),(25.,0.),(40.,0.),(45.,5.)):
        s=score_events(ref,response([event(timestamp=instant)]),[decision()],evaluator_id="test")
        assert s["f1"]==1 and s["temporal_error_mean_s"]==error
        assert s["temporal_metric"]=="distance_to_human_interval"
    assert score_events(ref,response([event(timestamp=45.01)]),[],evaluator_id="test")["f1"]==0
    for malformed in ([{"start":20.,"end":10.,"text":"x"}],
                      [{"start":20.,"end":61.,"text":"x"}],
                      [{"start":10.,"end":20.,"text":"x"},{"timestamp":10.,"text":"x"}]):
        with pytest.raises(ValueError): Reference.model_validate({"duration":60.,"events":malformed})


@pytest.mark.parametrize("change", [{"actor":"otra persona"},{"action":"no abre"},
                                   {"details":["la puerta está ardiendo"]}])
def test_contradictions_all_fields(change):
    row={**event(),**change}
    d=decision(value=row)
    key=next(iter(change));key="details/0" if key=="details" else key
    d["fields"][key]="contradiction"
    with pytest.raises(ValueError):score_events(REFERENCE,response([row]),[d],evaluator_id="test")
    d["relation"]="contradiction"
    score=score_events(REFERENCE,response([row]),[d],evaluator_id="test")
    assert score["f1"]==0 and score["contradictions"]==1


def test_unbacked_distinct_from_contradiction_and_missing_judge_fails():
    with pytest.raises(ValueError):score_events(REFERENCE,response(),[],evaluator_id="test")
    s=score_events(REFERENCE,response(),[decision(relation="unbacked")],evaluator_id="test")
    assert s["unbacked"]==1 and s["contradictions"]==0
    s=score_events(REFERENCE,response(),[decision(relation="uncertain")],evaluator_id="test",calibrated=True)
    assert s["provisional"] is True and s["uncertain"]==1


def test_maximum_matching_beats_greedy_and_duplicates_never_add_credit():
    ref={"duration":60.,"events":[{"timestamp":20.,"text":"door"},{"timestamp":21.,"text":"handle"}]}
    pred=response([event(),event(timestamp=21.,action="sujeta")])
    ds=[decision(0,0),decision(0,1),decision(1,0,value=pred["events"][1]),
        decision(1,1,"unbacked",pred["events"][1])]
    assert score_events(ref,pred,ds,evaluator_id="test")["f1"]==1
    same=response([event(),event()])
    ds=[decision(p,r) for p in range(2) for r in range(2)]
    assert score_events(ref,same,ds,evaluator_id="test")["matched"]==1


def test_bad_and_incomplete_envelopes():
    for raw in (b'{"task_id":"a","task_id":"b"}', b'{',b'NaN', b'x'*2097153):
        with pytest.raises(ResponseRejected):parse_response(raw)
    t=task();r=t.model_copy(deep=True);r.reconstruction=response();r.trace_summary={"status":"ok"}
    validate_response(t,r)
    with pytest.raises(ResponseRejected):parse_response(canonical_bytes({**r.model_dump(),"private_source":"canary"}))
    r.trace_summary["private_model_path"]="canary"
    with pytest.raises(ResponseRejected):validate_response(t,r)


def test_activation_floor_survives_restart_without_skipping_another_epoch(tmp_path):
    s=EventsState(tmp_path,netuid=20,target_hotkey="our-miner",start_after_epoch=100)
    assert not s.eligible(100) and s.eligible(101)
    s.close()
    s=EventsState(tmp_path,netuid=20,target_hotkey="our-miner",start_after_epoch=101)
    assert s.eligible(101)
    s.close()


def test_scheduler_restart_preserves_five_dispatches_and_skips_epochs(tmp_path):
    s=EventsState(tmp_path,netuid=20,target_hotkey="our-miner")
    jobs=[{"original":str(i)} for i in range(5)]
    rid=s.begin(10,jobs)
    for task_ in s.tasks(rid)[:2]:
        assert s.claim(task_["id"])
        s.finish_task(task_["id"],{"status":"ok"})
    interrupted=s.tasks(rid)[2]
    assert s.claim(interrupted["id"])
    s.close()
    s=EventsState(tmp_path,netuid=20,target_hotkey="our-miner");s.recover()
    assert s.tasks(rid)[2]["result"]["status"]=="interrupted_unknown"
    assert not s.claim(interrupted["id"])
    for task_ in s.tasks(rid)[3:]:
        assert s.claim(task_["id"]);s.finish_task(task_["id"],{"status":"ok"})
    s.finish_round(rid,12)
    assert not s.eligible(12) and not s.eligible(11) and s.eligible(13)
    assert s.begin(12,jobs) is None
    assert s.begin(13,jobs)
    assert len(s.tasks(rid))==5
    s.close()
    with pytest.raises(ValueError):EventsState(tmp_path,netuid=20,target_hotkey="another-miner")


def test_calibration_identity_bindings_thresholds_and_intervals():
    from witness.events_evaluation import calibrate,judge_response,wilson
    expected={k:v for k,v in decision().items() if k in {"relation","fields"}}
    cases=[{"case_id":str(i),"partition":"calibration","input":{"narration":"#C opens the door",
            "event_fields":scored_text(Event.model_validate(event()))},"event":event(),
            "expected":copy.deepcopy(expected)} for i in range(300)]
    for c in cases[:100]:
        c["input"]["narration"]="#C does not open the door"
        c["expected"]={"relation":"contradiction","fields":{k:"contradiction" for k in expected["fields"]}}
    def judge(_prompt,value):
        return cases[0]["expected"] if 'not open' in value['narration'] else expected
    report=calibrate(cases,judge,evaluator_id='test')
    assert report['passed'] and report['total']==300 and report['contradictions']==100
    assert report['accuracy_wilson_95'][0]<1 and report['contradiction_acceptance_wilson_95'][1]>0
    result=judge_response(REFERENCE,response(),judge,evaluator_id='test',calibration=report)
    assert result['score']['provisional'] is False
    assert judge_response(REFERENCE,response(),judge,evaluator_id='changed',calibration=report)['score']['provisional']
    report=calibrate(cases,lambda *_:expected,evaluator_id='test')
    assert not report['passed'] and report['contradiction_acceptance']==1
    malformed=calibrate(cases,lambda *_:{"malformed":"output"},evaluator_id='test')
    assert not malformed['passed'] and malformed['accuracy']==0 and len(malformed['rows'])==300
    assert wilson(0,0) is None


def test_bounded_process_cancels_descendants_and_limits_output(tmp_path):
    marker=tmp_path/"residual"
    code="import subprocess,time;subprocess.Popen(["+repr(sys.executable)+",'-c',"+repr(
        "import time;from pathlib import Path;time.sleep(1);Path("+repr(str(marker))+").write_text('bad')")+"]);time.sleep(10)"
    async def check():
        with pytest.raises(TimeoutError):await run_process([sys.executable,"-c",code],timeout=.15)
        await asyncio.sleep(1.1)
        assert not marker.exists()
        with pytest.raises(ValueError):await run_process([sys.executable,"-c","print('x'*10000)"],timeout=2,max_output=100)
    asyncio.run(check())


def test_bounded_process_drains_full_pipe_during_forced_kill(tmp_path):
    from pathlib import Path
    # Ignoring TERM lets stdout fill while cleanup waits before sending KILL.
    # Reaping without draining this pipe used to hang indefinitely.
    pidfile=tmp_path/'writer.pid'
    code=('import os,signal,time;from pathlib import Path;'
          'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
          f'Path({str(pidfile)!r}).write_text(str(os.getpid()));'
          'os.write(1,b"x"*10_000_000);time.sleep(30)')
    async def check():
        async with asyncio.timeout(3):
            with pytest.raises(ValueError,match='worker_output_too_large'):
                await run_process([sys.executable,'-c',code],timeout=1,max_output=100)
        assert not Path('/proc',pidfile.read_text()).exists()
    asyncio.run(check())


def test_v5_signed_sdk_http_full_body_and_disconnect(tmp_path,monkeypatch):
    monkeypatch.setattr('bittensor.core.dendrite.networking.get_external_ip',lambda:'127.0.0.1')
    caller,server=Keypair.create_from_uri('//Alice'),Keypair.create_from_uri('//Bob')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    axon=bt.Axon(wallet=SimpleNamespace(hotkey=server,coldkeypub=server),ip='127.0.0.1',external_ip='127.0.0.1',port=port)
    miner=WitnessMiner(feedback_dir=tmp_path/'feedback')
    axon.attach(forward_fn=miner.forward,blacklist_fn=miner.blacklist,priority_fn=miner.priority)
    axon.attach(forward_fn=miner.feedback.forward,blacklist_fn=miner.feedback.blacklist,priority_fn=miner.feedback.priority)
    axon.start()
    async def check():
        for _ in range(200):
            try:
                _,w=await asyncio.open_connection('127.0.0.1',port);w.close();await w.wait_closed();break
            except OSError:await asyncio.sleep(.01)
        dendrite=bt.Dendrite(wallet=caller)
        chain=object.__new__(BittensorChainAdapter);chain.dendrite=dendrite
        endpoint=MinerEndpoint(7,server.ss58_address,axon.info())
        try:
            sent=task()
            r=await asyncio.wait_for(chain.query(endpoint,sent,timeout=180),timeout=3)
            assert r.reconstruction==response([]) and r.trace_summary=={"status":"ok"}
            evidence=sent._transport_evidence
            assert evidence['send_attempted'] and evidence['send_confirmed']
            assert len(evidence['wire_response_sha256'])==64 and evidence['wire_response_bytes']>0
            assert len(evidence['request_body_sha256'])==64
            assert '_transport_evidence' not in sent.model_dump()
            cancelled=asyncio.Event()
            async def wait(_):
                try:await asyncio.Event().wait()
                finally:cancelled.set()
            miner.reconstruct=wait
            with pytest.raises(TimeoutError):await chain.query(endpoint,task(),timeout=.1)
            await asyncio.wait_for(cancelled.wait(),2)
            assert miner._active==0
            feedback=EventsFeedback(round_id='v5-'+'a'*32,validator_hotkey=caller.ss58_address,
                completed_at='2026-09-12T00:00:00Z',sent=5,completed=5,rejected=0,expired=0,
                scored=0,f1=None,precision=None,recall=None,provisional=True)
            receipt=await chain.send_feedback(endpoint,WitnessFeedback(report=feedback),timeout=3)
            assert receipt['status']=='accepted'
        finally:await dendrite.aclose_session()
    try:asyncio.run(check())
    finally:axon.stop();axon.thread_pool.shutdown(wait=False)
