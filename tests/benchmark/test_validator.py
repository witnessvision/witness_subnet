"""Mainnet contract tests: chain-only consensus, one-shot fairness and real P2P."""
from __future__ import annotations
import hashlib
import json
import multiprocessing
from pathlib import Path
import threading
import time

import pytest
from bittensor_wallet import Keypair

from witness.events import content_hash
from witness.storage import write_private
from witness.benchmark.protocol import BASELINE, CONTROLS_OK, REJECTED, Result, decide, quantize
from witness.benchmark.submission import Submission, challenge_id, parse_submission, prepare_manifest, validate_manifest
from witness.benchmark.triggers import Triggers
from witness.benchmark.ledger import Ledger
from witness.benchmark.commitments import FileCommitments
from witness.benchmark.p2p import Auth, ModelClient, ModelServer, auth_payload, certificate, sign

MINERS = ['5' + c * 47 for c in 'ABCDEFG']
VALS = ['5' + c * 47 for c in 'abc']
POLICY = 'f' * 64
MANIFESTS = [str(i) * 64 for i in range(1, 8)]
MODELS = [challenge_id(hotkey, manifest) for hotkey, manifest in zip(MINERS, MANIFESTS)]


def snapshot(block=1, **changes):
    keys = MINERS + VALS
    return {'block': block, 'block_hash': f'0x{block:064x}', 'epoch_index': block // 2,
            'uids': dict(zip(keys, range(len(keys)))), 'coldkeys': {k: 'cold_' + k for k in keys},
            'validators': {VALS[0]: 9, VALS[1]: 1}, 'permits': VALS[:2], 'burn_uid': 6,
            'tempo': 2, 'weights_rate_limit': 1, 'last_update': {}, **changes}


def submission(i=0):
    return Submission(MANIFESTS[i], 'e' * 64)


def entry(i=0, block=1, owner=None):
    return {'hotkey': MINERS[i], 'block': block, 'value': submission(i).commitment,
            'model_id': MODELS[i], 'uid': i, 'coldkey': owner or f'owner{i}'}


def result(voter, model, q, r, *, kq=40000, kr=30000, flags=CONTROLS_OK, window=1, block=5, policy=POLICY):
    value = Result(window, model, MODELS[model], policy[:24], 'd' * 64, q, r, kq, kr, flags)
    return {'hotkey': VALS[voter], 'block': block, 'value': value.commitment}


def test_wire_roundtrip_and_canonical_bounds():
    record = Result(2**32-1, 65535, 'a'*64, 'b'*24, 'c'*64, 65535, 65000, 12, 1)
    assert len(record.commitment.encode()) == 126
    assert Result.parse(record.commitment) == record
    assert parse_submission(submission().commitment) == submission()
    for value in (record.commitment + '=', record.commitment[:-1], 'wk1|old', 'wr2|'+'!'*122):
        with pytest.raises(ValueError):
            Result.parse(value)
    for value in (-.1, 1.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            quantize(value)
    with pytest.raises(ValueError):
        Result(0, 0, 'a'*64, 'b'*24, 'c'*64, 1, 2, 0, 0)
    with pytest.raises(ValueError):
        parse_submission('wx1|qwen2.5-omni|owner/repo|'+'a'*40)


def test_stake_average_is_paired_and_only_one_winner():
    king = entry(0)
    rows = [result(v, 0, 40000, 30000, flags=CONTROLS_OK|BASELINE) for v in (0, 1)]
    rows += [result(0, 1, 50000, 40000), result(1, 1, 50000, 35000),
             result(0, 2, 50000, 31000), result(1, 2, 65535, 65000)]
    outcome = decide(rows, window=1, policy=POLICY, king=king,
                     candidates={MODELS[i]: entry(i) for i in (1, 2)}, snapshot=snapshot())
    assert outcome['king']['hotkey'] == MINERS[1]
    assert outcome['uids'] == [1] and outcome['weights'] == [1.]
    # Repetition, another policy and a stale window must not affect stake.
    altered = rows + [*rows, result(0, 2, 65535, 65535, policy='e'*64),
                      result(0, 2, 65535, 65535, window=0)]
    assert decide(altered, window=1, policy=POLICY, king=king,
                  candidates={MODELS[i]: entry(i) for i in (1, 2)}, snapshot=snapshot()) == outcome


def test_one_evaluator_suffices_but_missing_or_mismatched_baseline_does_not():
    rows = [result(0, 0, 40000, 30000, flags=CONTROLS_OK|BASELINE), result(0, 1, 50000, 45000)]
    kwargs = dict(window=1, policy=POLICY, king=entry(0), candidates={MODELS[1]: entry(1)}, snapshot=snapshot())
    assert decide(rows, **kwargs)['uids'] == [1]
    assert decide(rows[1:], **kwargs)['uids'] == [0]
    assert decide([rows[0], result(0, 1, 50000, 45000, kr=1)], **kwargs)['uids'] == [0]
    assert decide([rows[0], result(0, 1, 0, 0, flags=REJECTED)], **kwargs)['uids'] == [0]


def test_margin_tie_and_deregistered_winner():
    rows = [result(0, 0, 40000, 30000, flags=CONTROLS_OK|BASELINE), result(0, 1, 40000, 30001)]
    kwargs = dict(window=1, policy=POLICY, king=entry(0), candidates={MODELS[1]: entry(1)}, snapshot=snapshot())
    assert decide(rows, **kwargs)['uids'] == [0]
    rows[-1] = result(0, 1, 50000, 45000)
    kwargs['snapshot'] = snapshot(uids={MINERS[0]: 0})
    assert decide(rows, **kwargs)['uids'] == [0]  # a departed challenger cannot evict the incumbent
    kwargs['snapshot'] = snapshot(uids={})
    assert decide(rows, **kwargs)['source'] == 'burn_no_king'


def observe(queue, rows):
    return queue.observe(rows, {r['hotkey'] for r in rows},
                         coldkeys={r['hotkey']: r['coldkey'] for r in rows}, uids={r['hotkey']: r['uid'] for r in rows})


def test_round_robin_coldkeys_persists_and_fifo_within_owner(tmp_path):
    rows = [entry(i, i+1, 'A' if i < 3 else f'owner{i}') for i in range(5)]
    queue = Triggers(tmp_path/'queue.sqlite3')
    observe(queue, rows)
    assert [r['hotkey'] for r in queue.pending(5)] == [MINERS[i] for i in (0, 3, 4, 1, 2)]
    row = queue.pending()[0]
    queue.start(row['id'], 1)
    queue.finish(row['id'], 'done', result={'reward': .3})
    queue.close()
    queue = Triggers(tmp_path/'queue.sqlite3')
    assert [r['hotkey'] for r in queue.pending(4)] == [MINERS[i] for i in (3, 4, 1, 2)]
    assert queue.rows()[0]['usage'] == 'consumed'
    assert queue.pending(10, before_block=3)[0]['hotkey'] == MINERS[1]


def test_hotkey_binds_one_model_even_after_infrastructure_failure_or_owner_change(tmp_path):
    queue = Triggers(tmp_path/'queue.sqlite3')
    observe(queue, [entry()])
    first = queue.pending()[0]
    queue.start(first['id'], 1)
    queue.defer(first['id'], 'infra', retry_block=9)
    changed = {**entry(), 'value': submission(1).commitment, 'block': 7, 'coldkey': 'new_owner'}
    assert observe(queue, [changed]) == []
    assert len(queue.rows()) == 1 and queue.rows()[0]['model_id'] == MODELS[0]
    assert queue.rows()[0]['coldkey'] == 'owner0' and queue.rows()[0]['usage'] == 'reserved'
    assert queue.pending(current_block=8) == []
    assert queue.pending(current_block=9)[0]['id'] == first['id']
    queue.start(first['id'], 2)
    queue.finish(first['id'], 'rejected', 'invalid_weights')
    assert observe(queue, [changed]) == []
    with pytest.raises(ValueError):
        queue.start(first['id'])


def test_legacy_consumed_uses_are_not_reset(tmp_path):
    import sqlite3
    path = tmp_path/'queue.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE triggers(id INTEGER PRIMARY KEY,hotkey TEXT,value TEXT,block INTEGER,status TEXT,reason TEXT,observed_unix REAL,finished_unix REAL,result TEXT,UNIQUE(hotkey,value,block))')
        db.execute("INSERT INTO triggers VALUES(1,?,'wx1|legacy',1,'done',NULL,0,1,NULL)", (MINERS[0],))
    queue = Triggers(path)
    assert observe(queue, [entry()]) == []
    assert queue.rows()[0]['usage'] == 'consumed'


def fill_ledger(tmp_path, scores=True):
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    events = {1: [entry(0), entry(1)], 5: [result(0, 0, 40000, 35000, kq=0, kr=0)] if scores else [],
              6: [result(0, 1, 50000, 45000, kq=0, kr=0, block=6)] if scores else []}
    for block in range(1, 9):
        ledger.ingest(snapshot(block), events.get(block, []))
        if block < 8:
            assert ledger.get('king') is None
    return ledger


def test_chain_only_bootstrap_closes_at_epoch_boundary_and_replays(tmp_path):
    ledger = fill_ledger(tmp_path)
    assert ledger.get('king')['hotkey'] == MINERS[1]
    assert ledger.get('decision')['uids'] == [1]
    assert ledger.is_closed(1)
    assert len(ledger.usage(VALS[0])) == 2
    ledger.close()
    reopened = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    assert reopened.cursor == 8 and reopened.get('king')['hotkey'] == MINERS[1]
    with pytest.raises(ValueError, match='noncontiguous'):
        reopened.ingest(snapshot(10), [])


def test_late_result_and_new_submission_do_not_enter_closed_window(tmp_path):
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    for block in range(1, 9):
        rows = [entry(0), entry(1)] if block == 1 else []
        if block == 8:
            rows = [result(0, 1, 60000, 50000, block=8), entry(2, block=8)]
        ledger.ingest(snapshot(block), rows)
    assert ledger.get('king') is None and ledger.usage(VALS[0]) == {}
    assert MODELS[2] not in ledger.active['candidates']


def test_file_transport_retains_overwritten_scores_and_is_readonly_by_default(tmp_path):
    store = FileCommitments(tmp_path)
    with pytest.raises(PermissionError):
        store.publish(VALS[0], result(0, 0, 4, 3, kq=0, kr=0)['value'], 1)
    store.writable = True
    first, second = result(0, 0, 4, 3, kq=0, kr=0)['value'], result(0, 1, 5, 4, kq=0, kr=0)['value']
    store.publish(VALS[0], first, 1); store.publish(VALS[0], second, 2)
    assert store.read_all()[VALS[0]]['value'] == second
    assert store.at(1)[0]['value'] == first


def keys():
    return Keypair.create_from_uri('//Alice'), Keypair.create_from_uri('//Bob')


def authorized(key, alpha=100000*10**9, permit=True):
    return {'observed_unix': time.time(), 'permits': [key.ss58_address] if permit else [],
            'alpha_rao': {key.ss58_address: alpha}}


def test_btauth_golden_payload_receiver_freshness_replay_and_alpha_boundary():
    alice, bob = keys()
    body = b'{"prompt": "hello"}'
    payload = auth_payload('POST', '/generate?stream=false', body, 1752076800000000000,
                           alice.ss58_address, bob.ss58_address)
    assert payload.decode().split('\n')[4] == '341c57448e531310fbbe83f44cea2a5e838bd9e8a6b82b269f01d0dbbc23c3cc'
    nonce = time.time_ns()
    verifier = Auth(bob.ss58_address, lambda: authorized(alice), clock=lambda: nonce)
    headers = sign(alice, 'GET', '/weights', bob.ss58_address, nonce=nonce)
    assert verifier.verify(headers, 'GET', '/weights') == alice.ss58_address
    with pytest.raises(PermissionError): verifier.verify(headers, 'GET', '/weights')
    with pytest.raises(PermissionError): verifier.verify(sign(alice, 'GET', '/weights', alice.ss58_address), 'GET', '/weights')
    with pytest.raises(PermissionError): verifier.verify(headers, 'GET', '/different')
    with pytest.raises(PermissionError): verifier.verify(sign(alice, 'GET', '/weights', bob.ss58_address, nonce=nonce-11*10**9), 'GET', '/weights')
    for alpha, permit in ((100000*10**9-1, True), (100000*10**9, False)):
        denied = Auth(bob.ss58_address, lambda: authorized(alice, alpha, permit), clock=lambda: nonce)
        with pytest.raises(PermissionError): denied.verify(headers, 'GET', '/weights')
    stale = Auth(bob.ss58_address, lambda: {**authorized(alice), 'observed_unix': time.time()-121}, clock=lambda: nonce)
    with pytest.raises(PermissionError): stale.verify(headers, 'GET', '/weights')


def model_dir(path):
    path.mkdir()
    (path/'config.json').write_text('{"model_type":"qwen2_5_omni"}')
    (path/'model.safetensors').write_bytes(b'fixture-weights-only-not-a-real-model' * 200)
    return prepare_manifest(path, 'qwen2.5-omni')


def test_actual_tls_p2p_resume_hashes_and_wrong_certificate(tmp_path):
    alice, bob = keys()
    root = tmp_path/'model'
    manifest = model_dir(root)
    cert, key, pin = certificate(tmp_path/'tls')
    info = validate_manifest(manifest)
    server = ModelServer(('127.0.0.1', 0), model_root=root, manifest=manifest,
                         auth=Auth(bob.ss58_address, lambda: authorized(alice)), certfile=cert, keyfile=key)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        item = Submission(info['model_id'], pin)
        client = ModelClient(alice, bob.ss58_address, '127.0.0.1', server.server_port, item, allow_loopback=True)
        partial = tmp_path/'cache'/item.model_id/'model.safetensors.part'
        partial.parent.mkdir(parents=True); partial.write_bytes((root/'model.safetensors').read_bytes()[:123])
        target, receipt = client.download(tmp_path/'cache')
        assert receipt['model_id'] == item.model_id
        assert (target/'model.safetensors').read_bytes() == (root/'model.safetensors').read_bytes()
        bad = ModelClient(alice, bob.ss58_address, '127.0.0.1', server.server_port,
                          Submission(info['model_id'], '0'*64), allow_loopback=True)
        with pytest.raises(ValueError, match='tls_pin'): bad.manifest()
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_manifest_bounds_nested_files_and_symlinks(tmp_path):
    root = tmp_path/'model'; manifest = model_dir(root)
    (root/'nested').mkdir(); (root/'nested'/'extra.safetensors').write_bytes(b'nested')
    nested = prepare_manifest(root, 'qwen2.5-omni')
    assert len(nested['files']) == 3 and validate_manifest(nested)['bytes'] > validate_manifest(manifest)['bytes']
    corrupt = json.loads(json.dumps(nested)); corrupt['files'][-1]['size'] = 100 * 10**9
    with pytest.raises(ValueError, match='too_large'): validate_manifest(corrupt)
    corrupt['files'][-1]['name'] = '../secret.json'
    with pytest.raises(ValueError): validate_manifest(corrupt)
    (root/'link.json').symlink_to(root/'config.json')
    with pytest.raises(ValueError, match='symlink'): prepare_manifest(root, 'qwen2.5-omni')


def fake_worker(job, pipe):
    pipe.send({'ready': True, 'hardware_id': 'fake-gpu'})
    while True:
        task = pipe.recv()
        if task is None: return
        if task['task_id'] == 'hang': time.sleep(5)
        pipe.send({'task_id': task['task_id'], 'hardware_id': 'fake-gpu', 'status': 'ok', 'raw': '{}', 'elapsed_s': .001})


def test_clip_timeout_kills_worker_and_next_clip_still_runs(tmp_path):
    from witness.benchmark.pod_runtime import supervise
    output = tmp_path/'out.jsonl'
    job = {'load_timeout_s': 2, 'tasks': [{'task_id': 'hang', 'deadline_s': .05}, {'task_id': 'next', 'deadline_s': 1}]}
    started = time.monotonic()
    assert supervise(job, output, worker=fake_worker, context=multiprocessing.get_context('fork')) == 0
    assert time.monotonic()-started < 3
    rows = [json.loads(r) for r in output.read_text().splitlines()]
    assert [r['status'] for r in rows] == ['timeout', 'ok']


def test_gpu_cap_stops_busy_pod_and_failed_stop_keeps_clock(tmp_path, monkeypatch):
    from witness.benchmark.gpu import RunPodGpu
    from witness.benchmark.contract import InfrastructureError
    monkeypatch.setenv('RUNPOD_API_KEY', 'test-only')
    gpu = RunPodGpu({'ssh_key': str(tmp_path/'key'), 'daily_cap_usd': .01}, tmp_path)
    gpu.state.update(pod_id='owned', running_since=time.time()-3600, cost_per_hr=1, last_used=time.time())
    calls = []
    monkeypatch.setattr(gpu, '_rest', lambda *args: calls.append(args) or {})
    with pytest.raises(InfrastructureError, match='daily_cost_cap'): gpu.ensure()
    assert calls == [('POST', '/pods/owned/stop')]
    assert 'running_since' not in gpu.state
    gpu.state['running_since'] = time.time()-3600
    def fail(*args): raise InfrastructureError('provider_unavailable')
    monkeypatch.setattr(gpu, '_rest', fail)
    with pytest.raises(InfrastructureError): gpu.enforce_budget()
    assert 'running_since' in gpu.state


def test_reward_averages_videos_and_random_batch_defaults():
    from witness.benchmark.reward import EVAL, windows, video_scores, eval_score, clip_reward
    assert (EVAL.videos, EVAL.clips_per_video, EVAL.references) == (10, 2, 2)
    assert windows('a', 100, 'a'*64) != windows('a', 100, 'b'*64)
    assert len(windows('a', 100, 'a'*64)) == 2
    assert clip_reward(0, 1, valid=True) == 0 and clip_reward(1, 0, valid=True) == .9
    rows = [{'video': 'a', 'quality': 1., 'reward': 1.}]*9 + [{'video': 'b', 'quality': 0., 'reward': 0.}]
    assert eval_score(video_scores(rows)) == {'videos': 2, 'quality': .5, 'reward': .5}




def test_bootstrap_uses_same_evaluator_panel_and_can_beat_rejected_model():
    candidates = {MODELS[i]: entry(i) for i in (0, 1)}
    kwargs = dict(window=1, policy=POLICY, king=None, candidates=candidates, snapshot=snapshot())
    disjoint = [result(0, 0, 50000, 45000), result(1, 1, 40000, 35000)]
    assert decide(disjoint, **kwargs)['king'] is None
    common = [result(0, 0, 50000, 45000), result(0, 1, 0, 0, flags=CONTROLS_OK|REJECTED)]
    assert decide(common, **kwargs)['king']['hotkey'] == MINERS[0]


def test_consumed_hotkey_cannot_vote_again_but_chain_audit_is_preserved(tmp_path):
    ledger = fill_ledger(tmp_path)
    second = result(0, 0, 65000, 65000, window=2, block=9)
    for block in range(9, 13):
        rows = [second] if block == 9 else []
        ledger.ingest(snapshot(block), rows)
    assert second in ledger.history(9, 10)
    assert ledger.get('king')['hotkey'] == MINERS[1]
    assert ledger.usage(VALS[0])[MINERS[0]]['window'] == 1


def test_evaluator_reuses_king_and_batch_but_never_consumes_on_local_completion(tmp_path):
    from witness.benchmark.evaluator import Evaluator
    from witness.benchmark.contract import Policy, Reference, Fact, Response, Execution
    from witness.benchmark.media import PREPROCESSOR_ID
    from witness.benchmark.simulation import FakeJudge, fixture_claims, RUNTIME
    root = tmp_path/'eval'; root.mkdir()
    ledger = Ledger(root/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    for block in range(1, 9):
        rows = [entry(i) for i in range(4)] if block == 1 else []
        if block == 5: rows = [result(0, 0, 40000, 35000, kq=0, kr=0)]
        if block == 6: rows = [result(0, 1, 50000, 45000, kq=0, kr=0, block=6)]
        ledger.ingest(snapshot(block), rows)
    policy = Policy(judge_id=FakeJudge.identity, runtime_hash=RUNTIME, preprocessing_hash=PREPROCESSOR_ID,
                    reference_kind='machine', min_annotators=1, screen_size=1, confirmation_size=2, quality_floor=0.)
    draws, calls = [], []
    def batch(window):
        draws.append(window['id']); output=[]
        for i in range(10):
            media=root/f'clip{i}.mp4';media.write_bytes(f'clip-{window["id"]}-{i}'.encode())
            digest=hashlib.sha256(media.read_bytes()).hexdigest()
            ref=Reference(kind='machine',clip_sha256=digest,duration=20.,annotators=['a'*64],
                          facts=[Fact(claim=c,salience='core') for c in fixture_claims(i,20.)])
            label=root/f'ref{i}.json';label.write_text(ref.model_dump_json())
            output.append({'video':f'video{i//2}','index':i%2,'start':i*25.,'file':media.name,
                           'clip_sha256':digest,'duration':20.,'media_path':str(media),'reference_paths':[str(label)]*2})
        return output
    def runner(items, job):
        def run(models,tasks,paths):
            calls.extend(models)
            return {model:{task.id:Execution(task_id=task.id,model_id=model,checkpoint_hash=model,
                clip_sha256=task.clip_sha256,runtime_hash=RUNTIME,preprocessing_hash=PREPROCESSOR_ID,
                status='ok',elapsed_s=10.,audio_tokens=1,video_tokens=1,
                raw=Response(claims=fixture_claims(int(path.stem[-1]),20.)).model_dump_json())
                for task,path in zip(tasks,paths)} for model in models}
        run.hardware_id='fixed-test-gpu'
        return run
    evaluator=Evaluator(root=root,hotkey=VALS[0],ledger=ledger,policy=policy,judge=FakeJudge(policy),
                        batch_factory=batch,download=lambda e,s,c:{'content_id':e['model_id']},runner=runner)
    evaluator.update(snapshot(8),start_worker=False)
    evaluator.run_once(ledger.active,snapshot(8));evaluator.run_once(ledger.active,snapshot(8))
    assert draws==[2]
    assert calls.count(MODELS[1])==1 and calls.count(MODELS[2])==1 and calls.count(MODELS[3])==1
    assert len(evaluator.outbox)==3  # one baseline and two challengers
    assert all(r['usage']=='reserved' for r in evaluator.triggers.rows() if r['hotkey'] in MINERS[2:4])
    assert {r['hotkey'] for r in evaluator.triggers.rows() if r['usage']=='consumed'}==set(MINERS[:2])
    # Retry before publication cannot run a model twice.
    evaluator.run_once(ledger.active,snapshot(8));assert len(calls)==3


def test_follower_replays_history_without_writes_and_restart_reconciles_unknown(tmp_path):
    from witness.benchmark.validator import Validator
    class Chain:
        height = 8
        applied = []
        def head(self): return self.height
        def frame(self, height): return snapshot(height)
        def snapshot(self, height=None): return snapshot(height or self.height)
        def applied_weights(self, hotkey, snap): return self.applied
    store = FileCommitments(tmp_path/'transport', writable=True)
    for i in (0, 1):
        store.publish(MINERS[i], submission(i).commitment, 1)
    for i, score in ((0, 50000), (1, 40000)):
        store.publish(VALS[0], result(0, i, score, score, kr=0, kq=0, block=5)['value'], 5+i)
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    chain = Chain()
    follower = Validator(hotkey=VALS[0], root=tmp_path, chain=chain, store=store, ledger=ledger)
    intent = follower.step()
    assert intent['uids'] == [0] and intent['weights'] == [1.]
    assert follower.future is None and not (tmp_path/'weight-send.json').exists()
    follower.close()
    write_private(tmp_path/'weight-send.json', {'status': 'submitting', 'block': 8, 'uids': [0], 'weights': [1.]})
    sent = []
    follower = Validator(hotkey=VALS[0], root=tmp_path, chain=chain, store=store, ledger=ledger,
                         set_weights=lambda *args: sent.append(args))
    assert not follower._due({'uids': [1], 'weights': [1.]}, snapshot(10))
    assert not sent and follower._load('weight-send.json')['status'] == 'unknown'
    # Reconcile the ORIGINAL pending vector, even if a new winner is desired.
    chain.applied = [(0, 65535)]
    assert not follower._due({'uids': [1], 'weights': [1.]}, snapshot(11))
    assert follower._load('weight-send.json')['status'] == 'applied'
    assert follower._due({'uids': [1], 'weights': [1.]}, snapshot(12))
    follower.close()
    ledger.close()


def test_burn_all_sends_every_weight_to_burn_uid_and_keeps_king_displayed(tmp_path, monkeypatch):
    from witness.benchmark.validator import Validator
    class Chain:
        def head(self): return 8
        def frame(self, height): return snapshot(height)
        def snapshot(self, height=None): return snapshot(height or 8)
        def applied_weights(self, *args): return []
    store = FileCommitments(tmp_path/'transport', writable=True)
    for i in (0, 1):
        store.publish(MINERS[i], submission(i).commitment, 1)
    for i, score in ((0, 50000), (1, 40000)):
        store.publish(VALS[0], result(0, i, score, score, kr=0, kq=0, block=5)['value'], 5+i)
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    monkeypatch.setenv('BURN_ALL', '1')
    sent = []
    validator = Validator(hotkey=VALS[0], root=tmp_path, chain=Chain(), store=store, ledger=ledger,
                          set_weights=lambda *args: sent.append(args) or {'success': True}, burn_all=True)
    intent = validator.step()
    validator.future.result(timeout=3)
    assert intent['uids'] == [6] and intent['weights'] == [1.] and intent['source'] == 'burn_override'
    assert intent['king']['hotkey'] == MINERS[0] and sent == [([6], [1.])]
    assert validator._load('weight-send.json')['hotkey'] is None
    validator.close()
    ledger.close()


def test_explicit_weight_write_failure_is_retried_after_backoff(tmp_path):
    from witness.benchmark.validator import Validator
    class Chain:
        height = 1
        def head(self): return self.height
        def frame(self, height): return snapshot(height)
        def snapshot(self, height=None): return snapshot(height or self.height)
        def applied_weights(self, *args): return []
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    sent = []
    def write(uids, weights):
        sent.append((uids, weights))
        return {'success': len(sent) > 1}
    chain = Chain()
    follower = Validator(hotkey=VALS[0], root=tmp_path, chain=chain,
                         store=FileCommitments(tmp_path/'transport'), ledger=ledger, set_weights=write)
    follower.step()
    follower.future.result(timeout=3)
    follower.step()
    assert len(sent) == 1 and follower._load('weight-send.json')['status'] == 'rejected'
    assert not (tmp_path/'last-weights.json').exists()
    chain.height = 6
    follower.step()
    follower.future.result(timeout=3)
    follower._complete_write()
    assert len(sent) == 2 and follower._load('last-weights.json')['receipt']['success']
    follower.close()
    ledger.close()


def test_fresh_window_sampling_is_private_and_restart_stable(tmp_path, monkeypatch):
    from witness.benchmark import pool
    draws = []
    def build(target, *, selected, salt, **kwargs):
        draws.append((selected, salt))
    monkeypatch.setattr(pool, 'build_pool', build)
    monkeypatch.setattr(pool, 'load_pool', lambda *a, **kwargs: [dict(video=v['identifier'], index=i)
                        for v in reversed(draws[-1][0]) for i in (1, 0)])
    for window in (1, 1, 2):
        rows = pool.window_batch(tmp_path, window, VALS[0], gpu=None, api=None, policy=None)
        assert [r['video'] for r in rows[::2]] == [v['identifier'] for v in draws[-1][0]]
        assert [r['index'] for r in rows] == [0, 1] * 5
    pool.window_batch(tmp_path/'another', 1, VALS[1], gpu=None, api=None, policy=None)
    assert draws[0] == draws[1]
    assert draws[0][0] != draws[2][0] and draws[0][1] != draws[2][1]
    assert draws[0] != draws[3]
    public = json.loads((tmp_path/'windows/1/draw.json').read_text())
    assert draws[0][1] not in json.dumps(public) and 'secret' not in public


def test_watchdog_never_starts_or_deletes_and_latches_before_stop(tmp_path):
    from witness.benchmark.watchdog import inspect
    calls = []
    class Gpu:
        state = {'owner': 'witness-mainnet-2', 'pod_id': 'owned', 'running_since': 1}
        config = {'daily_cap_usd': 2}
        def spend_today(self): return 3
        def _rest(self, method, path):
            calls.append((method, path))
            assert (tmp_path/'gpu-watchdog-stop.json').exists()
            return {'desiredStatus': 'RUNNING'}
    gpu = Gpu()
    write_private(tmp_path/'heartbeat.json', {'unix': 100})
    assert inspect(gpu, tmp_path, now=101)['action'] == 'would_stop' and not calls
    assert inspect(gpu, tmp_path, now=101, stop=True)['reason'] == 'daily_cost_cap'
    assert calls == [('GET', '/pods/owned'), ('POST', '/pods/owned/stop')]
    calls.clear()
    gpu.state = {'pod_id': 'someone_elses_pod', 'running_since': 1}
    assert inspect(gpu, tmp_path, now=1000, stop=True)['reason'] == 'no_owned_pod'
    assert not calls


def test_untrusted_manifest_types_fail_as_validation_errors():
    valid = {'schema_version': 'witness-model-2', 'arch': 'qwen2.5-omni',
             'files': [{'name': 'config.json', 'size': 1, 'sha256': 'a'*64},
                       {'name': 'model.safetensors', 'size': 1, 'sha256': 'b'*64}]}
    for value in (None, [], {**valid, 'arch': []}, {**valid, 'files': [None]},
                  {**valid, 'files': [{'name': [], 'size': 1, 'sha256': 12}]}):
        with pytest.raises(ValueError):
            validate_manifest(value)


def test_status_seals_open_windows_and_exports_scores_only(tmp_path):
    from witness.benchmark.status import Evidence
    from witness.benchmark.telemetry import public_status
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    ledger.ingest(snapshot(1), [])
    scores = {'quality': 0.5, 'reward': 0.45, 'latency_s': 3.0, 'time_score': 0.95}
    clip = {'id': 'c' * 64, 'start': 1.0, 'duration': 8.0, 'video_url': '/api/media/x.mp4',
            'response': {'claims': ['MODEL_OUTPUT']}, **scores, 'king': {'response': 'KING_OUTPUT', **scores}}
    body = {'available': True, 'validator': VALS[0], 'window_id': 0, 'model_id': MODELS[0], 'private_note': 'SECRET',
            'videos': [{'id': 'source', 'quality': 0.5, 'reward': 0.45, 'clips': [clip]}]}
    report_hash = content_hash(body)
    write_private(tmp_path/'reports'/f'{report_hash}.json', body)
    state = {'schema_version': 'witness-evaluator-status-2', 'validator': VALS[0], 'mode': 'evaluator',
             'block': 1, 'triggers': [], 'evaluations': [{'window_id': 0, 'model_id': MODELS[0],
                                                        'available': True, 'report_hash': report_hash}]}
    write_private(tmp_path/'queue.json', state)
    view = Evidence(tmp_path)
    assert view.evaluation(VALS[0], 0, MODELS[0])['available'] is False
    assert view.closed_reports() == []
    for block in range(2, 5): ledger.ingest(snapshot(block), [])
    report = view.evaluation(VALS[0], 0, MODELS[0])
    assert report['videos'][0]['clips'][0] == {'id': 'c' * 64, 'start': 1.0, 'duration': 8.0, **scores, 'king': scores}
    assert report['report_hash'] == report_hash
    telemetry = json.dumps(public_status(state, view.closed_reports()))
    assert 'OUTPUT' not in telemetry and 'SECRET' not in telemetry and 'video_url' not in telemetry
    assert report_hash in telemetry
    ledger.close()


def test_retrying_head_cannot_be_overtaken_by_same_coldkey(tmp_path):
    queue = Triggers(tmp_path/'queue.sqlite3')
    observe(queue, [entry(0, 1, 'A'), entry(1, 2, 'A'), entry(2, 3, 'B')])
    first = queue.pending()[0]
    queue.start(first['id'], 1)
    queue.defer(first['id'], 'provider_unavailable', 50)
    assert [r['hotkey'] for r in queue.pending(10, current_block=49)] == [MINERS[2]]
    assert [r['hotkey'] for r in queue.pending(10, current_block=50)] == [MINERS[2], MINERS[0], MINERS[1]]


def test_copied_commitment_cannot_reserve_another_miners_identity(tmp_path):
    from witness.benchmark.evaluator import Evaluator
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    first = entry(2, 1)
    duplicate = {**entry(0, 2), 'value': first['value'], 'model_id': first['model_id']}
    for block in range(1, 5):
        ledger.ingest(snapshot(block), [first] if block==1 else [duplicate] if block==2 else [])
    assert ledger.submissions()[0]['hotkey'] == MINERS[2]
    evaluator = Evaluator(root=tmp_path, hotkey=VALS[0], ledger=ledger, policy=None, judge=None,
                          batch_factory=None, download=None, runner=None)
    evaluator.update(snapshot(4), start_worker=False)
    rows = {r['hotkey']: r for r in evaluator.triggers.rows()}
    assert rows[MINERS[0]]['model_id'] != rows[MINERS[2]]['model_id']
    assert rows[MINERS[0]]['usage'] == 'reserved'
    assert rows[MINERS[2]]['usage'] == 'reserved'
    assert len(ledger.active['candidates']) == 2
    assert not evaluator.outbox


def test_no_architecture_is_enabled_implicitly(tmp_path):
    from witness.benchmark.evaluator import Evaluator
    for config in ({}, {'enabled_architectures': []}, {'enabled_architectures': ['unknown']}):
        with pytest.raises(ValueError, match='qualified_enabled_architectures'):
            Evaluator.from_config(config, tmp_path, VALS[0], None, None)


def test_judge_value_error_defers_without_zero_score_or_consumption(tmp_path, monkeypatch):
    from witness.benchmark.evaluator import Evaluator
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    for block in range(1, 5):
        ledger.ingest(snapshot(block), [entry(0), entry(1)] if block==1 else [])
    evaluator = Evaluator(root=tmp_path, hotkey=VALS[0], ledger=ledger, policy=None, judge=None,
                          batch_factory=None, download=None, runner=None)
    evaluator.update(snapshot(4), start_worker=False)
    monkeypatch.setattr(evaluator, '_batch', lambda window: ([], []))
    def fail(*args, **kwargs): raise ValueError('malformed_judge_response')
    monkeypatch.setattr(evaluator, '_evaluate', fail)
    with pytest.raises(ValueError): evaluator.run_once(ledger.active, snapshot(4))
    assert not evaluator.outbox
    assert all(r['usage']=='reserved' for r in evaluator.triggers.rows())
    assert evaluator.triggers.rows()[0]['retry_block'] > 4


def test_rejected_score_counts_as_zero_in_stake_denominator():
    rows = [result(v, 0, 40000, 30000, flags=CONTROLS_OK|BASELINE) for v in (0, 1)]
    rows += [result(0, 1, 0, 0, flags=CONTROLS_OK|REJECTED), result(1, 1, 65535, 65535)]
    outcome = decide(rows, window=1, policy=POLICY, king=entry(0),
                     candidates={MODELS[1]: entry(1)}, snapshot=snapshot())
    assert outcome['uids'] == [0]
    assert outcome['aggregates'][MODELS[1]]['quality'] == pytest.approx(.1)
    assert outcome['aggregates'][MODELS[1]]['stake'] == 10
    with pytest.raises(ValueError, match='score_zero'):
        result(0, 1, 50000, 45000, flags=CONTROLS_OK|REJECTED)


def test_partial_bootstrap_does_not_consume_and_offline_pair_rotates(tmp_path):
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    for block in range(1, 9):
        rows = [entry(i) for i in range(4)] if block==1 else []
        if block==5: rows = [result(0, 0, 50000, 45000, kq=0, kr=0)]
        ledger.ingest(snapshot(block), rows)
    assert ledger.usage(VALS[0]) == {} and ledger.get('king') is None
    assert set(ledger.active['candidates']) == set(MODELS[2:4])
    for block in range(9, 13):
        rows = []
        if block in (9, 10):
            i=block-7
            rows=[result(0, i, 50000, 45000-i*1000, kq=0, kr=0, window=2, block=block)]
        ledger.ingest(snapshot(block), rows)
    assert ledger.get('king')['hotkey'] == MINERS[2]
    assert set(ledger.usage(VALS[0])) == set(MINERS[2:4])
    # The first honest partial result remains an eligible reservation, not a burned attempt.
    assert MODELS[0] in ledger.active['candidates']


def test_partial_bootstrap_can_retry_same_hotkey_on_fresh_window(tmp_path):
    ledger = Ledger(tmp_path/'chain.sqlite3', activation_block=1, activation_epoch=0, policy=POLICY)
    for block in range(1, 13):
        rows = [entry(0),entry(1)] if block==1 else []
        if block==5: rows=[result(0,0,40000,35000,kq=0,kr=0)]
        if block==9: rows=[result(0,0,50000,45000,kq=0,kr=0,window=2,block=9)]
        if block==10: rows=[result(0,1,40000,35000,kq=0,kr=0,window=2,block=10)]
        ledger.ingest(snapshot(block), rows)
    assert ledger.get('king')['hotkey']==MINERS[0]
    assert {value['window'] for value in ledger.usage(VALS[0]).values()}=={2}


def test_burn_override_sends_only_burn_and_retains_shadow_king(tmp_path):
    from witness.benchmark.validator import Validator
    ledger = fill_ledger(tmp_path)
    class Chain:
        def head(self): return 8
        def snapshot(self, height=None): return snapshot(8)
        def applied_weights(self, *args): return [(6, 65535)]
    sent = []
    validator = Validator(hotkey=VALS[0], root=tmp_path, chain=Chain(),
        store=FileCommitments(tmp_path/'transport'), ledger=ledger, burn_all=True,
        set_weights=lambda u,w: sent.append((u,w)) or {'success':True})
    intent = validator.step()
    validator.future.result(timeout=3)
    assert sent == [([6],[1.])]
    assert intent['burn_all'] and intent['source'] == 'burn_override'
    assert intent['without_burn_override']['uids'] == [1]
    assert intent['king']['hotkey'] == MINERS[1]
    assert intent['weights_applied']['uids'] == [6]
    public = json.loads((tmp_path/'queue.json').read_text())
    assert public['weights']['without_burn_override']['uids'] == [1]
    validator.close(); ledger.close()


def test_reveal_observation_does_not_restart_chain_weight_cooldown(tmp_path):
    from witness.benchmark.validator import Validator
    ledger=fill_ledger(tmp_path)
    class Chain:
        def applied_weights(self,*args):return [(1,65535)]
    v=Validator(hotkey=VALS[0],root=tmp_path,chain=Chain(),store=FileCommitments(tmp_path/'transport'),ledger=ledger)
    write_private(tmp_path/'weight-send.json',{'status':'submitted','block':5,'uids':[1],'weights':[1.]})
    target={'uids':[6],'weights':[1.]}
    assert not v._due(target,snapshot(100,weights_rate_limit=10,last_update={VALS[0]:6}))
    saved=v._load('last-weights.json')
    assert saved['block']==5 and saved['applied_block']==100
    assert v._due(target,snapshot(101,weights_rate_limit=10,last_update={VALS[0]:6}))
    assert not v._due(target,snapshot(101,weights_rate_limit=10,last_update={VALS[0]:100}))
    v.close();ledger.close()
