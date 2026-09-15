import copy
import asyncio
import json
from types import SimpleNamespace

import pytest

from witness.events import content_hash
from witness.events_evaluation import PROMPT_HASH
from witness.subnet.production_state import ProductionState
from witness.subnet.production_weights import (activation_gate, policy, promotion_gate, read_latest,
    round_decision, pending_is_known, run)
from witness.subnet.chain import WeightSubmission


def evidence():
    calibration={'total':300,'passed':True,'accuracy':.98,'contradiction_acceptance':0.,
                 'prompt_hash':PROMPT_HASH,'evaluator_id':'luna'}
    identity={'evaluator':{'judge_model':'gpt-5.6-luna','judge_effort':'low','evaluator_id':'luna',
                           'prompt_hash':PROMPT_HASH},
              'calibration_hash':content_hash(calibration)}
    miner={'uid':117,'hotkey':'candidate','planned':5,'sent':5,'valid':5,'scored':5,
           'max_elapsed_s':175.,'mean_f1':.99,'mean_score':.97,'ema':.5,'eligible':True}
    reports=[{'round_id':str(i),'epoch':i,'finish_epoch':i,'complete':True,'identity':identity,
              'miners':[dict(miner)],'winner':dict(miner)} for i in range(3)]
    return calibration,reports


def test_promotion_requires_calibration_fifteen_valid_and_complete_comparisons():
    calibration,reports=evidence()
    assert promotion_gate(reports,calibration,'candidate')['passed']
    for field,value in [('accuracy',.94),('contradiction_acceptance',.03),('evaluator_id','sol'),
                        ('prompt_hash','different-prompt')]:
        changed={**calibration,field:value}
        assert not promotion_gate(reports,changed,'candidate')['passed']
    for field,value in [('valid',4),('scored',4),('sent',4),('mean_f1',.5),('mean_score',.8),('max_elapsed_s',180.)]:
        changed=copy.deepcopy(reports);changed[1]['miners'][0][field]=value
        assert not promotion_gate(changed,calibration,'candidate')['passed']
    changed=copy.deepcopy(reports);changed[1]['complete']=False
    assert not promotion_gate(changed,calibration,'candidate')['passed']


def test_field_judge_promotion_requires_matching_new_calibration_and_prompt():
    from witness.events_evaluation import FIELD_PROMPT_HASH
    calibration, reports = evidence()
    calibration['prompt_hash'] = FIELD_PROMPT_HASH
    calibration['evaluator_id'] = 'luna:fields-only-v2'
    for report in reports:
        report['identity']['evaluator'].update(prompt_hash=FIELD_PROMPT_HASH,
                                               evaluator_id=calibration['evaluator_id'])
        report['identity']['calibration_hash'] = content_hash(calibration)
    assert promotion_gate(reports, calibration, 'candidate')['passed']
    reports[0]['identity']['evaluator']['prompt_hash'] = PROMPT_HASH
    assert not promotion_gate(reports, calibration, 'candidate')['passed']


def test_pause_always_burns_even_with_a_qualified_winner():
    _,reports=evidence();report=reports[-1]
    args={'burn_uid':240,'epoch':2,'expected_identity':report['identity'],'registered_hotkey':lambda uid:'candidate'}
    assert policy(report,**args)==([240],[1.])
    for changed in (None,{**report,'complete':False},{**report,'winner':None}):
        assert policy(changed,**args)==([240],[1.])
    assert policy(report,**{**args,'epoch':4})==([240],[1.])
    assert policy(report,**{**args,'registered_hotkey':lambda uid:'new-hotkey'})==([240],[1.])


def test_writer_reads_authoritative_journal_during_an_unfinished_round(tmp_path):
    path=tmp_path/'scheduler.sqlite3';state=ProductionState(path,{'evaluator':'fixture'})
    state.begin(3,[{'original':str(i)} for i in range(5)],[{'uid':1,'hotkey':'h','url':'http://example'}])
    assert read_latest(path) is None
    state.close()


def test_failed_preparation_invalidates_the_previous_complete_comparison(tmp_path):
    path=tmp_path/'scheduler.sqlite3';state=ProductionState(path,{'evaluator':'fixture'})
    active=state.begin(3,[{'original':str(i)} for i in range(5)],[{'uid':1,'hotkey':'h','url':'http://example'}])
    for task in state.tasks(active['id']):
        assert state.claim(task['id'])
        state.record(task['id'],{'dispatch_attempted':True,'response_valid':True,
            'evaluation_status':'complete','f1':1.,'reward':.99,'miner_elapsed_s':6.})
    report=state.finish(active['id'],3)
    assert read_latest(path)==report
    state.skip(4)  # No new round row exists when source preparation fails.
    assert state.latest()==report
    assert read_latest(path) is None
    state.close()


def test_operator_quality_waiver_keeps_original_failure_and_other_gates():
    calibration, reports = evidence()
    reports[0]['miners'][0]['mean_f1'] = .5
    activation = {'calibration': calibration, 'reports': reports, 'own_hotkey': 'candidate',
                  'operational_probes_passed': True}
    assert not activation_gate(activation)['passed']
    activation['operator_authorization'] = {'policy': '70_burn_30_winner',
        'waive_initial_own_quality': True, 'authorized_at': '2026-01-01T00:00:00Z',
        'reason': 'Operator explicitly requested production weights after reviewing acceptance.'}
    gate = activation_gate(activation)
    assert gate['passed'] and not gate['original_gate']['passed']
    assert gate['waived_failures'] == ['own_miner_quality_failure']
    for field, value in [('valid', 4), ('scored', 4), ('sent', 4)]:
        changed = copy.deepcopy(activation)
        changed['reports'][0]['miners'][0][field] = value
        assert not activation_gate(changed)['passed']
    changed = copy.deepcopy(activation); changed['calibration']['accuracy'] = .9
    assert not activation_gate(changed)['passed']
    changed = copy.deepcopy(activation); changed['operational_probes_passed'] = False
    assert not activation_gate(changed)['passed']


def test_burn_decision_is_stable_per_epoch_and_never_waits_for_evaluation():
    _, reports = evidence(); report = reports[-1]
    progress = {'round_id': '2', 'epoch': 2, 'cursor': 2, 'status': 'complete', 'report': report}
    args = {'burn_uid': 240, 'epoch': 2, 'expected_identity': report['identity'],
            'registered_hotkey': lambda uid: 'candidate'}
    decision = round_decision(progress, **args)
    assert decision['weights'] == [1.]
    assert decision == round_decision(progress, **args)
    assert decision['decision_id'] != round_decision(progress, **{**args, 'epoch': 3})['decision_id']
    assert round_decision({**progress, 'status': 'running', 'report': None}, **args) == decision
    assert round_decision({**progress, 'status': 'running', 'report': None},
                         **{**args, 'epoch': 4})['weights'] == [1.]
    for changed in ({**progress, 'status': 'incomplete'}, {**progress, 'cursor': 3}):
        fallback = round_decision(changed, **args)
        assert fallback == decision
    assert round_decision(progress, **{**args, 'registered_hotkey': lambda uid: 'replacement'})['weights'] == [1.]


def test_unknown_pending_commit_blocks_new_round():
    receipt = {'submission': WeightSubmission('finalized', block_hash='block-a').as_dict()}
    assert pending_is_known([{'commit_block': 10}], [receipt], lambda h: 10)
    assert not pending_is_known([{'commit_block': 11}], [receipt], lambda h: 10)
    assert not pending_is_known([{'commit_block': 10}], [], lambda h: 10)


def test_writer_handover_round_journal_restart_and_ambiguous_stop(tmp_path, monkeypatch):
    import witness.subnet.production_weights as module
    calibration, reports = evidence()
    activation = {'calibration': calibration, 'reports': reports, 'own_hotkey': 'candidate',
                  'operational_probes_passed': True}
    activation_path = tmp_path/'activation.json'; activation_path.write_text(json.dumps(activation))
    config = {'root': str(tmp_path), 'activation': str(activation_path), 'scheduler': 'unused'}
    progress = {'round_id': '2', 'epoch': 2, 'cursor': 2, 'status': 'complete', 'report': reports[-1]}
    pending = [{'commit_block': 8}]
    submitted = []
    substrate = SimpleNamespace(get_block_number=lambda h: 10,
        query=lambda module, name, *a, **k: SimpleNamespace(value='candidate' if name == 'Keys' else []))
    def send(uids, weights):
        submitted.append((uids, weights))
        return WeightSubmission('finalized', block_hash='block-a')
    chain = SimpleNamespace(netuid=20, subtensor=SimpleNamespace(substrate=substrate),
        epoch_state=lambda: {'epoch_index': progress['epoch']}, set_weights=send)
    monkeypatch.setattr(module, 'burn_state', lambda *a: {'block': 10, 'block_hash': 'block-a',
        'burn_uid': 240, 'validator_uid': 240, 'blocks_until_submission': 0})
    monkeypatch.setattr(module, 'pending_commits', lambda *a: (pending, None))
    monkeypatch.setattr(module, 'read_progress', lambda *a: progress)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='inactive'))
    class StopLoop(Exception): pass
    async def stop(*a): raise StopLoop
    monkeypatch.setattr(module.asyncio, 'sleep', stop)
    def tick():
        with pytest.raises(StopLoop): asyncio.run(run(chain, config))
    tick()
    assert not submitted and not (tmp_path/'handover.json').exists()
    pending.clear(); tick()
    assert submitted == [([240], [1.])]
    tick()  # A service restart does not repeat a finalized decision.
    assert len(submitted) == 1
    pending.append({'commit_block': 10})
    progress.update(round_id='3', epoch=3, cursor=3, status='incomplete', report=None)
    tick()  # Even a known pending commitment blocks a different policy.
    assert len(submitted) == 1
    pending.clear(); tick()
    assert submitted[-1] == ([240], [1.]) and len(submitted) == 2
    pending.append({'commit_block': 10})
    next_report = {**reports[-1], 'round_id': '4', 'epoch': 4, 'finish_epoch': 4}
    progress.update(round_id='4', epoch=4, cursor=4, status='complete', report=next_report)
    tick()  # Drain the earlier epoch's commitment before refreshing burn.
    assert len(submitted) == 2
    pending.clear(); tick()
    assert submitted[-1] == ([240], [1.]) and len(submitted) == 3
    tick()
    assert len(submitted) == 3
    record = next((tmp_path/'submissions').glob('*.json'))
    value = json.loads(record.read_text()); value['submission']['status'] = 'unknown'
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='ambiguous_submission'):
        asyncio.run(run(chain, config))
    assert len(submitted) == 3


def test_burn_needs_no_activation_catalog_scheduler_or_provider(tmp_path, monkeypatch):
    import witness.subnet.production_weights as module
    old_gate = tmp_path/'activation-gate.json'
    old_gate.write_text('{"historical": true}')
    # An old malformed activation path must not prevent emergency burn.
    config = {'root': str(tmp_path), 'activation': '/missing/activation.json',
              'scheduler': '/missing/scheduler.sqlite3'}
    submitted = []
    def send(uids, weights):
        submitted.append((uids, weights))
        return WeightSubmission('finalized', block_hash='block-a')
    chain = SimpleNamespace(netuid=20, set_weights=send, epoch_state=lambda: {'epoch_index': 10},
        subtensor=SimpleNamespace(substrate=SimpleNamespace(get_block_number=lambda h: 100,
            query=lambda *a, **k: SimpleNamespace(value=[[240, 65535], [7, 28086]]))))
    state = {'block': 100, 'block_hash': 'block-a', 'burn_uid': 240, 'validator_uid': 3,
             'blocks_until_submission': 1}
    monkeypatch.setattr(module, 'burn_state', lambda *a: state)
    monkeypatch.setattr(module, 'pending_commits', lambda *a: ([], None))
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='inactive'))
    class StopLoop(Exception): pass
    async def stop(*a): raise StopLoop
    monkeypatch.setattr(module.asyncio, 'sleep', stop)
    with pytest.raises(StopLoop): asyncio.run(run(chain, config))
    assert not submitted
    state['blocks_until_submission'] = 0
    with pytest.raises(StopLoop): asyncio.run(run(chain, config))
    assert submitted == [([240], [1.])]
    assert json.loads(old_gate.read_text()) == {'historical': True}
    assert json.loads((tmp_path/'observation.json').read_text())['desired'] == {'uids': [240], 'weights': [1.]}
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='active'))
    with pytest.raises(ValueError, match='previous_weight_writer_not_stopped'):
        asyncio.run(run(chain, config))
    assert len(submitted) == 1
