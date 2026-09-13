import copy

from witness.events import content_hash
from witness.events_evaluation import PROMPT_HASH
from witness.subnet.production_state import ProductionState
from witness.subnet.production_weights import policy, promotion_gate, read_latest


def evidence():
    calibration={'total':300,'passed':True,'accuracy':.98,'contradiction_acceptance':0.,
                 'prompt_hash':PROMPT_HASH,'evaluator_id':'luna'}
    identity={'evaluator':{'judge_model':'gpt-5.6-luna','judge_effort':'low','evaluator_id':'luna'},
              'calibration_hash':content_hash(calibration)}
    miner={'uid':117,'hotkey':'candidate','planned':5,'sent':5,'valid':5,'scored':5,
           'max_elapsed_s':175.,'mean_f1':.99,'mean_score':.97,'ema':.5,'eligible':True}
    reports=[{'round_id':str(i),'epoch':i,'finish_epoch':i,'complete':True,'identity':identity,
              'miners':[dict(miner)],'winner':dict(miner)} for i in range(3)]
    return calibration,reports


def test_promotion_requires_calibration_fifteen_valid_and_complete_comparisons():
    calibration,reports=evidence()
    assert promotion_gate(reports,calibration,'candidate')['passed']
    for field,value in [('accuracy',.94),('contradiction_acceptance',.03),('evaluator_id','sol')]:
        changed={**calibration,field:value}
        assert not promotion_gate(reports,changed,'candidate')['passed']
    for field,value in [('valid',4),('scored',4),('sent',4),('mean_f1',.5),('mean_score',.8),('max_elapsed_s',180.)]:
        changed=copy.deepcopy(reports);changed[1]['miners'][0][field]=value
        assert not promotion_gate(changed,calibration,'candidate')['passed']
    changed=copy.deepcopy(reports);changed[1]['complete']=False
    assert not promotion_gate(changed,calibration,'candidate')['passed']


def test_policy_returns_to_burn_when_incomplete_stale_or_uid_reused():
    _,reports=evidence();report=reports[-1]
    args={'burn_uid':240,'epoch':2,'expected_identity':report['identity'],'registered_hotkey':lambda uid:'candidate'}
    assert policy(report,**args)==([240,117],[.7,.3])
    for changed in (None,{**report,'complete':False},{**report,'winner':None}):
        assert policy(changed,**args)==([240],[1.])
    assert policy(report,**{**args,'epoch':4})==([240],[1.])
    assert policy(report,**{**args,'registered_hotkey':lambda uid:'new-hotkey'})==([240],[1.])


def test_writer_reads_authoritative_journal_during_an_unfinished_round(tmp_path):
    path=tmp_path/'scheduler.sqlite3';state=ProductionState(path,{'evaluator':'fixture'})
    state.begin(3,[{'original':str(i)} for i in range(5)],[{'uid':1,'hotkey':'h','url':'http://example'}])
    assert read_latest(path) is None
    state.close()
