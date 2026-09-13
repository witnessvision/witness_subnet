import pytest
from witness.score_v5_1_0 import latency_reward, score_events


def test_time_has_thirty_percent_quality_gated_weight():
    assert latency_reward(1.,0.)['reward']==pytest.approx(1.)
    assert latency_reward(1.,90.)['reward']==pytest.approx(.85)
    assert latency_reward(.5,90.)['reward']==pytest.approx(.425)
    assert latency_reward(0.,0.)['reward']==0.
    assert latency_reward(1.,179.999)['reward']> .7
    assert latency_reward(1.,180.)['reward']==0.
    assert latency_reward(1.,181.)['reward']==0.
    assert latency_reward(1.,1.,response_received=False)['reward']==0.


@pytest.mark.parametrize('f1,elapsed', [(float('nan'),1.),(1.,float('inf')),
    (True,1.),(1.,False),(1.01,1.),(-.1,1.),(1.,-.1)])
def test_reject_invalid_measured_inputs(f1,elapsed):
    with pytest.raises(ValueError):latency_reward(f1,elapsed)


def test_wrapper_preserves_annotation_diagnostics_and_empty_response_zero():
    reference={'duration':60.,'events':[{'timestamp':30.,'text':'A person opens a door.'}]}
    response={'schema_version':'5.0','events':[]}
    score=score_events(reference,response,[],elapsed_s=.001,evaluator_id='frozen-judge')
    assert score['f1']==score['reward']==0.
    assert score['references']==1 and score['scorer_version']=='5.1.0'
    assert score['quality_scorer_version']=='5.0.0' and score['weights_enabled'] is False
