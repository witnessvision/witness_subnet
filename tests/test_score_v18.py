"""The frozen ASR benchmark cannot require an unscored speaker identity."""
from copy import deepcopy
import pytest
from witness.score_v17 import score_reconstruction as previous
from witness.score_v18 import score_reconstruction


@pytest.mark.parametrize('speaker', [None, '', 'unknown', 'alice'])
def test_speaker_identity_neither_blocks_nor_increases_lexical_credit(speaker):
    scene = {'difficulty': 1, 'fps': 24, 'duration_frames': 240,
             'dialogue': [{'start': 1, 'end': 3, 'speaker': 'private-actor', 'text': 'open the window'}]}
    row = {'start': 1, 'end': 3, 'text': 'open the window', 'speaker': speaker}
    result = score_reconstruction(scene, {'dialogue': [row]})
    assert result['family_scores']['dialogue'] == 1
    assert result['benchmark_version'] == '1.8'
    row.pop('speaker')
    assert score_reconstruction(scene, {'dialogue': [row]})['family_scores']['dialogue'] == 1
    assert previous(scene, {'dialogue': [row]})['family_scores']['dialogue'] == 0
    row['text'] = 'close the door'
    assert score_reconstruction(scene, {'dialogue': [row]})['family_scores']['dialogue'] < 1


def test_unknown_speaker_duplicates_and_wrong_intervals_still_lose_credit():
    scene = {'difficulty': 1, 'fps': 24, 'duration_frames': 240,
             'dialogue': [{'start': 1, 'end': 3, 'speaker': 'a', 'text': 'red'}]}
    prediction = {'start': 1, 'end': 3, 'text': 'red'}
    assert score_reconstruction(scene, {'dialogue': [prediction] * 100})['family_scores']['dialogue'] < .02
    wrong = deepcopy(prediction);wrong.update(start=5,end=7)
    assert score_reconstruction(scene, {'dialogue': [wrong]})['family_scores']['dialogue'] == 0
