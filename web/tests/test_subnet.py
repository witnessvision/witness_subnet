"""Dashboard consumes display projections only; absent live state remains unknown."""
from fastapi.testclient import TestClient
from witness.storage import write_private
from witness_web.app import create_app

HOTKEY = '5' + 'A'*47


def test_no_sources_does_not_invent_a_king_or_zero_scores(tmp_path):
    client = TestClient(create_app(sources=[]))
    result = client.get('/api/subnet')
    assert result.status_code == 200
    assert result.json()['king'] is None and result.json()['block'] is None
    assert result.json()['evaluations'] == []
    assert client.get('/api/hotkeys', params={'validator': HOTKEY}).status_code == 503


def test_hotkeys_are_scoped_searchable_and_paginated_without_private_fields(tmp_path):
    source = tmp_path/'validator'
    rows = [{'hotkey': f'public-{i}', 'coldkey': 'owner-A' if i%2 else 'owner-B',
             'model_id': 'b'*64, 'usage': 'consumed', 'status': 'done',
             'private_reference': 'DO_NOT_EXPOSE'} for i in range(137)]
    write_private(source/'queue.json', {'schema_version': 'witness-evaluator-status-2',
                   'validator': HOTKEY, 'mode': 'evaluator', 'block': 10, 'triggers': rows})
    client = TestClient(create_app(sources=[str(source)]))
    state = client.get('/api/subnet').json()
    assert state['validators'][0]['used_count'] == 137
    result = client.get('/api/hotkeys', params={'validator': HOTKEY, 'page': 3})
    assert result.json()['total'] == 137 and len(result.json()['rows']) == 37
    assert 'DO_NOT_EXPOSE' not in result.text
    found = client.get('/api/hotkeys', params={'validator': HOTKEY, 'q': 'owner-A'}).json()
    assert found['total'] == 68
    assert client.get('/api/hotkeys', params={'validator': HOTKEY, 'page': 0}).status_code == 400


def test_submission_uids_are_projected_without_guessing_or_private_fields(tmp_path):
    source = tmp_path / 'validator'
    rows = [{'uid': 7, 'hotkey': 'miner', 'coldkey': 'owner', 'model_id': 'a' * 64,
             'block': 1, 'usage': 'reserved', 'status': 'queued', 'reason': 'ValueError',
             'private_reference': 'DO_NOT_EXPOSE'},
            {'uid': 8, 'hotkey': 'other', 'model_id': 'b' * 64, 'block': 2,
             'usage': 'consumed', 'status': 'done'}]
    write_private(source / 'queue.json', {
        'schema_version': 'witness-evaluator-status-2', 'validator': HOTKEY,
        'mode': 'evaluator', 'block': 10, 'triggers': rows,
        'evaluations': [{'hotkey': 'other', 'model_id': 'b' * 64},
                        {'hotkey': 'other', 'model_id': 'c' * 64}]})
    client = TestClient(create_app(sources=[str(source)]))
    response = client.get('/api/subnet')
    state = response.json()
    assert state['queues'][HOTKEY][0]['uid'] == 7
    assert [r['uid'] for r in state['evaluations']] == [8, None]
    assert 'DO_NOT_EXPOSE' not in response.text
    keys = client.get('/api/hotkeys', params={'validator': HOTKEY}).json()
    assert keys['rows'][0]['uid'] == 8
