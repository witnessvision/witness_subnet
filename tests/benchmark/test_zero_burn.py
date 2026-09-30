"""Zero-burn transition preserves finalized evaluations and historical weights."""
import json
import pytest
from witness.benchmark.ledger import Ledger
from witness.benchmark.protocol import RESUME_POLICY, ZERO_BURN_WINDOW, Result, policy_identity, weight_vector, decide


def test_zero_burn_migration_preserves_closed_history_and_restart(tmp_path):
    path = tmp_path/'chain.sqlite3'
    old = Ledger(path, activation_block=1, activation_epoch=0, policy=RESUME_POLICY)
    opening = {'id': 15, 'policy_hash': RESUME_POLICY}
    old.db.execute('INSERT INTO windows VALUES (?,?,?)', (14, '{}', '{"king":"kept"}'))
    old.db.execute('INSERT INTO windows VALUES (?,?,?)', (15, json.dumps(opening), None))
    old.set('active', opening)
    old.set('cursor', 100)
    old.db.execute('INSERT INTO commitments VALUES (?,?,?)', (99, 'validator', Result(15, 1, 'a'*64, RESUME_POLICY[:24], 'b'*64, 0, 0, 0, 0).commitment))
    tables = {t: [tuple(r) for r in old.db.execute(f'SELECT * FROM {t}')] for t in ('windows','commitments','uses')}
    old.close()
    new = Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity())
    assert new.active == opening and new.cursor == 100
    assert new.policy_for_window(15) == RESUME_POLICY
    assert new.policy_for_window(ZERO_BURN_WINDOW) == policy_identity()
    assert new.get('policy_migration')['first_window'] == ZERO_BURN_WINDOW
    for t, rows in tables.items():
        assert [tuple(r) for r in new.db.execute(f'SELECT * FROM {t}')] == rows
    new.close()
    Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity()).close()


def test_zero_burn_migration_rejects_already_reported_new_window(tmp_path):
    path = tmp_path/'chain.sqlite3'
    old = Ledger(path, activation_block=1, activation_epoch=0, policy=RESUME_POLICY)
    old.db.execute('INSERT INTO commitments VALUES (?,?,?)', (100, 'validator', Result(ZERO_BURN_WINDOW, 1, 'a'*64, RESUME_POLICY[:24], 'b'*64, 0, 0, 0, 0).commitment))
    old.close()
    with pytest.raises(ValueError, match='requires_unreported'):
        Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity())


def test_zero_burn_preserves_historical_decision_allocation():
    snapshot = {'uids': {'king': 127}, 'burn_uid': 240, 'validators': {}}
    king = {'hotkey': 'king', 'model_id': 'a'*64}
    args = dict(results=[], window=15, king=king, candidates={}, snapshot=snapshot)
    old = decide(policy=RESUME_POLICY, **args)
    assert old['uids'] == [127, 240] and old['weights'] == pytest.approx([.3, .7])
    new = decide(policy=policy_identity(), **args)
    assert new['uids'] == [127] and new['weights'] == [1.]
    assert weight_vector(None, snapshot)['weights'] == [1.]
