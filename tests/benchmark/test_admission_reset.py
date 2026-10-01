import json
import pytest
from witness.benchmark.ledger import Ledger, HOTKEY_RESET_WINDOW
from witness.benchmark.protocol import policy_identity, WINDOW_EPOCHS
from witness.benchmark.triggers import Triggers
from test_validator import snapshot, entry, MINERS, VALS


def test_reset_preserves_king_history_and_requires_fresh_submission(tmp_path):
    path = tmp_path/'chain.db'
    ledger = Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity())
    king = entry(0)
    ledger.set('king', king)
    ledger.ingest(snapshot(1, epoch_index=(HOTKEY_RESET_WINDOW-1)*WINDOW_EPOCHS), [entry(1)])
    ledger.db.execute('INSERT INTO uses VALUES (?,?,?,?)', (VALS[0], MINERS[1], 30, '{}'))
    ledger.ingest(snapshot(2, epoch_index=HOTKEY_RESET_WINDOW*WINDOW_EPOCHS), [])
    assert ledger.get('king') == king
    assert ledger.submissions() == [] and ledger.usage(VALS[0]) == {}
    assert ledger.active['candidates'] == {}
    assert len(ledger.history(1, 3)) == 1
    assert ledger.db.execute('SELECT COUNT(*) FROM admission_archive').fetchone()[0] == 2
    assert ledger.is_closed(HOTKEY_RESET_WINDOW-1)
    ledger.ingest(snapshot(3, epoch_index=HOTKEY_RESET_WINDOW*WINDOW_EPOCHS), [entry(1, block=3)])
    assert len(ledger.submissions()) == 1
    assert ledger.active['candidates'] == {}
    ledger.close()
    ledger = Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity())
    ledger.ingest(snapshot(4, epoch_index=(HOTKEY_RESET_WINDOW+1)*WINDOW_EPOCHS), [])
    assert len(ledger.active['candidates']) == 1
    assert ledger.db.execute('SELECT COUNT(*) FROM admission_archive').fetchone()[0] == 2


def test_trigger_reset_archives_used_withdrawn_and_inflight_and_ignores_old(tmp_path):
    triggers = Triggers(tmp_path/'triggers.db')
    kwargs = dict(coldkeys={h:'owner' for h in MINERS}, uids={h:i for i,h in enumerate(MINERS)})
    triggers.observe([entry(i) for i in range(3)], MINERS, **kwargs)
    ids = [r['id'] for r in triggers.rows()]
    triggers.finish(ids[0], 'done', result={'quality':.5})
    triggers.start(ids[1], 31)
    triggers.db.execute("UPDATE triggers SET status='withdrawn' WHERE id=?", (ids[2],))
    reset = {'window':32, 'block':10}
    triggers.apply_reset(reset)
    triggers.finish(ids[1], 'done')
    assert all(r['status'] == 'reset' for r in triggers.rows())
    assert triggers.pending() == []
    assert triggers.observe([entry(i) for i in range(3)], MINERS, **kwargs) == []
    triggers.observe([entry(0, block=10)], MINERS, **kwargs)
    assert len(triggers.pending()) == 1
    triggers.apply_reset(reset)
    assert len(triggers.pending()) == 1
    archived = [json.loads(r[0]) for r in triggers.db.execute('SELECT record FROM admission_archive')]
    assert {r['status'] for r in archived} == {'done','running','withdrawn'}
    with pytest.raises(ValueError, match='reset_changed'):
        triggers.apply_reset({'window':32,'block':11})


def test_second_upgrade_preserves_original_schedule_and_restart(tmp_path):
    path = tmp_path/'chain.db'
    old, middle = 'e'*64, 'd'*64
    first = {'first_window':30, 'previous_policy':old}
    ledger = Ledger(path, activation_block=1, activation_epoch=0, policy=old)
    ledger.close()
    ledger = Ledger(path, activation_block=1, activation_epoch=0, policy=middle, policy_upgrade=first)
    ledger.close()
    second = {'first_window':32,'previous_policy':middle,'prior_upgrade':first}
    ledger = Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity(), policy_upgrade=second)
    assert ledger.policy_for_window(29) == old
    assert ledger.policy_for_window(30) == middle
    assert ledger.policy_for_window(31) == middle
    assert ledger.policy_for_window(32) == policy_identity()
    ledger.close()
    Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity(), policy_upgrade=second).close()
    with pytest.raises(ValueError, match='schedule_changed'):
        Ledger(path, activation_block=1, activation_epoch=0, policy=policy_identity(), policy_upgrade=first)


def test_only_reviewed_reset_release_can_evaluate_pre_activation(tmp_path):
    from witness.benchmark.ledger import PRE_RESET_POLICY, validate_upgrade
    upgrade = {'first_window':32,'previous_policy':PRE_RESET_POLICY,'admission_reset':32,
               'prior_upgrade':{'first_window':30,'previous_policy':'e'*64}}
    ledger = Ledger(tmp_path/'chain.db', activation_block=1, activation_epoch=0,
                    policy=policy_identity(), policy_upgrade=upgrade)
    assert ledger.evaluation_compatible(30) and ledger.evaluation_compatible(31)
    assert not ledger.evaluation_compatible(29)
    assert not ledger.evaluation_compatible(32)
    with pytest.raises(ValueError, match='invalid_admission_reset_upgrade'):
        validate_upgrade({**upgrade,'previous_policy':'d'*64}, policy_identity())


def test_explicit_replacement_only_changes_unopened_schedule(tmp_path):
    from witness.benchmark.ledger import PRE_RESET_POLICY
    path=tmp_path/'ledger.db';pending='c'*64
    prior={'first_window':30,'previous_policy':'e'*64}
    old={'first_window':32,'previous_policy':PRE_RESET_POLICY,'prior_upgrade':prior,'admission_reset':32}
    ledger=Ledger(path,activation_block=1,activation_epoch=0,policy=pending,policy_upgrade=old)
    ledger.db.execute('INSERT INTO windows VALUES (?,?,NULL)',(30,'preserved'));ledger.close()
    new={**old,'first_window':31}
    with pytest.raises(ValueError,match='schedule_changed'):
        Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity(),policy_upgrade=new)
    ledger=Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity(),policy_upgrade=new,
                  replace_pending_policy=pending)
    assert ledger.db.execute('SELECT opening FROM windows WHERE id=30').fetchone()[0]=='preserved'
    assert ledger.get('pending_policy_replacement')['from']==pending
    assert ledger.policy_for_window(30)==PRE_RESET_POLICY
    assert ledger.policy_for_window(31)==policy_identity()
    ledger.close()
    Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity(),policy_upgrade=new,
           replace_pending_policy=pending).close()


def test_pending_replacement_rejects_opened_boundary(tmp_path):
    from witness.benchmark.ledger import PRE_RESET_POLICY
    path=tmp_path/'ledger.db';pending='c'*64
    old={'first_window':32,'previous_policy':PRE_RESET_POLICY,'admission_reset':32}
    ledger=Ledger(path,activation_block=1,activation_epoch=0,policy=pending,policy_upgrade=old)
    ledger.db.execute('INSERT INTO windows VALUES (?,?,NULL)',(31,'already_open'));ledger.close()
    with pytest.raises(ValueError,match='requires_unopened_window'):
        Ledger(path,activation_block=1,activation_epoch=0,policy=policy_identity(),
               policy_upgrade={**old,'first_window':31},replace_pending_policy=pending)
