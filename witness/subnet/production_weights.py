"""Guarded single writer: 70% burn/30% winner only with complete evidence."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from witness.events import content_hash
from witness.events_evaluation import PROMPT_HASH
from witness.storage import write_private
from witness.subnet.burn import burn_state
from witness.subnet.chain import BittensorChainAdapter, WeightSubmission
from witness.subnet.mainnet import FINNEY_GENESIS


def promotion_gate(reports, calibration, own_hotkey):
    failures = []
    if (not calibration or calibration.get('total') != 300 or calibration.get('passed') is not True
            or calibration.get('accuracy', 0) < .95
            or calibration.get('contradiction_acceptance') is None
            or calibration['contradiction_acceptance'] > .02
            or calibration.get('prompt_hash') != PROMPT_HASH):
        failures.append('luna_calibration_not_passed')
    if len(reports) != 3 or len({r['round_id'] for r in reports}) != 3:
        failures.append('three_distinct_rounds_required')
    if len(reports) == 3 and any(b['epoch'] != a['epoch']+1 for a,b in zip(reports,reports[1:])):
        failures.append('nonconsecutive_epochs')
    own = []
    for report in reports:
        evaluator = report['identity']['evaluator']
        if (evaluator.get('judge_model') != 'gpt-5.6-luna' or evaluator.get('judge_effort') != 'low'
                or evaluator.get('evaluator_id') != calibration.get('evaluator_id')
                or report['identity'].get('calibration_hash') != content_hash(calibration)):
            failures.append('evaluator_identity_mismatch')
        if not report.get('complete'):
            failures.append('incomplete_comparison')
        if any(m['planned'] != 5 or m['sent'] != 5 for m in report['miners']):
            failures.append('dispatch_count_mismatch')
        candidates = [m for m in report['miners'] if m['hotkey'] == own_hotkey]
        if len(candidates) != 1:
            failures.append('own_miner_absent')
            continue
        miner = candidates[0]
        own.append(miner)
        if miner['uid'] != 117 or miner['valid'] != 5 or miner['scored'] != 5 or miner['max_elapsed_s'] >= 180:
            failures.append('own_miner_operational_failure')
    if len(own) != 3 or any(m['mean_f1'] is None or m['mean_score'] is None for m in own):
        failures.append('incomplete_own_scores')
    elif sum(m['mean_f1'] for m in own)/3 < .98 or sum(m['mean_score'] for m in own)/3 < .96:
        failures.append('own_miner_quality_failure')
    if reports and len({content_hash(r['identity']) for r in reports}) != 1:
        failures.append('ranking_series_changed')
    return {'passed': not failures, 'failures': sorted(set(failures))}


def read_latest(path):
    # Read the journal, not a possibly stale JSON projection after a crash.
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        row = db.execute('''SELECT r.status,r.report,c.epoch FROM rounds r
            JOIN cursor c ON c.id=1 ORDER BY r.epoch DESC LIMIT 1''').fetchone()
        if not row or row[0] != 'complete':
            return None
        report = json.loads(row[1])
        # Source preparation can fail before a round has tasks. Its consumed
        # epoch is still durable; never pay using the preceding comparison.
        return report if row[2] <= report['finish_epoch'] else None
    finally:
        db.close()


def activation_gate(activation):
    """Keep failed acceptance evidence when an operator changes launch policy."""
    gate = promotion_gate(activation['reports'], activation['calibration'], activation['own_hotkey'])
    authorization = activation.get('operator_authorization', {})
    waived = []
    if (authorization.get('policy') == '70_burn_30_winner'
            and authorization.get('waive_initial_own_quality') is True
            and isinstance(authorization.get('reason'), str) and authorization['reason'].strip()
            and isinstance(authorization.get('authorized_at'), str) and authorization['authorized_at'].strip()
            and 'own_miner_quality_failure' in gate['failures']):
        waived = ['own_miner_quality_failure']
    failures = [f for f in gate['failures'] if f not in waived]
    if activation.get('operational_probes_passed') is not True:
        failures.append('operational_probes_not_passed')
    return {'passed': not failures, 'failures': failures, 'original_gate': gate,
            'waived_failures': waived}


def read_progress(path):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        row = db.execute('''SELECT r.id,r.epoch,r.status,r.report,c.epoch FROM rounds r
            JOIN cursor c ON c.id=1 ORDER BY r.epoch DESC LIMIT 1''').fetchone()
        if not row:
            return {'round_id': None, 'epoch': -1, 'status': 'absent', 'report': None, 'cursor': -1}
        return dict(zip(('round_id', 'epoch', 'status', 'report', 'cursor'),
                        (*row[:3], json.loads(row[3]) if row[3] else None, row[4])))
    finally:
        db.close()


def round_decision(progress, *, burn_uid, epoch, expected_identity, registered_hotkey):
    # A running round has no result yet. Commit once it closes; do not commit
    # full burn merely because a polling tick falls during normal evaluation.
    # A round spanning more than one further epoch is stale and fails to burn.
    if progress['status'] == 'running' and epoch <= progress['epoch'] + 1:
        return None
    report = progress['report']
    if (progress['status'] != 'complete' or not report
            or progress['cursor'] > report['finish_epoch']):
        report = None
    uids, weights = policy(report, burn_uid=burn_uid, epoch=epoch,
        expected_identity=expected_identity, registered_hotkey=registered_hotkey)
    source = {'round_id': progress['round_id'], 'consumed_epoch': progress['cursor'],
              'uids': uids, 'weights': weights}
    return {**source, 'decision_id': content_hash(source)}


def submission_records(root):
    records = []
    for path in sorted((root/'submissions').glob('*.json')):
        record = json.loads(path.read_text())
        if record['submission']['status'] in ('prepared', 'unknown'):
            raise ValueError('ambiguous_submission_requires_reconciliation')
        records.append(record)
    return records


def pending_is_known(pending, records, block_number):
    """Only our persisted receipts may coexist with the next round's commit."""
    known = {block_number(r['submission']['block_hash']) for r in records
             if r['submission']['status'] in ('included', 'finalized', 'submitted')
             and r['submission'].get('block_hash')}
    return all(p['commit_block'] in known for p in pending)


def policy(report, *, burn_uid, epoch, expected_identity, registered_hotkey):
    burn = ([burn_uid], [1.])
    if (not report or report.get('complete') is not True or not report.get('winner')
            or report.get('identity') != expected_identity or epoch > report['finish_epoch']+1):
        return burn
    winner = report['winner']
    if (not winner.get('eligible') or winner['mean_score'] <= 0 or winner['uid'] == burn_uid
            or registered_hotkey(winner['uid']) != winner['hotkey']):
        return burn
    return [burn_uid, winner['uid']], [.7, .3]


def pending_commits(chain, block_hash):
    s = chain.subtensor.substrate
    pending = []
    for _, entries in s.query_map('SubtensorModule','TimelockedWeightCommits',[chain.netuid],block_hash=block_hash):
        for hotkey, at, _, reveal_round in getattr(entries, 'value', entries):
            if hotkey == chain.validator_hotkey:
                pending.append({'commit_block':at,'reveal_round':reveal_round})
    legacy = s.query('SubtensorModule','WeightCommits',[chain.netuid,chain.validator_hotkey],block_hash=block_hash).value
    return pending, legacy


async def run(chain, config):
    root = Path(config['root'])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    activation = json.loads(Path(config['activation']).read_text())
    gate = activation_gate(activation)
    if not gate['passed']:
        raise ValueError('promotion_gates_not_met')
    expected = activation['reports'][0]['identity']
    path = root/'submission.json'
    if path.exists() and json.loads(path.read_text()).get('submission',{}).get('status') in ('prepared','unknown'):
        raise ValueError('ambiguous_submission_requires_reconciliation')
    # This handover check complements the writer lock: the superseded legacy
    # service predates this lock and must be stopped before starting this writer.
    previous = subprocess.run(['systemctl','is-active','witness-burn.service'],
                              capture_output=True, text=True, timeout=10)
    if previous.stdout.strip() not in ('inactive','failed'):
        raise ValueError('previous_weight_writer_not_stopped')
    with (root/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root/'submissions').mkdir(mode=0o700, exist_ok=True)
        write_private(root/'activation-gate.json', gate)
        while True:
            previous = subprocess.run(['systemctl','is-active','witness-burn.service'],
                                      capture_output=True, text=True, timeout=10)
            if previous.stdout.strip() not in ('inactive','failed'):
                raise ValueError('previous_weight_writer_not_stopped')
            records = submission_records(root)
            state = burn_state(chain, None)
            pending, legacy = pending_commits(chain, state['block_hash'])
            epoch = chain.epoch_state()['epoch_index']
            def registered(uid):
                return chain.subtensor.substrate.query('SubtensorModule','Keys',[chain.netuid,uid],block_hash=state['block_hash']).value
            try:
                progress = read_progress(Path(config['scheduler']))
            except (sqlite3.Error, ValueError, OSError):
                progress = {'round_id': None, 'epoch': epoch, 'cursor': epoch,
                            'status': 'unavailable', 'report': None}
            decision = round_decision(progress, burn_uid=state['burn_uid'], epoch=epoch,
                                      expected_identity=expected, registered_hotkey=registered)
            handover = root/'handover.json'
            if not handover.exists() and not pending and not legacy:
                write_private(handover, {'block': state['block'], 'block_hash': state['block_hash'],
                                        'old_commits_drained': True})
            reconciled = (handover.exists() and not legacy
                          and pending_is_known(pending, records, chain.subtensor.substrate.get_block_number))
            observation = {'unix':time.time(),'block':state['block'],'block_hash':state['block_hash'],
                           'epoch': epoch, 'source_status': progress['status'],
                           'decision': decision, 'handover_complete': handover.exists(),
                           'pending_reconciled': reconciled,
                           'desired': {'uids': decision['uids'], 'weights': decision['weights']} if decision else None,
                           'pending_commits':pending,'legacy_commits':legacy,
                           'active_weights':chain.subtensor.substrate.query('SubtensorModule','Weights',
                              [chain.netuid,state['validator_uid']],block_hash=state['block_hash']).value}
            write_private(root/'observation.json', observation)
            # Each closed round has a durable id. Existing known commits are
            # expected with timelock reveal; unknown/old commitments block us.
            already_sent = decision and any(r.get('decision', {}).get('decision_id') == decision['decision_id']
                                            for r in records)
            if decision and not already_sent and not state['blocks_until_submission'] and reconciled:
                record = {**observation,'submission':WeightSubmission('prepared').as_dict()}
                journal = root/'submissions'/(decision['decision_id']+'.json')
                write_private(journal, record)
                write_private(path, record)
                try:
                    submission = chain.set_weights(decision['uids'], decision['weights'])
                except Exception as error:
                    submission = WeightSubmission('unknown', error_type=type(error).__name__)
                record['submission'] = submission.as_dict()
                write_private(journal, record)
                write_private(path, record)
                if submission.status in ('unknown','prepared'):
                    raise RuntimeError('ambiguous_submission_requires_reconciliation')
            await asyncio.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    chain = BittensorChainAdapter(netuid=20, network=config.get('network','finney'), wallet_name=config['wallet'],
        wallet_hotkey=config['wallet_hotkey'], wallet_path=config['wallet_path'], with_dendrite=False)
    try:
        if chain.validator_hotkey != config['expected_hotkey'] or chain.subtensor.substrate.get_block_hash(0) != FINNEY_GENESIS:
            raise ValueError('wrong_chain_or_writer')
        asyncio.run(run(chain, config))
    except (ValueError, RuntimeError) as error:
        if str(error) in {'wrong_chain_or_writer', 'promotion_gates_not_met',
                          'previous_weight_writer_not_stopped',
                          'ambiguous_submission_requires_reconciliation'}:
            print(json.dumps({'stopped': str(error)}), file=sys.stderr)
            raise SystemExit(78) from None
        raise
    finally:
        chain.close()


if __name__ == '__main__':
    main()
