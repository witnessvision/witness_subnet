"""Signed optional display telemetry. Never used by the consensus ledger."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import httpx
from bittensor_wallet import Keypair

from witness.events import canonical_bytes
from witness.storage import write_private
from .status import DIGEST, scores_only
from .submission import SS58

DOMAIN = b'witness-dashboard-telemetry-1\0'
MAX_BYTES = 1024 * 1024
FIELDS = ('schema_version', 'validator', 'mode', 'block', 'updated_unix', 'policy_hash',
          'commitment_block', 'caught_up', 'progress', 'triggers', 'evaluations')
TRIGGER_FIELDS = ('hotkey', 'coldkey', 'model_id', 'block', 'position', 'status', 'reason', 'usage', 'window_id')
EVAL_FIELDS = ('validator', 'window_id', 'hotkey', 'model_id', 'quality', 'reward',
               'quality_upper', 'reward_upper', 'status', 'report_hash')
PROGRESS_FIELDS = ('stage', 'window_id', 'model_id', 'completed_clips', 'total_clips', 'updated_unix')


def public_status(status, reports=()):
    data = {k: status.get(k) for k in FIELDS}
    data['triggers'] = [{k: r.get(k) for k in TRIGGER_FIELDS} for r in status.get('triggers', [])]
    # Remote scores are reported observations; the local chain remains authoritative.
    data['evaluations'] = [{**{k: r.get(k) for k in EVAL_FIELDS}, 'available': False}
                           for r in status.get('evaluations', [])]
    data['progress'] = ({k: status['progress'].get(k) for k in PROGRESS_FIELDS}
                        if isinstance(status.get('progress'), dict) else None)
    # Closed-window reports travel as scores only; responses and media never leave the validator.
    data['reports'] = [{**scores_only(r), 'report_hash': r.get('report_hash')} for r in reports]
    return data


def _plain(value, depth=0):
    """Report values are numbers, short strings or small nested records, never free text."""
    if isinstance(value, dict):
        return depth < 6 and all(isinstance(k, str) and _plain(v, depth + 1) for k, v in value.items())
    if isinstance(value, list):
        return depth < 6 and all(_plain(v, depth + 1) for v in value)
    return value is None or isinstance(value, (bool, int, float)) or isinstance(value, str) and len(value) <= 256


class TelemetryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / 'telemetry'
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.Lock()

    def accept(self, message):
        if not isinstance(message, dict) or set(message) != {'status', 'signature'}:
            raise ValueError('invalid_telemetry_envelope')
        status = message['status']
        raw = canonical_bytes(status)
        if len(raw) > MAX_BYTES or not isinstance(status, dict):
            raise ValueError('invalid_telemetry_size')
        hotkey = status.get('validator', '')
        now = time.time()
        if (not isinstance(hotkey, str) or not SS58.fullmatch(hotkey)
                or status.get('schema_version') != 'witness-evaluator-status-2'
                or status.get('mode') not in ('evaluator', 'follower')
                or type(status.get('block')) is not int or status['block'] < 0
                or type(status.get('updated_unix')) not in (int, float)
                or abs(now - status['updated_unix']) > 120
                or type(status.get('caught_up')) is not bool):
            raise ValueError('invalid_telemetry_status')
        peers = json.loads((self.root / 'telemetry-peers.json').read_text())
        if now - peers['updated_unix'] > 120 or hotkey not in peers['validators'] or hotkey == peers['self']:
            raise ValueError('telemetry_validator_not_eligible')
        try:
            signature = bytes.fromhex(message['signature'])
            verified = Keypair(ss58_address=hotkey).verify(DOMAIN + raw, signature)
        except (TypeError, ValueError):
            verified = False
        if not verified:
            raise ValueError('invalid_telemetry_signature')
        reports = status.get('reports')
        if not isinstance(reports, list) or len(reports) > 64 or not all(isinstance(r, dict) for r in reports):
            raise ValueError('invalid_telemetry_reports')
        data = public_status(status, reports)
        # Reject extra/private payload fields rather than storing them accidentally.
        if canonical_bytes(data) != raw:
            raise ValueError('invalid_public_telemetry_fields')
        if len(data['triggers']) > 4096 or len(data['evaluations']) > 4096:
            raise ValueError('telemetry_rows_limit')
        for rows in (data['triggers'], data['evaluations'], [data['progress']] if data['progress'] else []):
            for row in rows:
                if any(isinstance(v, (dict, list)) or isinstance(v, str) and len(v) > 256 for v in row.values()):
                    raise ValueError('invalid_telemetry_row')
        published = {(r['window_id'], r['model_id']): r['report_hash'] for r in data['evaluations']}
        for report in data['reports']:
            window, model = report.get('window_id'), report.get('model_id')
            if (report.get('available') is not True or report.get('validator') != hotkey or not _plain(report)
                    or type(window) is not int or not 0 <= window < 2**32 or not DIGEST.fullmatch(str(model))
                    or published.get((window, model)) != report.get('report_hash')):
                raise ValueError('invalid_telemetry_report')
        target = self.directory / (hotkey + '.json')
        with self.lock:
            old = json.loads(target.read_text()) if target.exists() else {}
            if status['updated_unix'] <= old.get('updated_unix', 0) or now - old.get('received_unix', 0) < 10:
                raise ValueError('telemetry_replay_or_rate_limit')
            for report in data['reports']:
                write_private(self.directory / 'reports' / hotkey / f"{report['window_id']}-{report['model_id']}.json", report)
            status = {k: v for k, v in data.items() if k != 'reports'}
            write_private(target, {**status, 'stake': peers['validators'][hotkey] / 1e9,
                                   'received_unix': now, 'telemetry_verified': True})
        return {'accepted': True}


class TelemetrySender:
    def __init__(self, url, keypair):
        if not url.startswith('https://'):
            raise ValueError('telemetry_requires_https')
        self.url, self.keypair = url, keypair
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='witness-telemetry')
        self.future = None

    def submit(self, status, reports=()):
        if self.future and not self.future.done():
            return
        data = public_status(status, reports)
        signature = self.keypair.sign(DOMAIN + canonical_bytes(data)).hex()
        self.future = self.executor.submit(self._send, {'status': data, 'signature': signature})

    def _send(self, message):
        try:
            with httpx.Client(timeout=10, trust_env=False) as client:
                client.post(self.url, json=message).raise_for_status()
        except httpx.HTTPError:
            return False  # Display transport failure must never interrupt chain work.
        return True

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
