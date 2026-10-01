"""Local read-only status and finalized evidence. Publishes only scores, never model outputs or media."""
import json
from pathlib import Path
import re
import sqlite3
from witness.events import content_hash
from .submission import SS58

DIGEST = re.compile('[0-9a-f]{64}')
SCORES = ('quality', 'reward', 'latency_s', 'time_score')
REPORT_FIELDS = ('schema_version', 'available', 'validator', 'evaluation_status', 'early_stop', 'window_id',
                 'hotkey', 'model_id', 'opponent', 'policy_hash', 'total')
VIDEO_FIELDS = ('id', 'quality', 'reward', 'king_quality', 'king_reward')


def scores_only(report):
    """Aggregate and per-clip scores of a report; model responses and clip media stay private."""
    if not report.get('available'):
        return report
    pick = lambda row, keys: {k: row.get(k) for k in keys}
    clip = lambda c: {**pick(c, ('id', 'start', 'duration') + SCORES),
                      'king': pick(c['king'], SCORES) if c.get('king') else None}
    return {**pick(report, REPORT_FIELDS),
            'videos': [{**pick(v, VIDEO_FIELDS), 'clips': [clip(c) for c in v.get('clips', [])]}
                       for v in report.get('videos', [])]}

class Evidence:
    def __init__(self, root):
        self.root = Path(root)

    def status(self):
        return json.loads((self.root / 'queue.json').read_text())

    def _status(self, validator):
        if not SS58.fullmatch(validator):
            raise ValueError('invalid_validator')
        status = self.status()
        if status.get('validator') != validator:
            raise FileNotFoundError('validator_unavailable')
        return status

    def _closed(self, window):
        path = self.root / 'chain.sqlite3'
        if not path.exists():
            return False
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            row = db.execute('SELECT decision FROM windows WHERE id=?', (window,)).fetchone()
        return bool(row and row[0])

    def evaluation(self, validator, window, model):
        if not DIGEST.fullmatch(model) or not 0 <= window < 2**32:
            raise ValueError('invalid_evaluation')
        status = self._status(validator)
        row = next((r for r in status.get('evaluations', []) if r['window_id'] == window and r['model_id'] == model), None)
        if row is None or not row.get('available'):
            return {'available': False, 'reason': 'window_open_or_result_unavailable'}
        if not self._closed(window):
            return {'available': False, 'reason': 'window_not_finalized'}
        digest = row.get('report_hash', '')
        if not DIGEST.fullmatch(digest):
            raise ValueError('invalid_report_hash')
        path = self.root / 'reports' / (digest + '.json')
        if not path.exists():
            return {'available': False, 'reason': 'report_unavailable'}
        body = json.loads(path.read_text())
        if content_hash(body) != digest:
            raise ValueError('report_hash_mismatch')
        return {**scores_only(body), 'report_hash': digest}

    def closed_reports(self, limit=24):
        """Scores of this validator's latest closed-window reports, for signed telemetry."""
        status = self.status()
        rows = sorted((r for r in status.get('evaluations', []) if r.get('available')),
                      key=lambda r: r['window_id'], reverse=True)[:limit]
        reports = []
        for row in rows:
            try:
                report = self.evaluation(status['validator'], row['window_id'], row['model_id'])
            except (OSError, ValueError, sqlite3.Error):
                continue  # A missing or damaged report is not published; status still is.
            if report.get('available'):
                reports.append(report)
        return reports
