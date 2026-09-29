"""Optional read-only validator status and finalized evidence export. No UI."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import threading
from urllib.parse import urlsplit
import httpx
from witness.events import content_hash
from witness.storage import sha256_file
from .submission import SS58

DIGEST = re.compile('[0-9a-f]{64}')

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
        return {**body, 'report_hash': digest}

    def media(self, validator, window, digest):
        if not DIGEST.fullmatch(digest) or not 0 <= window < 2**32:
            raise ValueError('invalid_media')
        status = self._status(validator)
        evaluations = [r for r in status.get('evaluations', []) if r['window_id'] == window and r.get('available')]
        allowed = False
        for row in evaluations:
            report = self.evaluation(validator, window, row['model_id'])
            if any(c['id'] == digest for v in report.get('videos', []) for c in v['clips']):
                allowed = True
                break
        if not allowed:
            raise FileNotFoundError('media_not_in_closed_report')
        root = self.root.resolve()
        rows = json.loads((root / 'windows' / str(window) / 'batch.json').read_text())
        row = next(r for r in rows if r['clip_sha256'] == digest)
        path = Path(row['media_path']).resolve(strict=True)
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError('media_integrity_failed')
        return path


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _json(self, body, code=200):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlsplit(self.path)
        view = self.server.projection
        try:
            if parsed.path in ('/api/subnet', '/api/state'):
                self._json(view.status())
            elif parsed.path == '/queue.json' and self.server.local_root:
                self._json(view.status())
            elif match := re.fullmatch(r'/api/evaluations/([^/]+)/([0-9]+)/([0-9a-f]{64})', parsed.path):
                self._json(view.evaluation(match[1], int(match[2]), match[3]))
            elif match := re.fullmatch(r'/api/media/([^/]+)/([0-9]+)/([0-9a-f]{64})\.mp4', parsed.path):
                target = view.media(match[1], int(match[2]), match[3])
                size = target.stat().st_size
                start, end = 0, size-1
                requested = self.headers.get('Range')
                if requested:
                    limits = re.fullmatch(r'bytes=([0-9]+)-([0-9]*)', requested)
                    if not limits:
                        self.send_error(416)
                        return
                    start = int(limits[1]); end = min(int(limits[2]) if limits[2] else size-1, size-1)
                    if not 0 <= start <= end < size:
                        self.send_error(416)
                        return
                self.send_response(206 if requested else 200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(end-start+1))
                if requested:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                with target.open('rb') as stream:
                    stream.seek(start)
                    remaining = end-start+1
                    while remaining:
                        chunk = stream.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_error(404)
        except (ValueError, KeyError):
            self._json({'error': 'invalid_request_or_evidence'}, 400)
        except (OSError, StopIteration, httpx.HTTPError, sqlite3.Error):
            self._json({'error': 'evidence_unavailable'}, 503)


def serve_status(root, host, port):
    server = ThreadingHTTPServer((host, port), _Handler)
    server.projection = Evidence(root)
    server.local_root = str(root)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

