"""Read-only display projection. Never supplies inputs to validator consensus."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from witness.events import content_hash
from witness.storage import sha256_file
from .submission import SS58

PAGE = Path(__file__).with_name('dashboard.html')
DIGEST = re.compile('[0-9a-f]{64}')


def load(source):
    if str(source).startswith(('http://', 'https://')):
        with httpx.stream('GET', source, timeout=5, follow_redirects=False) as response:
            response.raise_for_status()
            body = bytearray()
            for part in response.iter_bytes():
                body.extend(part)
                if len(body) > 8 * 1024 * 1024:
                    raise ValueError('status_too_large')
            return json.loads(body)
    path = Path(source)
    return json.loads((path / 'queue.json' if path.is_dir() else path).read_text())


def merge(sources):
    statuses, errors = [], []
    for index, source in enumerate(sources):
        try:
            state = load(source)
            if state.get('schema_version') != 'witness-evaluator-status-2' or not SS58.fullmatch(state['validator']):
                raise ValueError('invalid_status_schema')
            statuses.append((index, state))
        except (OSError, ValueError, KeyError, httpx.HTTPError) as error:
            errors.append({'source': f'source_{index + 1}', 'error': type(error).__name__})
    # The explicitly configured first source is the chain projection authority.
    # Unavailability means unknown, never a majority vote from dashboard sources.
    authority = next((s for i, s in statuses if i == 0), {})
    validators, queues, evaluations = [], {}, []
    for _, state in statuses:
        hotkey = state['validator']
        if hotkey in queues:
            continue
        rows = state.get('triggers', [])
        fresh = state.get('caught_up', False) and time.time() - state.get('updated_unix', 0) <= 120
        validators.append({'hotkey': hotkey, 'mode': state['mode'], 'stake': state.get('stake'),
                           'block': state['block'], 'commitment_block': state.get('commitment_block'),
                           'freshness': 'fresh' if fresh else 'stale',
                           'progress': state.get('progress'),
                           'used_count': sum(r.get('usage') == 'consumed' for r in rows),
                           'reserved_count': sum(r.get('usage') == 'reserved' for r in rows)})
        queues[hotkey] = sorted([{k: r.get(k) for k in ('hotkey', 'coldkey', 'model_id', 'block', 'position',
                                                       'status', 'reason', 'usage', 'window_id')}
                                 for r in rows if r.get('usage') == 'reserved'],
                                key=lambda r: (r['position'] is None, r['position'] or 0, r['block']))
        evaluations += state.get('evaluations', [])
    return {'schema_version': 'witness-dashboard-2', 'is_demo': False, 'block': authority.get('block'),
            'policy_hash': authority.get('policy_hash'), 'window': authority.get('window'),
            'king': authority.get('king'), 'weights': authority.get('weights', {}),
            'validators': validators, 'queues': queues, 'evaluations': evaluations, 'errors': errors}


class Projection:
    def __init__(self, sources, peer_directory=None):
        self.configured_sources = list(map(str, sources))
        self.peer_directory = Path(peer_directory) if peer_directory else None

    @property
    def sources(self):
        peers = sorted(self.peer_directory.glob('*.json')) if self.peer_directory else []
        return self.configured_sources + list(map(str, peers))

    def state(self):
        return merge(self.sources)

    def source(self, validator):
        if not SS58.fullmatch(validator):
            raise ValueError('invalid_validator')
        for source in self.sources:
            try:
                status = load(source)
                if status.get('validator') == validator:
                    return source, status
            except (OSError, ValueError, httpx.HTTPError):
                continue
        raise FileNotFoundError('validator_unavailable')

    def hotkeys(self, validator, state='consumed', q='', page=1):
        if state not in ('consumed', 'reserved') or not 1 <= page <= 1_000_000 or len(q) > 128:
            raise ValueError('invalid_hotkey_query')
        _, status = self.source(validator)
        rows = [{k: r.get(k) for k in ('hotkey', 'coldkey', 'model_id', 'status', 'reason', 'window_id')}
                for r in status.get('triggers', []) if r.get('usage') == state
                and any(q.casefold() in str(r.get(k, '')).casefold() for k in ('hotkey', 'coldkey', 'model_id'))]
        return {'total': len(rows), 'page': page, 'page_size': 50, 'rows': rows[(page-1)*50:page*50]}

    @staticmethod
    def _remote_base(source):
        parsed = urlsplit(source)
        return f'{parsed.scheme}://{parsed.netloc}'

    @staticmethod
    def _root(source):
        path = Path(source)
        return path if path.is_dir() else path.parent

    @staticmethod
    def _closed(root, window):
        path = root / 'chain.sqlite3'
        if not path.exists():
            return False
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            row = db.execute('SELECT decision FROM windows WHERE id=?', (window,)).fetchone()
        return bool(row and row[0])

    def evaluation(self, validator, window, model):
        if not DIGEST.fullmatch(model) or not 0 <= window < 2**32:
            raise ValueError('invalid_evaluation')
        source, status = self.source(validator)
        row = next((r for r in status.get('evaluations', []) if r['window_id'] == window and r['model_id'] == model), None)
        if row is None or not row.get('available'):
            return {'available': False, 'reason': 'window_open_or_result_unavailable'}
        if source.startswith(('http://', 'https://')):
            body = load(f'{self._remote_base(source)}/api/evaluations/{validator}/{window}/{model}')
            if not body.get('available'):
                return body
            body = {k: v for k, v in body.items() if k != 'report_hash'}
        else:
            root = self._root(source)
            if not self._closed(root, window):
                return {'available': False, 'reason': 'window_not_finalized'}
            digest = row.get('report_hash', '')
            if not DIGEST.fullmatch(digest):
                raise ValueError('invalid_report_hash')
            path = root / 'reports' / (digest + '.json')
            if not path.exists():
                return {'available': False, 'reason': 'report_unavailable'}
            body = json.loads(path.read_text())
        if content_hash(body) != row['report_hash']:
            raise ValueError('report_hash_mismatch')
        return {**body, 'report_hash': row['report_hash']}

    def media(self, validator, window, digest):
        if not DIGEST.fullmatch(digest) or not 0 <= window < 2**32:
            raise ValueError('invalid_media')
        source, status = self.source(validator)
        evaluations = [r for r in status.get('evaluations', []) if r['window_id'] == window and r.get('available')]
        allowed = False
        for row in evaluations:
            report = self.evaluation(validator, window, row['model_id'])
            if any(c['id'] == digest for v in report.get('videos', []) for c in v['clips']):
                allowed = True
                break
        if not allowed:
            raise FileNotFoundError('media_not_in_closed_report')
        if source.startswith(('http://', 'https://')):
            # The caller streams/proxies only this fixed route from its configured evaluator.
            return f'{self._remote_base(source)}/api/media/{validator}/{window}/{digest}.mp4'
        root = self._root(source).resolve()
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
                self._json(view.state())
            elif parsed.path == '/queue.json' and self.server.local_root:
                self._json(load(self.server.local_root))
            elif parsed.path == '/api/hotkeys':
                q = parse_qs(parsed.query)
                self._json(view.hotkeys(q.get('validator', [''])[0], q.get('state', ['consumed'])[0],
                                       q.get('q', [''])[0], int(q.get('page', ['1'])[0])))
            elif match := re.fullmatch(r'/api/evaluations/([^/]+)/([0-9]+)/([0-9a-f]{64})', parsed.path):
                self._json(view.evaluation(match[1], int(match[2]), match[3]))
            elif match := re.fullmatch(r'/api/media/([^/]+)/([0-9]+)/([0-9a-f]{64})\.mp4', parsed.path):
                target = view.media(match[1], int(match[2]), match[3])
                if isinstance(target, str):
                    with httpx.stream('GET', target, timeout=30, headers={'Range': self.headers.get('Range', 'bytes=0-')}) as r:
                        r.raise_for_status()
                        self.send_response(r.status_code)
                        for key in ('Content-Length', 'Content-Range', 'Accept-Ranges', 'Content-Type'):
                            if key in r.headers:
                                self.send_header(key, r.headers[key])
                        self.end_headers()
                        total = 0
                        for chunk in r.iter_bytes():
                            total += len(chunk)
                            if total > 256 * 1024 * 1024:
                                break
                            self.wfile.write(chunk)
                else:
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
            elif parsed.path == '/':
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        except (ValueError, KeyError):
            self._json({'error': 'invalid_request_or_evidence'}, 400)
        except (OSError, StopIteration, httpx.HTTPError, sqlite3.Error):
            self._json({'error': 'evidence_unavailable'}, 503)


def serve_status(root, host, port):
    server = ThreadingHTTPServer((host, port), _Handler)
    server.projection = Projection([str(root)])
    server.local_root = str(root)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, help='First source is the authoritative chain projection')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8098)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.projection, server.local_root = Projection(args.source), None
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
