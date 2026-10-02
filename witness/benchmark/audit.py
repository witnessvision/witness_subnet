"""Bounded operational audit records. Never logs URLs, query strings, headers or bodies."""
import ipaddress
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import threading
import time

_lock = threading.Lock()
_handlers = {}


class PrivateRotatingHandler(RotatingFileHandler):
    def _open(self):
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, 'a', encoding='utf-8')


def record(event, *, status, method='GET', route='unmatched', peer=None, identity=None, elapsed_s=0):
    destination = os.environ.get('WITNESS_AUDIT_LOG')
    if not destination:
        return
    try:
        peer = str(ipaddress.ip_address(peer)) if peer else None
    except ValueError:
        peer = None
    if identity and not re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{46,48}', identity):
        identity = None
    # Callers supply route templates, not user-controlled request targets.
    data = {'unix': time.time(), 'event': event, 'status': int(status),
            'method': method if method in ('GET','HEAD','POST','PUT','DELETE','PATCH','OPTIONS') else 'OTHER',
            'route': route, 'peer_ip': peer, 'authenticated_hotkey': identity,
            'duration_ms': round(max(0, elapsed_s)*1000, 2)}
    with _lock:
        handler = _handlers.get(destination)
        if handler is None:
            path = Path(destination)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            handler = PrivateRotatingHandler(path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(message)s'))
            _handlers[destination] = handler
        handler.emit(logging.LogRecord('witness.audit', logging.INFO, '', 0, json.dumps(data, separators=(',', ':')), (), None))
