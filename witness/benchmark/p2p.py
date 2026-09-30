"""Pinned TLS and btauth/1 for streaming model files. No model code is served."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
from http.client import HTTPException, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
from queue import LifoQueue
import re
import shutil
import socket
import ssl
import threading
import time
from urllib.parse import parse_qs, urlsplit

from witness.events import content_hash
from witness.storage import sha256_file, write_private
from .compression import (MAX_CHUNK_BYTES, MAX_FRAME_OVERHEAD, CompressionRequired,
                          decode_model_chunk, encode_model_chunk)
from .download_budget import DownloadBudget, DownloadBudgetExceeded
from .submission import (MAX_MANIFEST_BYTES, Submission, check_config, validate_manifest,
                         verify_directory)

MIN_STAKE_ALPHA = 100_000
CHUNK = MAX_CHUNK_BYTES
DOWNLOAD_CONNECTIONS = 2  # The existing server's per-validator request limit.


def auth_payload(method, target, body, nonce, sender, receiver, scheme='sr25519'):
    return '\n'.join(('btauth/1', scheme, method.upper(), target, hashlib.sha256(body).hexdigest(),
                      str(nonce), sender, receiver)).encode()


def sign(keypair, method: str, target: str, receiver: str, body=b'', nonce=None) -> dict:
    nonce = time.time_ns() if nonce is None else nonce
    scheme = {0: 'ed25519', 1: 'sr25519'}.get(keypair.crypto_type)
    if scheme is None:
        raise ValueError('unsupported_signature_scheme')
    sender = keypair.ss58_address
    headers = {'X-Bittensor-Version': '1', 'X-Bittensor-Hotkey': sender, 'X-Bittensor-Receiver': receiver,
               'X-Bittensor-Nonce': str(nonce), 'X-Bittensor-Signature': '0x' + bytes(keypair.sign(
                   auth_payload(method, target, body, nonce, sender, receiver, scheme))).hex()}
    if scheme != 'sr25519':
        headers['X-Bittensor-Crypto'] = scheme
    return headers


class Auth:
    def __init__(self, receiver, snapshot, *, min_stake_alpha=MIN_STAKE_ALPHA, clock=time.time_ns):
        self.receiver, self.snapshot, self.clock = receiver, snapshot, clock
        if type(min_stake_alpha) is not int or min_stake_alpha < MIN_STAKE_ALPHA:
            raise ValueError('validator_stake_floor_required')
        self.minimum = min_stake_alpha * 10**9
        self.seen, self.lock = {}, threading.Lock()

    def verify(self, headers, method, target, body=b''):
        from bittensor_wallet import Keypair
        h = {k.lower(): v for k, v in headers.items()}
        try:
            sender, nonce = h['x-bittensor-hotkey'], h['x-bittensor-nonce']
            scheme = h.get('x-bittensor-crypto', 'sr25519')
            if (h['x-bittensor-version'] != '1' or h['x-bittensor-receiver'] != self.receiver
                    or scheme not in ('sr25519', 'ed25519') or not re.fullmatch('[0-9]{1,20}', nonce)
                    or not re.fullmatch('0x[0-9a-f]{128}', h['x-bittensor-signature'])):
                raise ValueError('invalid_auth')
            now, stamp = self.clock(), int(nonce)
            if not now - 10_000_000_000 <= stamp <= now + 2_000_000_000:
                raise ValueError('stale_request')
            key = Keypair(ss58_address=sender, crypto_type=1 if scheme == 'sr25519' else 0)
            if not key.verify(auth_payload(method, target, body, nonce, sender, self.receiver, scheme),
                              bytes.fromhex(h['x-bittensor-signature'][2:])):
                raise ValueError('bad_signature')
        except (KeyError, TypeError, ValueError) as error:
            raise PermissionError('request_authentication_failed') from error
        snapshot = self.snapshot()
        if time.time() - snapshot.get('observed_unix', 0) > 120:
            raise PermissionError('chain_state_stale')
        if sender not in snapshot.get('permits', []) or snapshot.get('alpha_rao', {}).get(sender, 0) < self.minimum:
            raise PermissionError('validator_access_denied')
        with self.lock:
            self.seen = {k: expiry for k, expiry in self.seen.items() if expiry >= now}
            identity = (sender, nonce)
            if identity in self.seen or len(self.seen) >= 10000:
                raise PermissionError('replayed_or_rate_limited')
            self.seen[identity] = stamp + 10_000_000_000
        return sender


def certificate(root: Path) -> tuple[Path, Path, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    certfile, keyfile = root / 'tls.crt', root / 'tls.key'
    if not certfile.exists() and not keyfile.exists():
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Witness model server')])
        now = datetime.now(timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
                .not_valid_after(now + timedelta(days=365)).sign(key, hashes.SHA256()))
        for path, data in ((keyfile, key.private_bytes(serialization.Encoding.PEM,
                                                      serialization.PrivateFormat.PKCS8,
                                                      serialization.NoEncryption())),
                           (certfile, cert.public_bytes(serialization.Encoding.PEM))):
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
    if keyfile.stat().st_mode & 0o077:
        raise ValueError('unsafe_tls_key_permissions')
    der = ssl.PEM_cert_to_DER_cert(certfile.read_text())
    return certfile, keyfile, hashlib.sha256(der).hexdigest()


def endpoint(host: str, port: int, *, allow_loopback=False):
    ip = ipaddress.ip_address(host)  # no DNS rebinding or miner-controlled redirects
    if not (ip.is_global or (allow_loopback and ip.is_loopback)) or not 1 <= port <= 65535:
        raise ValueError('unsafe_model_endpoint')
    return str(ip), port


class ModelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, *, model_root, manifest, auth, certfile, keyfile, max_downloads=4):
        self.root, self.manifest, self.auth = Path(model_root).resolve(), manifest, auth
        self.info = verify_directory(self.root, manifest)
        self.slots = threading.BoundedSemaphore(max_downloads)
        self.connections = threading.BoundedSemaphore(max_downloads + 8)
        self.active_senders, self.sender_lock = {}, threading.Lock()
        super().__init__(address, _ModelHandler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile, keyfile)
        self.socket = ctx.wrap_socket(self.socket, server_side=True, do_handshake_on_connect=False)

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, address)

    def process_request_thread(self, request, address):
        try:
            request.settimeout(15)
            super().process_request_thread(request, address)
        finally:
            self.connections.release()


class _ModelHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def do_GET(self):
        server = self.server
        sender = None
        if not server.slots.acquire(blocking=False):
            self.send_error(429)
            return
        try:
            if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length', '0') != '0':
                self.send_error(400)
                return
            identity = server.auth.verify(self.headers, 'GET', self.path)
            with server.sender_lock:
                if server.active_senders.get(identity, 0) >= 2:
                    self.send_error(429)
                    return
                sender = identity
                server.active_senders[sender] = server.active_senders.get(sender, 0) + 1
            parsed = urlsplit(self.path)
            prefix = f'/v1/models/{server.info["model_id"]}'
            if parsed.path == prefix + '/manifest' and not parsed.query:
                body = json.dumps(server.manifest, sort_keys=True, separators=(',', ':')).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            match = re.fullmatch(re.escape(prefix) + r'/files/([0-9]+)', parsed.path)
            query = parse_qs(parsed.query, strict_parsing=True)
            if match is None or set(query) != {'offset', 'length'} or any(len(v) != 1 for v in query.values()):
                self.send_error(404)
                return
            index = int(match[1])
            row = server.manifest['files'][index]
            offset, length = int(query['offset'][0]), int(query['length'][0])
            if not (0 <= offset < row['size'] and 1 <= length <= CHUNK and offset + length <= row['size']):
                self.send_error(416)
                return
            path = server.root / row['name']
            if any(p.is_symlink() for p in (path, *path.parents)):
                self.send_error(409)
                return
            with path.open('rb') as stream:
                if os.fstat(stream.fileno()).st_size != row['size']:
                    self.send_error(409)
                    return
                stream.seek(offset)
                body = stream.read(length)
            body, encoding = encode_model_chunk(body, accept_encoding=self.headers.get('Accept-Encoding', ''))
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Vary', 'Accept-Encoding')
            if encoding:
                self.send_header('Content-Encoding', encoding)
            self.end_headers()
            self.wfile.write(body)
        except PermissionError:
            self.send_error(403)
        except (ValueError, IndexError, KeyError):
            self.send_error(400)
        except (OSError, TimeoutError):
            self.close_connection = True
        finally:
            if sender is not None:
                with server.sender_lock:
                    server.active_senders[sender] -= 1
                    if not server.active_senders[sender]:
                        del server.active_senders[sender]
            server.slots.release()


class ModelClient:
    def __init__(self, keypair, receiver, host, port, submission: Submission, *, allow_loopback=False):
        self.address = endpoint(host, port, allow_loopback=allow_loopback)
        self.keypair, self.receiver, self.submission = keypair, receiver, submission
        self.cancelled = lambda: False
        self.remaining_s = lambda: 30.
        self._connections = None
        self._last_error = None

    def get(self, target, limit, *, compressed=False):
        remaining = min(30., self.remaining_s())
        if self.cancelled() or remaining <= 0:
            raise InterruptedError('model_download_cancelled')
        deadline = time.monotonic() + remaining
        finished, expired = threading.Event(), threading.Event()
        active_socket = [None]

        def guard():
            # Socket timeouts reset on incoming bytes. A peer dripping HTTP
            # headers could otherwise hold getresponse() indefinitely.
            while not finished.wait(min(.1, max(0., deadline - time.monotonic()))):
                if self.cancelled() or time.monotonic() >= deadline:
                    expired.set()
                    sock = active_socket[0]
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return

        def timeout():
            left = deadline - time.monotonic()
            if expired.is_set() or self.cancelled() or left <= 0:
                raise InterruptedError('model_download_cancelled')
            return min(5., left)

        pool = self._connections
        connection = pool.get() if pool is not None else None
        reusable = False
        response = None
        watchdog = threading.Thread(target=guard, name='model-request-deadline', daemon=True)
        watchdog.start()
        try:
            for attempt in range(2):
                reused = connection is not None
                if connection is None:
                    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE  # Pin is bound by the on-chain commitment.
                    connection = HTTPSConnection(*self.address, context=ctx, timeout=timeout())
                    connection.connect()
                    if hashlib.sha256(connection.sock.getpeercert(binary_form=True)).hexdigest() != self.submission.certificate:
                        raise ValueError('miner_tls_pin_mismatch')
                try:
                    active_socket[0] = connection.sock
                    connection.sock.settimeout(timeout())
                    headers = sign(self.keypair, 'GET', target, self.receiver)
                    if compressed:
                        headers['Accept-Encoding'] = 'zstd, identity;q=0'
                    connection.request('GET', target, headers=headers)
                    connection.sock.settimeout(timeout())
                    sock = connection.sock
                    response = connection.getresponse()
                    break
                except (OSError, HTTPException):
                    # A peer may close an idle keep-alive socket. Retry this GET
                    # once on a freshly pinned connection, with a fresh nonce.
                    if not reused or attempt:
                        raise
                    connection.close()
                    connection = None
            if compressed and response.status == 406:
                raise CompressionRequired('zstandard_required')
            if response.status != 200:
                raise OSError(f'model_server_status_{response.status}')
            encoding = response.getheader('Content-Encoding', 'identity').strip().lower()
            if compressed and encoding != 'zstd':
                raise CompressionRequired('zstandard_required')
            if not compressed and encoding != 'identity':
                raise ValueError('unsupported_model_encoding')
            wire_limit = limit + MAX_FRAME_OVERHEAD if compressed else limit
            try:
                length = int(response.getheader('Content-Length', '-1'))
            except ValueError as error:
                if compressed:
                    raise CompressionRequired('invalid_compressed_model_length') from error
                raise
            if response.getheader('Transfer-Encoding') or length not in range(wire_limit + 1):
                if compressed:
                    raise CompressionRequired('unbounded_compressed_model_response')
                raise ValueError('unbounded_model_response')
            # Keep the socket reference: HTTP/1.0 detaches it from connection.
            body = bytearray()
            while len(body) < length:
                wait = timeout()
                sock.settimeout(wait)
                chunk = response.read1(min(65536, length - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > wire_limit:
                    break
            if len(body) > wire_limit:
                raise ValueError('model_response_too_large')
            if len(body) != length:
                raise OSError('incomplete_model_response')
            timeout()  # Include decoding in the request's cancellation/deadline checks.
            try:
                body = decode_model_chunk(body, limit) if compressed else bytes(body)
            except ValueError as error:
                raise CompressionRequired('invalid_compressed_model_chunk') from error
            timeout()
            reusable = not response.will_close and connection.sock is not None
            return body
        except Exception as error:
            if self._last_error is None:
                detail = {'type': type(error).__name__}
                status = re.fullmatch(r'model_server_status_([0-9]{3})', str(error))
                if status:
                    detail['http_status'] = int(status[1])
                elif str(error) in ('incomplete_model_response', 'incomplete_model_chunk'):
                    detail['code'] = str(error)
                self._last_error = detail  # Never store raw exceptions, URLs or authentication.
            if expired.is_set():
                raise InterruptedError('model_download_cancelled') from error
            raise
        finally:
            finished.set()
            watchdog.join()  # Never let a late watchdog close a pooled/reassigned socket.
            if response is not None:
                response.close()
            if connection is not None and (pool is None or not reusable or expired.is_set()):
                connection.close()
                connection = None
            if pool is not None:
                pool.put(connection)

    def require_compression(self):
        """Small authenticated probe, also required for cached challengers."""
        target = f'/v1/models/{self.submission.model_id}/files/0?offset=0&length=1'
        self.get(target, 1, compressed=True)

    def manifest(self):
        value = json.loads(self.get(f'/v1/models/{self.submission.model_id}/manifest', MAX_MANIFEST_BYTES))
        validate_manifest(value, self.submission.model_id)
        return value

    def download(self, cache: Path, *, cancelled=lambda: False, accepted_architectures=None,
                 cache_key=None) -> tuple[Path, dict]:
        """Two bounded, reusable connections; only contiguous bytes reach .part."""
        if self._connections is not None:
            raise RuntimeError('model_download_already_running')
        self._last_error = None
        with DownloadBudget(cache.parent / 'download-budgets', cache_key or self.submission.model_id) as budget:
            remaining = self.remaining_s
            self.remaining_s = lambda: min(remaining(), budget.remaining_s())
            try:
                return self._download_turn(cache, cancelled, accepted_architectures, cache_key)
            except Exception as error:
                try:
                    write_private(budget.path.with_suffix('.error.json'),
                                  {'unix': time.time(), **(self._last_error or {'type': type(error).__name__})})
                except OSError:
                    pass  # Diagnostics must not replace the original acquisition failure.
                if budget.exhausted():
                    raise DownloadBudgetExceeded('model_download_time_budget_exhausted') from error
                raise
            finally:
                self.remaining_s = remaining

    def _download_turn(self, cache, cancelled, accepted_architectures, cache_key):
        original_cancelled = self.cancelled
        aborted = threading.Event()
        self.cancelled = lambda: aborted.is_set() or original_cancelled() or cancelled()
        self._connections = LifoQueue(DOWNLOAD_CONNECTIONS)
        for _ in range(DOWNLOAD_CONNECTIONS):
            self._connections.put(None)
        workers = ThreadPoolExecutor(max_workers=DOWNLOAD_CONNECTIONS, thread_name_prefix='model-download')
        try:
            return self._download(cache, workers, accepted_architectures, cache_key)
        finally:
            aborted.set()
            workers.shutdown(wait=True, cancel_futures=True)
            while not self._connections.empty():
                connection = self._connections.get_nowait()
                if connection is not None:
                    connection.close()
            self._connections = None
            self.cancelled = original_cancelled

    def _download(self, cache, workers, accepted_architectures, cache_key):
        manifest = self.manifest()
        if accepted_architectures is not None and manifest['arch'] not in accepted_architectures:
            raise NotImplementedError('architecture_not_enabled')
        info = validate_manifest(manifest, self.submission.model_id)
        cache_key = cache_key or self.submission.model_id
        if not re.fullmatch('[0-9a-f]{64}', cache_key):
            raise ValueError('invalid_cache_key')
        root = cache / cache_key
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError('model_cache_symlink')
        for index, row in enumerate(manifest['files']):
            target = root / row['name']
            if target.is_file() and not target.is_symlink() and target.stat().st_size == row['size']:
                if sha256_file(target) == row['sha256']:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            part = target.with_name(target.name + '.part')
            if any(p.is_symlink() for p in (part, target, *target.parents)):
                raise ValueError('model_cache_symlink')
            offset = part.stat().st_size if part.exists() else 0
            if offset > row['size']:
                part.unlink()
                offset = 0
            if shutil.disk_usage(root).free < row['size'] - offset + 1024**3:
                raise OSError('insufficient_model_cache_space')
            with part.open('ab') as stream:
                os.chmod(part, 0o600)
                pending = deque()
                requested = offset
                while offset < row['size']:
                    if self.cancelled():
                        raise InterruptedError('window_closed')
                    while len(pending) < DOWNLOAD_CONNECTIONS and requested < row['size']:
                        length = min(CHUNK, row['size'] - requested)
                        target_url = f'/v1/models/{self.submission.model_id}/files/{index}?offset={requested}&length={length}'
                        pending.append((length, workers.submit(self.get, target_url, length, compressed=True)))
                        requested += length
                    length, future = pending.popleft()
                    body = future.result()
                    if len(body) != length:
                        raise OSError('incomplete_model_chunk')
                    stream.write(body)
                    offset += length
            if sha256_file(part) != row['sha256']:
                part.unlink()
                raise ValueError('model_file_hash_mismatch')
            part.replace(target)
        verify_directory(root, manifest)
        write_private(root / 'witness-manifest.json', manifest)
        return root, {**info, 'arch': manifest['arch'], 'manifest': manifest}
