"""Pinned TLS and btauth/1 for streaming model files. No model code is served."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from http.client import HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import threading
import time
from urllib.parse import parse_qs, urlsplit

from witness.events import content_hash
from witness.storage import sha256_file, write_private
from .submission import (MAX_MANIFEST_BYTES, Submission, check_config, validate_manifest,
                         verify_directory)

MIN_STAKE_ALPHA = 100_000
CHUNK = 4 * 1024 * 1024


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
        self.minimum = int(min_stake_alpha) * 10**9
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
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'application/octet-stream')
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

    def get(self, target, limit):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # certificate hash is bound by the miner's on-chain commitment
        connection = HTTPSConnection(*self.address, context=ctx, timeout=30)
        try:
            connection.connect()
            if hashlib.sha256(connection.sock.getpeercert(binary_form=True)).hexdigest() != self.submission.certificate:
                raise ValueError('miner_tls_pin_mismatch')
            connection.request('GET', target, headers=sign(self.keypair, 'GET', target, self.receiver))
            response = connection.getresponse()
            if response.status != 200:
                raise OSError(f'model_server_status_{response.status}')
            if int(response.getheader('Content-Length', '-1')) not in range(limit + 1):
                raise ValueError('unbounded_model_response')
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError('model_response_too_large')
            return body
        finally:
            connection.close()

    def manifest(self):
        value = json.loads(self.get(f'/v1/models/{self.submission.model_id}/manifest', MAX_MANIFEST_BYTES))
        validate_manifest(value, self.submission.model_id)
        return value

    def download(self, cache: Path, *, cancelled=lambda: False, accepted_architectures=None,
                 cache_key=None) -> tuple[Path, dict]:
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
                while offset < row['size']:
                    if cancelled():
                        raise InterruptedError('window_closed')
                    length = min(CHUNK, row['size'] - offset)
                    body = self.get(f'/v1/models/{self.submission.model_id}/files/{index}?offset={offset}&length={length}', length)
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
