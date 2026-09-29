"""Authenticated TLS transfers, including old peers and interrupted parallel reads."""
from contextlib import contextmanager
import os
import json
import ssl
import threading
import time
from urllib.parse import parse_qs, urlsplit

from bittensor_wallet import Keypair
import pytest
import zstandard

from witness.benchmark.p2p import Auth, CHUNK, ModelClient, ModelServer, _ModelHandler, certificate
from witness.benchmark.submission import Submission, prepare_manifest, validate_manifest, verify_directory
from witness.benchmark.compression import (CompressionRequired, MAX_FRAME_OVERHEAD,
                                           decode_model_chunk, encode_model_chunk)
from witness.events import content_hash


@contextmanager
def peer(tmp_path, *, version='HTTP/1.1', before=None, chunks=4, legacy=False, noise=False):
    alice, bob = Keypair.create_from_uri('//Alice'), Keypair.create_from_uri('//Bob')
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'config.json').write_text('{"model_type":"qwen2_5_omni"}')
    # Transport fixture only. Different ranges make out-of-order writes visible.
    with (root / 'model.safetensors').open('wb') as stream:
        for i in range(chunks):
            stream.write(os.urandom(CHUNK) if noise else bytes([i + 1]) * CHUNK)
        stream.write(b'last partial range')
    manifest = prepare_manifest(root, 'qwen2.5-omni')
    cert, key, pin = certificate(tmp_path / 'tls')
    state = {'connections': 0, 'active': 0, 'peak': 0, 'ranges': [], 'nonces': [], 'headers': []}
    lock = threading.Lock()

    class Handler(_ModelHandler):
        protocol_version = version

        def send_header(self, keyword, value):
            with lock:
                state['headers'].append((self.path, keyword.lower(), value))
            super().send_header(keyword, value)

        def do_GET(self):
            if legacy:
                # Simulate a pre-compression server that ignores negotiation.
                del self.headers['Accept-Encoding']
            parsed = urlsplit(self.path)
            weight_range = parsed.path.endswith('/files/1')
            with lock:
                state['nonces'].append(self.headers.get('X-Bittensor-Nonce'))
                if weight_range:
                    state['active'] += 1
                    state['peak'] = max(state['peak'], state['active'])
                    state['ranges'].append(int(parse_qs(parsed.query)['offset'][0]))
            try:
                if before is None or not before(self, state):
                    super().do_GET()
            finally:
                if weight_range:
                    with lock:
                        state['active'] -= 1

    class Server(ModelServer):
        def process_request(self, request, address):
            state['connections'] += 1
            super().process_request(request, address)

    server = Server(('127.0.0.1', 0), model_root=root, manifest=manifest,
                    auth=Auth(bob.ss58_address, lambda: {'observed_unix': time.time(),
                              'permits': [alice.ss58_address],
                              'alpha_rao': {alice.ss58_address: 100_000 * 10**9}}),
                    certfile=cert, keyfile=key)
    server.RequestHandlerClass = Handler
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    submission = Submission(validate_manifest(manifest)['model_id'], pin)
    client = ModelClient(alice, bob.ss58_address, '127.0.0.1', server.server_port, submission,
                         allow_loopback=True)
    try:
        yield client, state, root, manifest
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('version', ['HTTP/1.0', 'HTTP/1.1'])
def test_parallel_resume_is_bounded_and_reuses_supported_connections(tmp_path, version):
    both_started = threading.Barrier(2)

    def overlap(handler, state):
        if handler.path.endswith(f'offset=137&length={CHUNK}') or handler.path.endswith(
                f'offset={137 + CHUNK}&length={CHUNK}'):
            both_started.wait(timeout=5)
            # Complete the later range first to exercise ordered persistence.
            if f'offset=137&' in handler.path:
                time.sleep(.05)

    with peer(tmp_path, version=version, before=overlap) as (client, state, source, manifest):
        root = tmp_path / 'cache' / client.submission.model_id
        root.mkdir(parents=True)
        partial = root / 'model.safetensors.part'
        partial.write_bytes((source / 'model.safetensors').read_bytes()[:137])
        downloaded, _ = client.download(tmp_path / 'cache')
        assert verify_directory(downloaded, manifest)['model_id'] == client.submission.model_id
        assert not partial.exists()
        assert state['peak'] == 2
        assert min(state['ranges']) == 137
        assert len(state['nonces']) == len(set(state['nonces']))
        compressed = [v for _, key, v in state['headers'] if key == 'content-encoding']
        assert compressed
        weight_bytes = sum(int(value) for path, key, value in state['headers']
                           if '/files/1?' in path and key == 'content-length')
        assert weight_bytes < (source / 'model.safetensors').stat().st_size // 10
        if version == 'HTTP/1.1':
            assert state['connections'] == 2
        else:
            assert state['connections'] == len(state['nonces'])


def test_truncated_parallel_response_preserves_contiguous_prefix_and_retries(tmp_path):
    broken = True

    def truncate(handler, state):
        if broken and f'/files/1?offset={CHUNK}&' in handler.path:
            handler.send_response(200)
            handler.send_header('Content-Length', str(CHUNK))
            handler.send_header('Content-Encoding', 'zstd')
            handler.end_headers()
            handler.wfile.write(b'short')
            handler.close_connection = True
            return True

    with peer(tmp_path, before=truncate) as (client, state, source, manifest):
        root = tmp_path / 'cache' / client.submission.model_id
        with pytest.raises(OSError, match='incomplete_model_response'):
            client.download(tmp_path / 'cache')
        partial = root / 'model.safetensors.part'
        assert partial.read_bytes() == (source / 'model.safetensors').read_bytes()[:CHUNK]
        assert not (root / 'witness-manifest.json').exists()
        broken = False
        state['ranges'].clear()
        downloaded, _ = client.download(tmp_path / 'cache')
        assert min(state['ranges']) == CHUNK
        verify_directory(downloaded, manifest)


def test_download_cancellation_keeps_prefix_and_closes_workers(tmp_path):
    with peer(tmp_path) as (client, state, source, manifest):
        root = tmp_path / 'cache' / client.submission.model_id
        partial = root / 'model.safetensors.part'
        cancelled = lambda: partial.exists() and partial.stat().st_size >= CHUNK
        with pytest.raises(InterruptedError):
            client.download(tmp_path / 'cache', cancelled=cancelled)
        assert partial.read_bytes() == (source / 'model.safetensors').read_bytes()[:CHUNK]
        assert not (root / 'witness-manifest.json').exists()
        assert not any(t.name.startswith('model-download') for t in threading.enumerate())
        downloaded, _ = client.download(tmp_path / 'cache')
        verify_directory(downloaded, manifest)


def test_corrupt_completed_weights_never_become_verified_cache(tmp_path):
    def corrupt(handler, state):
        if '/files/1?offset=0&' in handler.path:
            handler.send_response(200)
            body = zstandard.ZstdCompressor().compress(b'wrong bytes'.ljust(CHUNK, b'!'))
            handler.send_header('Content-Length', str(len(body)))
            handler.send_header('Content-Encoding', 'zstd')
            handler.end_headers()
            handler.wfile.write(body)
            return True

    with peer(tmp_path, before=corrupt) as (client, state, source, manifest):
        with pytest.raises(ValueError, match='model_file_hash_mismatch'):
            client.download(tmp_path / 'cache')
        root = tmp_path / 'cache' / client.submission.model_id
        assert not (root / 'model.safetensors.part').exists()
        assert not (root / 'model.safetensors').exists()
        assert not (root / 'witness-manifest.json').exists()


def test_idle_connection_closed_by_peer_is_repinned_and_reauthenticated(tmp_path):
    def close_after_manifest(handler, state):
        if handler.path.endswith('/manifest'):
            _ModelHandler.do_GET(handler)
            # A keep-alive response followed by an idle disconnect, without a
            # Connection: close header, must not poison the next request.
            handler.close_connection = True
            return True

    with peer(tmp_path, before=close_after_manifest) as (client, state, source, manifest):
        downloaded, _ = client.download(tmp_path / 'cache')
        verify_directory(downloaded, manifest)
        assert 3 <= state['connections'] <= 4
        assert len(state['nonces']) == len(set(state['nonces']))


def test_reconnect_checks_changed_certificate_before_sending_auth(tmp_path):
    other_cert, other_key, _ = certificate(tmp_path / 'other-tls')
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(other_cert, other_key)

    def replace_certificate(handler, state):
        if handler.path.endswith('/manifest'):
            _ModelHandler.do_GET(handler)
            handler.server.socket.context = context
            handler.close_connection = True
            return True

    with peer(tmp_path, before=replace_certificate) as (client, state, source, manifest):
        with pytest.raises(ValueError, match='miner_tls_pin_mismatch'):
            client.download(tmp_path / 'cache')
        assert len(state['nonces']) == 1  # Only the original, correctly pinned manifest GET.


@pytest.mark.parametrize('chunked', [False, True])
def test_unbounded_response_is_not_accepted_or_reused(tmp_path, chunked):
    def unbounded(handler, state):
        handler.send_response(200)
        handler.send_header('Content-Length', '1' if chunked else str(2**40))
        if chunked:
            handler.send_header('Transfer-Encoding', 'chunked')
        handler.end_headers()
        handler.close_connection = True
        return True

    with peer(tmp_path, before=unbounded) as (client, state, source, manifest):
        with pytest.raises(ValueError, match='unbounded_model_response'):
            client.download(tmp_path / 'cache')
        assert state['connections'] == 1
        assert not any(t.name.startswith('model-download') for t in threading.enumerate())


def test_attempt_deadline_unwinds_both_download_workers(tmp_path):
    released = threading.Event()

    def stall_weights(handler, state):
        if '/files/1?' in handler.path:
            handler.send_response(200)
            handler.send_header('Content-Length', str(CHUNK))
            handler.send_header('Content-Encoding', 'zstd')
            handler.end_headers()
            released.wait(timeout=3)
            handler.close_connection = True
            return True

    with peer(tmp_path, before=stall_weights) as (client, state, source, manifest):
        start = time.monotonic()
        client.remaining_s = lambda: start + .8 - time.monotonic()
        try:
            with pytest.raises((TimeoutError, InterruptedError)):
                client.download(tmp_path / 'cache')
            assert time.monotonic() - start < 2
            assert state['peak'] == 2
            assert not any(t.name.startswith('model-download') for t in threading.enumerate())
            assert not (tmp_path / 'cache' / client.submission.model_id / 'witness-manifest.json').exists()
        finally:
            released.set()


@pytest.mark.parametrize('version', ['HTTP/1.0', 'HTTP/1.1'])
def test_deadline_bounds_stalled_body_on_both_peer_versions(tmp_path, version):
    released = threading.Event()

    def stall(handler, state):
        handler.send_response(200)
        handler.send_header('Content-Length', '10')
        handler.end_headers()
        released.wait(timeout=2)
        handler.close_connection = True
        return True

    with peer(tmp_path, version=version, before=stall) as (client, state, source, manifest):
        client.remaining_s = lambda: .15
        start = time.monotonic()
        try:
            with pytest.raises((TimeoutError, InterruptedError)):
                client.manifest()
            assert time.monotonic() - start < 1.5
        finally:
            released.set()


def test_old_client_gets_original_bytes_from_new_server(tmp_path):
    with peer(tmp_path, chunks=1) as (client, state, source, manifest):
        target = f'/v1/models/{client.submission.model_id}/files/1?offset=0&length={CHUNK}'
        assert client.get(target, CHUNK) == (source / 'model.safetensors').read_bytes()[:CHUNK]
        assert not any(key == 'content-encoding' for _, key, _ in state['headers'])


def test_incompressible_ranges_still_use_bounded_zstd(tmp_path):
    with peer(tmp_path, chunks=1, noise=True) as (client, state, source, manifest):
        downloaded, _ = client.download(tmp_path / 'cache')
        verify_directory(downloaded, manifest)
        assert any(key == 'content-encoding' and value == 'zstd' for _, key, value in state['headers'])


@pytest.mark.parametrize('case', ['oversized', 'unknown_size', 'corrupt', 'truncated',
                                  'trailing', 'concatenated', 'unsupported'])
def test_invalid_compressed_range_never_enters_cache(tmp_path, case):
    raw = b'x' * CHUNK
    encoder = zstandard.ZstdCompressor(level=1, write_checksum=True)
    encoded = encoder.compress(raw)
    if case == 'oversized':
        encoded = encoder.compress(raw * 2)
    elif case == 'unknown_size':
        encoded = zstandard.ZstdCompressor(write_content_size=False).compress(raw)
    elif case == 'corrupt':
        encoded = encoded[:-1] + bytes([encoded[-1] ^ 1])
    elif case == 'truncated':
        encoded = encoded[:-4]
    elif case == 'trailing':
        encoded += b'extra'
    elif case == 'concatenated':
        encoded += encoder.compress(b'extra')

    def invalid(handler, state):
        if '/files/1?offset=0&' in handler.path:
            handler.send_response(200)
            handler.send_header('Content-Length', str(len(encoded)))
            handler.send_header('Content-Encoding', 'gzip' if case == 'unsupported' else 'zstd')
            handler.end_headers()
            handler.wfile.write(encoded)
            return True

    with peer(tmp_path, chunks=1, before=invalid) as (client, state, source, manifest):
        error = 'zstandard_required' if case == 'unsupported' else 'invalid_compressed_model_chunk'
        with pytest.raises(CompressionRequired, match=error):
            client.download(tmp_path / 'cache')
        root = tmp_path / 'cache' / client.submission.model_id
        assert (root / 'model.safetensors.part').stat().st_size == 0
        assert not (root / 'model.safetensors').exists()
        assert not (root / 'witness-manifest.json').exists()


def test_valid_compressed_frame_still_requires_committed_file_hash(tmp_path):
    def different_weights(handler, state):
        if '/files/1?offset=0&' in handler.path:
            encoded = zstandard.ZstdCompressor().compress(b'wrong'.ljust(CHUNK, b'!'))
            handler.send_response(200)
            handler.send_header('Content-Length', str(len(encoded)))
            handler.send_header('Content-Encoding', 'zstd')
            handler.end_headers()
            handler.wfile.write(encoded)
            return True

    with peer(tmp_path, chunks=1, before=different_weights) as (client, state, source, manifest):
        with pytest.raises(ValueError, match='model_file_hash_mismatch'):
            client.download(tmp_path / 'cache')
        root = tmp_path / 'cache' / client.submission.model_id
        assert not (root / 'model.safetensors.part').exists()
        assert not (root / 'model.safetensors').exists()
        assert not (root / 'witness-manifest.json').exists()


@pytest.mark.parametrize('accept,compressed', [('zstd', True), ('gzip, zstd;q=0.5', True),
                                              ('ZSTD; q=1', True), ('zstd;q=0', False),
                                              ('zstd;q=invalid', False), ('zstd;q=nan', False),
                                              ('gzip', False), ('', False)])
def test_miner_compression_helper_negotiates_and_roundtrips(accept, compressed):
    raw = b'original weights ' * 1024
    wire, encoding = encode_model_chunk(raw, accept_encoding=accept)
    assert (encoding == 'zstd') is compressed
    assert (decode_model_chunk(wire, len(raw)) if encoding else wire) == raw
    assert len(wire) < len(raw) if compressed else wire is raw


@pytest.mark.parametrize('size', [0, CHUNK + 1])
def test_miner_helper_rejects_out_of_range_before_compressing(size):
    with pytest.raises(ValueError, match='invalid_model_chunk_size'):
        encode_model_chunk(b'x' * size, accept_encoding='zstd')


@pytest.mark.parametrize('cancel', [False, True])
def test_absolute_deadline_and_cancellation_stop_dripping_headers(tmp_path, cancel):
    released = threading.Event()
    started = threading.Event()
    cancelled = threading.Event()

    def drip(handler, state):
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nX-Drip: ')
        started.set()
        try:
            while not released.wait(.03):
                handler.wfile.write(b'x')  # Always faster than the socket timeout.
        except OSError:
            pass
        handler.close_connection = True
        return True

    with peer(tmp_path, before=drip) as (client, state, source, manifest):
        client.cancelled = cancelled.is_set
        client.remaining_s = lambda: 30. if cancel else .25
        timer = threading.Timer(.25, cancelled.set)
        if cancel:
            timer.start()
        start = time.monotonic()
        try:
            with pytest.raises(InterruptedError, match='model_download_cancelled'):
                client.manifest()
            assert started.is_set() and time.monotonic() - start < 1.5
            assert not any(t.name == 'model-request-deadline' for t in threading.enumerate())
        finally:
            released.set()
            if cancel:
                timer.cancel()
                timer.join()


def test_oversized_uncompressed_manifest_is_rejected_before_any_file_request(tmp_path, monkeypatch):
    manifest = {'schema_version': 'witness-model-2', 'arch': 'qwen2.5-omni', 'files': [
        {'name': 'config.json', 'size': 30, 'sha256': 'a' * 64},
        {'name': 'model.safetensors', 'size': 24_000_000_000, 'sha256': 'b' * 64}]}
    client = ModelClient(None, 'miner', '127.0.0.1', 1234,
                         Submission(content_hash(manifest), 'c' * 64), allow_loopback=True)
    requested = []
    def get(target, limit, **kwargs):
        requested.append(target)
        assert target.endswith('/manifest')
        return json.dumps(manifest).encode()
    monkeypatch.setattr(client, 'get', get)
    with pytest.raises(ValueError, match='weights_too_large'):
        client.download(tmp_path / 'cache')
    assert len(requested) == 1
    assert not (tmp_path / 'cache').exists()


def test_total_download_budget_preserves_prefix_and_blocks_restart(tmp_path, monkeypatch):
    from witness.benchmark import download_budget
    from witness.benchmark.download_budget import DownloadBudgetExceeded
    monkeypatch.setattr(download_budget, 'MODEL_DOWNLOAD_BUDGET_S', .6)
    release = threading.Event()
    def stall(handler, state):
        if f'/files/1?offset={CHUNK}&' in handler.path:
            handler.send_response(200)
            handler.send_header('Content-Length', str(CHUNK))
            handler.send_header('Content-Encoding', 'zstd')
            handler.end_headers()
            release.wait(timeout=3)
            handler.close_connection = True
            return True
    with peer(tmp_path, chunks=2, before=stall) as (client, state, source, manifest):
        try:
            with pytest.raises(DownloadBudgetExceeded):
                client.download(tmp_path / 'cache')
            root = tmp_path / 'cache' / client.submission.model_id
            assert (root / 'model.safetensors.part').read_bytes() == (source / 'model.safetensors').read_bytes()[:CHUNK]
            assert not (root / 'witness-manifest.json').exists()
            requests = len(state['nonces'])
            with pytest.raises(DownloadBudgetExceeded):
                client.download(tmp_path / 'cache')
            assert len(state['nonces']) == requests
            assert not any(t.name.startswith(('model-download', 'model-request-deadline')) for t in threading.enumerate())
        finally:
            release.set()


def test_transport_failure_diagnostic_contains_only_safe_status(tmp_path):
    def unavailable(handler, state):
        handler.send_error(429)
        return True
    with peer(tmp_path, before=unavailable) as (client, state, source, manifest):
        with pytest.raises(OSError, match='model_server_status_429'):
            client.download(tmp_path / 'cache')
        path = tmp_path / 'download-budgets' / (client.submission.model_id + '.error.json')
        error = json.loads(path.read_text())
        assert set(error) == {'unix', 'type', 'http_status'}
        assert error['type'] == 'OSError' and error['http_status'] == 429


@pytest.mark.parametrize('legacy', [False, True])
def test_compression_probe_is_tiny_and_rejects_raw_before_download(tmp_path, legacy):
    with peer(tmp_path, chunks=1, legacy=legacy) as (client, state, source, manifest):
        if legacy:
            with pytest.raises(CompressionRequired, match='zstandard_required'):
                client.require_compression()
            with pytest.raises(CompressionRequired):
                client.download(tmp_path / 'cache')
            assert not (tmp_path / 'cache' / client.submission.model_id / 'witness-manifest.json').exists()
        else:
            client.require_compression()
        assert not state['ranges']  # Only a one-byte config range was requested.


@pytest.mark.parametrize('raw', [b'x', os.urandom(CHUNK)])
def test_required_encoding_never_falls_back_for_tiny_or_incompressible_ranges(raw):
    wire, encoding = encode_model_chunk(raw, accept_encoding='zstd, identity;q=0')
    assert encoding == 'zstd' and len(wire) <= len(raw) + MAX_FRAME_OVERHEAD
    assert decode_model_chunk(wire, len(raw)) == raw


@pytest.mark.parametrize('length', ['-1', 'invalid', str(CHUNK + MAX_FRAME_OVERHEAD + 1)])
def test_invalid_compressed_length_is_transport_exclusion_not_model_rejection(tmp_path, length):
    def invalid(handler, state):
        handler.send_response(200)
        handler.send_header('Content-Encoding', 'zstd')
        handler.send_header('Content-Length', length)
        handler.end_headers()
        handler.close_connection = True
        return True
    with peer(tmp_path, chunks=1, before=invalid) as (client, state, source, manifest):
        with pytest.raises(CompressionRequired):
            client.require_compression()
