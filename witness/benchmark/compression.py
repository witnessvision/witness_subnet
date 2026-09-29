"""Lossless model-range transport helpers for miner HTTP integrations.

Use these after authenticating the request and validating its committed file
range. Offsets, sizes and SHA-256 hashes always refer to the original bytes.
"""
from __future__ import annotations

import zstandard

MAX_CHUNK_BYTES = 4 * 1024 * 1024
MAX_FRAME_OVERHEAD = 64 * 1024


class CompressionRequired(OSError):
    """Retryable transport noncompliance; never a scored model rejection."""


def encode_model_chunk(raw: bytes, *, accept_encoding: str = '') -> tuple[bytes, str | None]:
    """Return response bytes and Content-Encoding (None means omit the header).

    With identity;q=0 every range is encoded, including tiny/incompressible
    ranges. Optional negotiation retains identity when it saves bandwidth.
    """
    if not 0 < len(raw) <= MAX_CHUNK_BYTES:
        raise ValueError('invalid_model_chunk_size')
    qualities = {}
    for item in accept_encoding.lower().split(','):
        name, *parameters = (part.strip() for part in item.split(';'))
        quality = 1.
        for parameter in parameters:
            key, separator, value = parameter.partition('=')
            if key.strip() == 'q':
                try:
                    quality = float(value) if separator else 0.
                except ValueError:
                    quality = 0.
        qualities[name] = quality if 0 <= quality <= 1 else 0.
    accepted = qualities.get('zstd', 0.) > 0
    required = qualities.get('identity', 1.) == 0
    if required and not accepted:
        raise ValueError('no_acceptable_model_encoding')
    if accepted and (required or len(raw) >= 1024):
        encoded = zstandard.ZstdCompressor(level=1, write_checksum=True).compress(raw)
        if required or len(encoded) + 64 < len(raw):
            return encoded, 'zstd'
    return raw, None


def decode_model_chunk(body: bytes, expected_size: int) -> bytes:
    """Decode one bounded zstd range; reject bombs, dictionaries and extra frames."""
    try:
        frame = zstandard.get_frame_parameters(body)
        if (not 0 < expected_size <= MAX_CHUNK_BYTES or len(body) > expected_size + MAX_FRAME_OVERHEAD
                or frame.content_size != expected_size or frame.window_size > MAX_CHUNK_BYTES
                or frame.dict_id):
            raise ValueError('invalid_compressed_model_chunk')
        decoded = zstandard.ZstdDecompressor(max_window_size=MAX_CHUNK_BYTES // 1024).decompress(
            body, max_output_size=expected_size, allow_extra_data=False)
    except zstandard.ZstdError as error:
        raise ValueError('invalid_compressed_model_chunk') from error
    if len(decoded) != expected_size:
        raise ValueError('invalid_compressed_model_chunk')
    return decoded
