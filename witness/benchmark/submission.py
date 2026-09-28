"""Immutable P2P submissions. Only validator-owned loaders execute code."""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal

from witness.events import content_hash
from witness.storage import sha256_file
from .protocol import MAX_COMMITMENT_BYTES, decode_bytes, encode_bytes

SUBMISSION_PREFIX = 'wm2'
SS58 = re.compile(r'^[1-9A-HJ-NP-Za-km-z]{46,48}$')
ARCHITECTURES = {'salmonn2-pro': {'model_type': 'qwen3_vl', 'max_bytes': 24 * 10**9},
                 'qwen2.5-omni': {'model_type': 'qwen2_5_omni', 'max_bytes': 24 * 10**9},
                 'qwen3-omni': {'model_type': 'qwen3_omni_moe', 'max_bytes': 44 * 10**9}}
Architecture = Literal['salmonn2-pro', 'qwen2.5-omni', 'qwen3-omni']
ALLOWED_FILES = ('*.safetensors', '*.json', '*.jinja', 'merges.txt', 'vocab.txt', 'tokenizer.model')
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILES = 1024


def challenge_id(hotkey: str, manifest_id: str) -> str:
    """Bind the evaluated identity to the submitting hotkey, not just public bytes.

    Copying a pending commitment cannot reserve another miner's model identity.
    Actual duplicate weights are detected only after authenticated acquisition.
    """
    return content_hash({'schema': 'witness-challenge-2', 'hotkey': hotkey, 'manifest': manifest_id})


@dataclass(frozen=True)
class Submission:
    model_id: str
    certificate: str

    def __post_init__(self):
        if any(len(v) != 64 or re.fullmatch('[0-9a-f]{64}', v) is None for v in
               (self.model_id, self.certificate)):
            raise ValueError('invalid_submission_digest')

    @property
    def commitment(self):
        return SUBMISSION_PREFIX + '|' + encode_bytes(bytes.fromhex(self.model_id) + bytes.fromhex(self.certificate))


def parse_submission(value: str) -> Submission:
    if len(value.encode()) > MAX_COMMITMENT_BYTES or not value.startswith(SUBMISSION_PREFIX + '|'):
        raise ValueError('not_a_v2_p2p_submission')
    raw = decode_bytes(value[4:], 64)
    return Submission(raw[:32].hex(), raw[32:].hex())


def safe_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    p = PurePosixPath(name)
    return (bool(name) and len(name) <= 240 and not p.is_absolute() and p.as_posix() == name
            and all(part not in ('.', '..') and re.fullmatch(r'[A-Za-z0-9_.-]+', part) for part in p.parts)
            and any(fnmatch(p.name, pattern) for pattern in ALLOWED_FILES))


def validate_manifest(manifest: dict, expected: str | None = None) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'arch', 'files'} or manifest['schema_version'] != 'witness-model-2':
        raise ValueError('invalid_manifest_schema')
    if not isinstance(manifest['arch'], str) or manifest['arch'] not in ARCHITECTURES:
        raise ValueError('unsupported_architecture')
    files = manifest['files']
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError('invalid_file_count')
    names = []
    total = 0
    for row in files:
        if (not isinstance(row, dict) or set(row) != {'name', 'size', 'sha256'} or not safe_name(row['name'])
                or type(row['size']) is not int or row['size'] <= 0
                or not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('invalid_manifest_file')
        names.append(row['name'])
        total += row['size']
        if row['name'] == 'witness-manifest.json' or (row['name'].endswith('.json') and row['size'] > 16 * 1024 * 1024):
            raise ValueError('reserved_or_oversized_configuration')
    if names != sorted(set(names)) or 'config.json' not in names or not any(n.endswith('.safetensors') for n in names):
        raise ValueError('missing_or_duplicate_model_files')
    named = set(names)
    if any(str(parent) in named for name in names for parent in PurePosixPath(name).parents):
        raise ValueError('model_file_directory_collision')
    if total > ARCHITECTURES[manifest['arch']]['max_bytes']:
        raise ValueError('weights_too_large')
    if expected is not None and content_hash(manifest) != expected:
        raise ValueError('manifest_hash_mismatch')
    # Independent of directory, filenames and certificate: byte-identical mirrors collide.
    return {'model_id': content_hash(manifest), 'bytes': total,
            'content_id': content_hash({'arch': manifest['arch'],
                                       'weights': sorted((r['sha256'], r['size']) for r in files
                                                         if r['name'].endswith('.safetensors'))})}


def check_config(root: Path, manifest: dict) -> None:
    config = json.loads((root / 'config.json').read_text())
    if not isinstance(config, dict) or config.get('auto_map') or config.get('model_type') != ARCHITECTURES[manifest['arch']]['model_type']:
        raise ValueError('config_does_not_match_architecture')
    for row in manifest['files']:
        if row['name'].endswith('.json'):
            if row['size'] > 16 * 1024 * 1024:
                raise ValueError('configuration_too_large')
            obj = json.loads((root / row['name']).read_text())
            if isinstance(obj, dict) and obj.get('auto_map'):
                raise ValueError('remote_code_forbidden')
        if row['name'].endswith('.safetensors.index.json'):
            index = json.loads((root / row['name']).read_text())
            listed = {f['name'] for f in manifest['files']}
            if not isinstance(index, dict) or not isinstance(index.get('weight_map'), dict):
                raise ValueError('invalid_weight_index')
            if any(not safe_name(n) or n not in listed for n in index.get('weight_map', {}).values()):
                raise ValueError('unlisted_weight_shard')


def prepare_manifest(root: Path, arch: str) -> dict:
    root = root.resolve(strict=True)
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('model_symlink_forbidden')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if safe_name(name):
                files.append({'name': name, 'size': path.stat().st_size, 'sha256': sha256_file(path)})
    manifest = {'schema_version': 'witness-model-2', 'arch': arch, 'files': files}
    validate_manifest(manifest)
    check_config(root, manifest)
    return manifest


def verify_directory(root: Path, manifest: dict) -> dict:
    info = validate_manifest(manifest)
    for row in manifest['files']:
        path = root / row['name']
        if (any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()
                or path.stat().st_size != row['size'] or sha256_file(path) != row['sha256']):
            raise ValueError('model_file_integrity_failed')
    check_config(root, manifest)
    return info
