"""Source lineage, near-duplicate quarantine and auditable source-group and clip draws."""
from __future__ import annotations

import hashlib
import random
from itertools import combinations
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from witness.events import StrictModel, content_hash
from witness.storage import sha256_file
from .contract import CLIP_GRID_S, Digest, Policy
from .media import probe

# Covers the fingerprint and every matching constant below; bump it when any
# changes, because the catalog hash must change with the grouping it implies.
FINGERPRINT_VERSION = "witness-av-lineage-3"
UNIFORM_FRAME_STD = 3.  # black, solid or faded frames carry no source identity
# Seconds of aligned near-identical fingerprints that prove shared footage.
# Across 4,950 pairs of 100 unrelated Archive videos, 3 s linked 148 pairs by
# chance; 8 s linked 9, consistent with shared templates or intros.
MIN_ALIGNED_S = 8


def _run(command: list[str]) -> bytes:
    result = subprocess.run(command, capture_output=True, timeout=900, check=False)
    if result.returncode:
        raise ValueError("media_fingerprint_failed")
    return result.stdout


def fingerprint(path: Path) -> tuple[list[int | None], list[int | None]]:
    """1 Hz perceptual video and audio fingerprints, independent of container bytes.

    Uniform frames and silent seconds are ``None``: two unrelated films must not
    join one lineage group because both fade to black or pause.
    """
    frames = _run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                   "fps=1,scale=9:8,format=gray", "-f", "rawvideo", "-"])
    if len(frames) % 72:
        raise ValueError("invalid_video_fingerprint_stream")
    visual: list[int | None] = []
    for offset in range(0, len(frames), 72):
        image = np.frombuffer(frames[offset:offset + 72], dtype=np.uint8).reshape(8, 9)
        if float(image.std()) < UNIFORM_FRAME_STD:
            visual.append(None)
            continue
        bits = image[:, 1:] > image[:, :-1]
        visual.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    pcm = _run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                "-ar", "8000", "-f", "s16le", "-"])
    audio: list[int | None] = []
    for offset in range(0, len(pcm) - 15999, 16000):
        samples = np.frombuffer(pcm[offset:offset + 16000], dtype="<i2").astype(np.float32)
        if float(np.sqrt(np.mean(samples * samples))) < 100:
            audio.append(None)
            continue
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        bands = np.array([float(np.mean(spectrum[a:b])) for a, b in
                          zip(np.geomspace(15, 3900, 33).astype(int)[:-1],
                              np.geomspace(15, 3900, 33).astype(int)[1:])])
        bits = np.log1p(bands[1:]) > np.log1p(bands[:-1])
        audio.append(int.from_bytes(np.packbits(np.pad(bits, (0, 1))).tobytes(), "big"))
    if not visual or not audio:
        raise ValueError("source_missing_audio_or_video")
    return visual, audio


class Source(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
    original_sha256: Digest
    duration: float = Field(gt=0, allow_inf_nan=False)
    creator: str | None = None
    parents: list[str] = Field(default_factory=list)
    visual: list[int | None] = Field(min_length=1)
    audio: list[int | None] = Field(min_length=1)
    license_url: str = Field(min_length=1)


def source_from_media(path: Path, *, source_id: str, license_url: str,
                      creator: str | None = None, parents: list[str] | None = None) -> Source:
    visual, audio = fingerprint(path)
    return Source(id=source_id, original_sha256=sha256_file(path),
                  duration=float(probe(path)["format"]["duration"]), creator=creator,
                  parents=parents or [], visual=visual, audio=audio, license_url=license_url)


def _matches(a: list[int | None], b: list[int | None], *, width: int, distance: int) -> bool:
    """Find MIN_ALIGNED_S aligned nontrivial one-second windows, including fragments."""
    buckets: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    # With at most six bit errors across eight chunks, at least two chunks
    # remain identical. Pair keys avoid missing a re-encode whose errors fall
    # in every 16-bit quarter.
    part_width = width // 8
    mask = (1 << part_width) - 1
    pairs = list(combinations(range(8), 2))
    for j, value in enumerate(b):
        if value is None:
            continue
        for first, second in pairs:
            buckets[(first, second, (value >> (part_width * first)) & mask,
                     (value >> (part_width * second)) & mask)].append(j)
    offsets = set()
    for i, value in enumerate(a):
        if value is None:
            continue
        for first, second in pairs:
            locations = buckets[(first, second, (value >> (part_width * first)) & mask,
                                 (value >> (part_width * second)) & mask)]
            offsets.update(j - i for j in locations)
    for delta in offsets:
        run = 0
        for i in range(max(0, -delta), min(len(a), len(b) - delta)):
            x, y = a[i], b[i + delta]
            run = run + 1 if x is not None and y is not None and (x ^ y).bit_count() <= distance else 0
            if run >= MIN_ALIGNED_S:
                return True
    return False


def related(a: Source, b: Source) -> bool:
    if (a.id == b.id or a.original_sha256 == b.original_sha256
            or a.id in b.parents or b.id in a.parents
            or a.creator and b.creator and a.creator.strip() and b.creator.strip()
            and a.creator.casefold().strip() == b.creator.casefold().strip()):
        return True
    return (_matches(a.visual, b.visual, width=64, distance=6)
            or _matches(a.audio, b.audio, width=32, distance=5))


class Catalog(StrictModel):
    schema_version: Literal["witness-source-catalog-3"] = "witness-source-catalog-3"
    fingerprint_version: Literal["witness-av-lineage-3"] = FINGERPRINT_VERSION
    sources: list[Source]
    split_by_source: dict[str, Literal["train", "dev", "eval", "quarantine"]]
    group_by_source: dict[str, str]  # computed once by build(); its hash commits to the grouping

    @model_validator(mode="after")
    def consistent(self):
        ids = {source.id for source in self.sources}
        if (len(ids) != len(self.sources) or set(self.split_by_source) != ids
                or set(self.group_by_source) != ids
                or any(self.group_by_source[group] != group for group in self.group_by_source.values())):
            raise ValueError("catalog_source_identity_mismatch")
        if any(parent not in ids for source in self.sources for parent in source.parents):
            raise ValueError("unknown_source_parent")
        for group in self.groups():
            splits = {self.split_by_source[source_id] for source_id in group} - {"quarantine"}
            if len(splits) > 1:
                raise ValueError("source_lineage_crosses_splits")
        return self

    @classmethod
    def build(cls, sources: list[Source], split_by_source: dict[str, str]) -> "Catalog":
        """Group related sources once (pairwise, union-find); the group ID is its smallest source ID."""
        parent = {source.id: source.id for source in sources}

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for i, source in enumerate(sources):
            for other in sources[i + 1:]:
                if find(source.id) != find(other.id) and related(source, other):
                    parent[find(other.id)] = find(source.id)
        members: dict[str, list[str]] = defaultdict(list)
        for source_id in parent:
            members[find(source_id)].append(source_id)
        group_by_source = {source_id: min(group) for group in members.values() for source_id in group}
        return cls(sources=sources, split_by_source=split_by_source, group_by_source=group_by_source)

    def groups(self) -> list[set[str]]:
        members: dict[str, set[str]] = defaultdict(set)
        for source_id, group in self.group_by_source.items():
            members[group].add(source_id)
        return sorted(members.values(), key=min)

    def eval_groups(self) -> list[tuple[str, list[str]]]:
        return [(min(group), sorted(group)) for group in self.groups()
                if {self.split_by_source[source_id] for source_id in group} == {"eval"}]

    @property
    def identity(self) -> str:
        return content_hash(self.model_dump())


class DrawCommit(StrictModel):
    challenge_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    catalog_hash: Digest
    policy_hash: Digest
    champion_hash: Digest
    challenger_hashes: list[Digest] = Field(min_length=1)
    finalized_block_after_freeze: int = Field(ge=1)
    used_groups_hash: Digest

    @model_validator(mode="after")
    def distinct_models(self):
        models = [self.champion_hash, *self.challenger_hashes]
        if len(models) != len(set(models)):
            raise ValueError("duplicate_round_model")
        return self

    @property
    def identity(self) -> str:
        return content_hash(self.model_dump())


class ClipSpec(StrictModel):
    group: str
    source_id: str
    start: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(gt=0, allow_inf_nan=False)


class Draw(StrictModel):
    commit: DrawCommit
    finalized_block_hash: Digest
    groups: list[str]
    reserve: list[str]
    exclusions: dict[str, str]


def _usable(catalog: Catalog, members: list[str], policy: Policy) -> list[str]:
    sources = {source.id: source for source in catalog.sources}
    return [source_id for source_id in members if sources[source_id].duration >= policy.clip_min_s + CLIP_GRID_S]


def _steps(seconds: float) -> int:
    return int(seconds / CLIP_GRID_S + 1e-9)


def _seed(commit_hash: str, block_hash: str) -> bytes:
    return hashlib.sha256(bytes.fromhex(commit_hash) + bytes.fromhex(block_hash)).digest()


def select(commit: DrawCommit, catalog: Catalog, policy: Policy, *, block_number: int, block_hash: str,
           used_groups: set[str]) -> Draw:
    if (catalog.identity != commit.catalog_hash or policy.identity != commit.policy_hash
            or content_hash(sorted(used_groups)) != commit.used_groups_hash
            or block_number != commit.finalized_block_after_freeze or len(block_hash) != 64):
        raise ValueError("draw_commit_or_entropy_mismatch")
    if len(commit.challenger_hashes) > policy.max_challengers:
        raise ValueError("too_many_round_challengers")
    available = [group_id for group_id, members in catalog.eval_groups()
                 if group_id not in used_groups and _usable(catalog, members, policy)]
    random.Random(int.from_bytes(_seed(commit.identity, block_hash), "big")).shuffle(available)
    if len(available) < policy.confirmation_size:
        raise ValueError("insufficient_fresh_source_groups")
    return Draw(commit=commit, finalized_block_hash=block_hash, groups=available[:policy.confirmation_size],
                reserve=available[policy.confirmation_size:], exclusions={})


def clip_spec(draw: Draw, group: str, catalog: Catalog, policy: Policy) -> ClipSpec:
    """Uniform source, duration and start for one group, fixed by the committed entropy.

    Nothing is skipped for being dull: black, silent or broken stretches are
    valid clips. Each group's draw is independent of its position in the list,
    so a reserve replacement cannot change any other clip.
    """
    members = dict(catalog.eval_groups()).get(group)
    if not members or group not in draw.groups:
        raise ValueError("clip_group_not_drawn")
    usable = _usable(catalog, members, policy)
    rng = random.Random(int.from_bytes(hashlib.sha256(
        _seed(draw.commit.identity, draw.finalized_block_hash) + group.encode()).digest(), "big"))
    source = {item.id: item for item in catalog.sources}[rng.choice(usable)]
    longest = min(policy.clip_max_s, source.duration - CLIP_GRID_S)
    duration = rng.randint(_steps(policy.clip_min_s), _steps(longest)) * CLIP_GRID_S
    start = rng.randint(0, _steps(source.duration - duration - CLIP_GRID_S)) * CLIP_GRID_S
    return ClipSpec(group=group, source_id=source.id, start=start, duration=duration)


def exclude(draw: Draw, group_id: str, reason: str) -> Draw:
    """Replace an unusable source only from the committed deterministic reserve."""
    if group_id not in draw.groups or reason not in {"media_unavailable", "license_invalid", "preprocess_failed",
                                                     "prohibited_content"} or not draw.reserve:
        raise ValueError("invalid_draw_exclusion")
    groups = draw.groups.copy()
    groups[groups.index(group_id)] = draw.reserve[0]
    return draw.model_copy(update={"groups": groups, "reserve": draw.reserve[1:],
                                   "exclusions": {**draw.exclusions, group_id: reason}})
