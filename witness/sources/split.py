"""Create deterministic, format-stratified source splits from pool manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "eval", "hidden")
DEFAULT_RATIOS = {"train": 0.6, "eval": 0.2, "hidden": 0.2}
DEFAULT_SEED = 20260904


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _targets(count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if set(ratios) != set(SPLITS):
        raise ValueError(f"ratios must define exactly {', '.join(SPLITS)}")
    if any(not isinstance(ratios[name], (int, float)) or ratios[name] < 0 for name in SPLITS):
        raise ValueError("split ratios must be non-negative numbers")
    total = sum(float(ratios[name]) for name in SPLITS)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    raw = {name: count * float(ratios[name]) for name in SPLITS}
    result = {name: math.floor(raw[name]) for name in SPLITS}
    remainder = count - sum(result.values())
    order = sorted(SPLITS, key=lambda name: (-(raw[name] - result[name]), SPLITS.index(name)))
    for name in order[:remainder]:
        result[name] += 1
    return result


def load_sources(manifest_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load unique video sources, retaining every pool occurrence."""

    sources: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted((path.resolve() for path in manifest_paths), key=lambda path: path.as_posix()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        videos = manifest.get("videos")
        if not isinstance(videos, list):
            raise ValueError(f"manifest must contain a videos array: {manifest_path}")
        manifest_name = _portable(manifest_path)
        for index, video in enumerate(videos):
            if not isinstance(video, dict) or not video.get("id"):
                raise ValueError(f"video {index} lacks an id: {manifest_path}")
            source_id = str(video["id"])
            format_name = str(video.get("format") or "").strip()
            if not format_name:
                raise ValueError(f"source {source_id} lacks a format tag: {manifest_path}")
            if source_id in sources:
                if sources[source_id]["format"] != format_name:
                    raise ValueError(f"source {source_id} has conflicting format tags")
                sources[source_id]["manifests"].append(manifest_name)
                continue
            duration = video.get("duration")
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                raise ValueError(f"source {source_id} has invalid duration: {manifest_path}")
            sources[source_id] = {
                "id": source_id,
                "format": format_name,
                "duration": float(duration),
                "uploader": str(video.get("uploader") or ""),
                "manifests": [manifest_name],
            }
    if not sources:
        raise ValueError("source split requires at least one pooled video")
    return [sources[source_id] for source_id in sorted(sources)]


def _format_allocations(
    format_sizes: Mapping[str, int],
    targets: Mapping[str, int],
    ratios: Mapping[str, float],
) -> dict[str, dict[str, int]]:
    """Find exact global counts minimizing squared per-format ratio error."""

    formats = sorted(format_sizes)
    # state -> (cost, allocation history). Hidden count is implied by processed total.
    states: dict[tuple[int, int], tuple[float, tuple[tuple[int, int, int], ...]]] = {
        (0, 0): (0.0, ())
    }
    processed = 0
    for format_name in formats:
        size = format_sizes[format_name]
        next_states: dict[tuple[int, int], tuple[float, tuple[tuple[int, int, int], ...]]] = {}
        for (used_train, used_eval), (cost, history) in states.items():
            used_hidden = processed - used_train - used_eval
            for train_count in range(size + 1):
                for eval_count in range(size - train_count + 1):
                    hidden_count = size - train_count - eval_count
                    if used_train + train_count > targets["train"]:
                        continue
                    if used_eval + eval_count > targets["eval"]:
                        continue
                    if used_hidden + hidden_count > targets["hidden"]:
                        continue
                    allocation = (train_count, eval_count, hidden_count)
                    error = sum(
                        (allocation[index] - size * float(ratios[name])) ** 2 / max(1, size)
                        for index, name in enumerate(SPLITS)
                    )
                    candidate = (cost + error, history + (allocation,))
                    key = (used_train + train_count, used_eval + eval_count)
                    current = next_states.get(key)
                    if current is None or candidate < current:
                        next_states[key] = candidate
        states = next_states
        processed += size
    final = states.get((targets["train"], targets["eval"]))
    if final is None:
        raise ValueError("could not construct split with requested global counts")
    return {
        format_name: dict(zip(SPLITS, allocation, strict=True))
        for format_name, allocation in zip(formats, final[1], strict=True)
    }


def build_source_split(
    manifest_paths: Sequence[Path],
    *,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
) -> dict[str, Any]:
    """Assign each unique source ID once, with exact global and optimal format counts."""

    sources = load_sources(manifest_paths)
    target_counts = _targets(len(sources), ratios)
    format_sizes = Counter(source["format"] for source in sources)
    allocations = _format_allocations(format_sizes, target_counts, ratios)
    by_format: dict[str, list[dict[str, Any]]] = {name: [] for name in format_sizes}
    for source in sources:
        by_format[source["format"]].append(source)

    split_sources: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for format_name in sorted(by_format):
        ranked = sorted(
            by_format[format_name],
            key=lambda source: (
                hashlib.sha256(f"{seed}:{format_name}:{source['id']}".encode()).digest(),
                source["id"],
            ),
        )
        offset = 0
        for split_name in SPLITS:
            count = allocations[format_name][split_name]
            split_sources[split_name].extend(ranked[offset : offset + count])
            offset += count

    for values in split_sources.values():
        values.sort(key=lambda source: source["id"])
    all_ids = [source["id"] for values in split_sources.values() for source in values]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("source split is not disjoint")
    if Counter({name: len(values) for name, values in split_sources.items()}) != Counter(target_counts):
        raise AssertionError("source split counts do not match targets")

    format_counts = {
        format_name: {
            **{name: sum(source["format"] == format_name for source in split_sources[name]) for name in SPLITS},
            "total": format_sizes[format_name],
        }
        for format_name in sorted(format_sizes)
    }
    return {
        "schema_version": "1.0",
        "seed": seed,
        "ratios": {name: float(ratios[name]) for name in SPLITS},
        "source_count": len(sources),
        "counts": {name: len(split_sources[name]) for name in SPLITS},
        "format_counts": format_counts,
        "splits": split_sources,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build a deterministic pooled-source split")
    result.add_argument("--manifest", action="append", type=Path, default=[])
    result.add_argument("--out", type=Path, default=Path("data/source_split.json"))
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    manifests = args.manifest or sorted(Path("data").glob("pool*/manifest.json"))
    try:
        result = build_source_split(manifests, seed=args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"output: {args.out}")
    print(f"sources: {result['source_count']}")
    for name in SPLITS:
        print(f"{name}: {result['counts'][name]}")


if __name__ == "__main__":
    main()
