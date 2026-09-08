"""Benchmark freeze: lock scene truth files and the scoring/recomposition code by hash.

Create:  .venv/bin/python -m witness.freeze --version 1.5 --scenes data/scenes/hidden-v15 --out data/benchmark_v1.5.lock.json
Verify:  .venv/bin/python -m witness.freeze --verify data/benchmark_v1.5.lock.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_GLOBS = ("witness/score/*.py", "witness/recompose/*.py", "witness/contract.py", "witness/validate.py", "witness/tools/metering.py")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes() -> dict[str, str]:
    files: list[Path] = []
    for pattern in CODE_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    return {str(p.relative_to(ROOT)): _sha256(p) for p in files if p.name != "__pycache__"}


def scene_hashes(scene_roots: list[Path]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for root in scene_roots:
        for scene_json in sorted(root.glob("*/scene.json")):
            scene = json.loads(scene_json.read_text(encoding="utf-8"))
            video = scene_json.parent / "video.mp4"
            out[scene_json.parent.name] = {
                "scene_json_sha256": _sha256(scene_json),
                "video_sha256": _sha256(video) if video.exists() else "",
                "schema_version": str(scene.get("schema_version", "")),
                "tier": int(scene.get("difficulty", 0)),
                "source_id": str((scene.get("source") or {}).get("id", "")),
            }
    return out


def build_lock(version: str, scene_roots: list[Path]) -> dict:
    scenes = scene_hashes(scene_roots)
    return {
        "benchmark_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": code_hashes(),
        "scenes": scenes,
        "scene_count": len(scenes),
        "schema_versions": sorted({s["schema_version"] for s in scenes.values()}),
    }


def verify_lock(lock: dict, scene_roots: list[Path]) -> list[str]:
    problems: list[str] = []
    for rel, digest in lock["code"].items():
        path = ROOT / rel
        if not path.exists():
            problems.append(f"missing code file {rel}")
        elif _sha256(path) != digest:
            problems.append(f"code changed since freeze: {rel}")
    current = scene_hashes(scene_roots)
    for name, entry in lock["scenes"].items():
        cur = current.get(name)
        if cur is None:
            problems.append(f"missing scene {name}")
        elif cur["scene_json_sha256"] != entry["scene_json_sha256"]:
            problems.append(f"scene.json changed: {name}")
        elif entry.get("video_sha256") and cur["video_sha256"] != entry["video_sha256"]:
            problems.append(f"video changed: {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze or verify the Witness benchmark")
    ap.add_argument("--version", default="1.5")
    ap.add_argument("--scenes", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--verify")
    args = ap.parse_args(argv)
    if args.verify:
        lock = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        roots = [Path(p) for p in args.scenes] or [
            ROOT / "data/scenes/hidden-v15", ROOT / "data/scenes/dev-v15"
        ]
        problems = verify_lock(lock, roots)
        for p in problems:
            print("FAIL:", p)
        print("verified" if not problems else f"{len(problems)} problems", lock["benchmark_version"], lock["scene_count"], "scenes")
        return 1 if problems else 0
    roots = [Path(p) for p in args.scenes]
    lock = build_lock(args.version, roots)
    out = Path(args.out or f"data/benchmark_v{args.version}.lock.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print("locked", lock["scene_count"], "scenes,", len(lock["code"]), "code files ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
