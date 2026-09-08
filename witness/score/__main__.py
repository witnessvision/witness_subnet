"""Command-line interface for deterministic Witness scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .scorer import score_reconstruction


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a reconstruction against Witness scene.json v1.1")
    parser.add_argument("--scene", required=True, type=Path, help="scene directory or scene.json path")
    parser.add_argument("--reconstruction", required=True, type=Path)
    parser.add_argument("--cost", required=True, type=Path)
    args = parser.parse_args()
    scene_path = args.scene / "scene.json" if args.scene.is_dir() else args.scene
    try:
        report = score_reconstruction(_read_json(scene_path), _read_json(args.reconstruction), _read_json(args.cost))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid scoring input: {exc}") from exc
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
