"""CLI for deterministic Witness scene generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render import render_video
from .scene import build_scene


def generate(seed: int, tier: int, output: Path, *, debug_labels: bool = False) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    scene = build_scene(seed, tier, debug_labels=debug_labels)
    scene_path = output / "scene.json"
    video_path = output / "video.mp4"
    scene_path.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_video(scene, video_path)
    return scene_path, video_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate deterministic Witness scenes")
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--tier", type=int, choices=(1, 2, 3), required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--count", type=int, default=1, help="sequential seeds; output becomes a batch root")
    result.add_argument(
        "--debug-labels",
        action="store_true",
        help="burn in internal actor/object/shot labels (tier 1 only)",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.debug_labels and args.tier in (2, 3):
        raise SystemExit("--debug-labels is forbidden at tiers 2 and 3")
    if args.count == 1:
        paths = [generate(args.seed, args.tier, args.out, debug_labels=args.debug_labels)]
    else:
        paths = [
            generate(
                seed,
                args.tier,
                args.out / f"scene_{seed}",
                debug_labels=args.debug_labels,
            )
            for seed in range(args.seed, args.seed + args.count)
        ]
    for scene_path, video_path in paths:
        print(f"generated {scene_path} and {video_path}")


if __name__ == "__main__":
    main()
