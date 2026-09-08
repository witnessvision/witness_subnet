"""CLI entrypoint for real-video recomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_recomposition
from .tts import DEFAULT_MODEL, logged_call_count


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate a real-video Witness scene")
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--tier", type=int, choices=(1, 2, 3), required=True)
    result.add_argument("--pool", type=Path, default=Path("data/pool/manifest.json"))
    result.add_argument("--out", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    scene, video = generate_recomposition(args.seed, args.tier, args.pool, args.out)
    print(f"generated {scene} and {video}")
    print(f"tts_model: {DEFAULT_MODEL}")
    print(f"tts_calls_logged: {logged_call_count()}")


if __name__ == "__main__":
    main()
