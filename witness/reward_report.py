"""Recalculate saved v1.8 rounds under both policies without miner inference.

Usage: python -m witness.reward_report ROUND.json [ROUND.json ...] --out report.json
Inputs must share a scoring identity; multiple rounds are pooled per miner UID.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from witness.reward import MIN_PARTIAL_QUALITY, PARTIAL_BAND, summarize_records
from witness.score_v1_0_0 import SCORER_VERSION
from witness.subnet.validator import apply_duplicate_sharing, apply_relative_gate


def compare_rounds(paths: list[Path]) -> dict:
    records = {"1.8": [], SCORER_VERSION: []}
    provenance, identity, seen = [], None, set()
    for path in paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            raise ValueError("duplicate round input")
        seen.add(digest)
        artifact = json.loads(raw)
        current = artifact["scoring_identity"]
        if current["version"] != "1.8":
            raise ValueError("reward comparison requires v1.8 quality records")
        if identity is not None and identity != current:
            raise ValueError("cannot pool different scoring identities")
        identity = current
        provenance.append({"path": str(path), "sha256": digest})
        for version in records:
            rows = deepcopy([r for miner in artifact["miners"].values() for r in miner["scenes"]])
            # Relative competition and duplicate sharing stay within each round.
            apply_relative_gate(rows, score_version=version)
            apply_duplicate_sharing(rows)
            records[version].extend(rows)
    uids = sorted({int(r["uid"]) for r in records["1.8"]})
    return {
        "comparison": "reward policy only; identical saved reconstructions and quality",
        "quality_identity": identity,
        "reward_policy": {
            "version": SCORER_VERSION, "partial_band": PARTIAL_BAND,
            "minimum_partial_quality": MIN_PARTIAL_QUALITY,
            "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("reward.py", "reward_report.py", "subnet/validator.py")},
        },
        "inputs": provenance,
        "miners": {str(uid): {
            version: summarize_records([r for r in rows if int(r["uid"]) == uid])
            for version, rows in records.items()} for uid in uids},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rounds", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = compare_rounds(args.rounds)
    # Preserve historical reports; callers choose a fresh output path.
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
