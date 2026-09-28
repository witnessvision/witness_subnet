"""Local Witness benchmark: catalog, commit, clip preparation and round evaluation.

No command in this module connects to a chain or submits weights. Human
references are produced outside this package and placed under
``ROOT/references/<clip_sha256>.json`` before ``evaluate``. Judge, reviewer
and runner factories are operator-owned and use their own spend budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import json
import os
from pathlib import Path
import re

from witness.events import content_hash
from witness.storage import sha256_file, store_immutable, write_private
from .archive import acquire_archive, license_ok
from .contract import Case, Clip, Policy, Reference, Task
from .history import load_history, reserve, used_groups
from .lineage import Catalog, Draw, DrawCommit, clip_spec, exclude, related, select, source_from_media
from .media import render_clip
from .round import run_round
from .simulation import simulate


def _factory(name: str):
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", name):
        raise ValueError("invalid_operator_factory")
    module, attribute = name.split(":", 1)
    return getattr(import_module(module), attribute)


def _private_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    public = Path(__file__).resolve().parents[2]
    if root.is_relative_to(public):
        raise ValueError("benchmark_data_must_be_outside_public_repository")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_mode & 0o077:
        raise ValueError("benchmark_root_must_be_private")
    return root


def build_catalog(manifest: list[dict], output: Path, history_dir: Path | None = None) -> dict:
    required = {"id", "path", "creator", "parents", "license_url", "split"}
    if any(set(row) != required for row in manifest):
        raise ValueError("invalid_source_manifest")
    if any(row["license_url"] != "synthetic-local-fixture" and not license_ok(row["license_url"]) for row in manifest):
        raise ValueError("source_license_not_approved")
    # Fingerprinting decodes whole originals; run several ffmpeg processes at once.
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as pool:
        sources = list(pool.map(lambda row: source_from_media(
            Path(row["path"]), source_id=row["id"], creator=row["creator"], parents=row["parents"],
            license_url=row["license_url"]), manifest))
    splits = {row["id"]: row["split"] for row in manifest}
    provisional = Catalog.build(sources, {source.id: "quarantine" for source in sources})
    quarantined = []
    historical = [source for row in load_history(history_dir) for source in row.selected_sources] if history_dir else []
    for group in provisional.groups():
        if (len({splits[source_id] for source_id in group} - {"quarantine"}) > 1
                or any(related(source, prior) for source in sources if source.id in group
                       for prior in historical)):
            for source_id in group:
                splits[source_id] = "quarantine"
                quarantined.append(source_id)
    catalog = Catalog(sources=sources, split_by_source=splits, group_by_source=provisional.group_by_source)
    _private_root(str(output.parent))
    store_immutable(output, catalog.model_dump())
    return {"catalog_hash": catalog.identity, "source_groups": len(catalog.groups()),
            "quarantined_sources": sorted(quarantined), "output": str(output)}


def make_commit(config: dict) -> dict:
    required = {"catalog", "policy", "champion_hash", "challenger_hashes", "future_finalized_block",
                "history_dir", "challenge_id", "output"}
    if set(config) != required:
        raise ValueError("invalid_commit_configuration")
    catalog = Catalog.model_validate_json(Path(config["catalog"]).read_text())
    policy = Policy.model_validate_json(Path(config["policy"]).read_text())
    history = [row for row in load_history(Path(config["history_dir"]))
               if row.challenge_id != config["challenge_id"]]
    commit = DrawCommit(challenge_id=config["challenge_id"], catalog_hash=catalog.identity,
                        policy_hash=policy.identity, champion_hash=config["champion_hash"],
                        challenger_hashes=config["challenger_hashes"],
                        finalized_block_after_freeze=config["future_finalized_block"],
                        used_groups_hash=content_hash(sorted(used_groups(catalog, history))))
    if len(commit.challenger_hashes) > policy.max_challengers:
        raise ValueError("too_many_round_challengers")
    output = Path(config["output"])
    _private_root(str(output.parent))
    store_immutable(output, commit.model_dump())
    return {"commit_hash": commit.identity, "future_finalized_block": commit.finalized_block_after_freeze,
            "output": str(output), "published_before_block": False}


def _render(draw: Draw, group: str, catalog: Catalog, policy: Policy, config: dict, root: Path) -> Clip:
    spec = clip_spec(draw, group, catalog, policy)
    source = {item.id: item for item in catalog.sources}[spec.source_id]
    if spec.source_id not in config["source_media"]:
        raise FileNotFoundError(spec.source_id)
    source_path = Path(config["source_media"][spec.source_id]).resolve(strict=True)
    if sha256_file(source_path) != source.original_sha256:
        raise RuntimeError("source_media_changed")  # tampering is never a silent replacement
    clip = root / "clips" / (group + ".mp4")
    digest = render_clip(source_path, clip, start=spec.start, duration=spec.duration)
    task = Task(id="task_" + content_hash({"draw": draw.commit.identity, "clip": digest})[:32],
                clip_sha256=digest, duration=spec.duration)
    return Clip(task=task, source_group=group, source_id=spec.source_id,
                source_sha256=source.original_sha256, source_path=str(source_path),
                source_start=spec.start, media_path=str(clip))


def prepare(config: dict) -> dict:
    """Draw groups, render one random clip per group and list them for human labeling.

    ``exclusions`` maps group IDs the operator must not use to
    ``license_invalid`` or ``prohibited_content``; missing or unrenderable media
    is replaced automatically. Every replacement comes from the committed reserve.
    """
    required = {"catalog", "policy", "commit", "finalized_block", "finalized_block_hash",
                "history_dir", "source_media", "exclusions", "root"}
    if set(config) != required:
        raise ValueError("invalid_prepare_configuration")
    root = _private_root(config["root"])
    if (root / "prepared.json").exists():
        raise ValueError("already_prepared_use_saved_bundle")
    catalog = Catalog.model_validate_json(Path(config["catalog"]).read_text())
    policy = Policy.model_validate_json(Path(config["policy"]).read_text())
    commit = DrawCommit.model_validate_json(Path(config["commit"]).read_text())
    history_dir = _private_root(config["history_dir"])
    history = [row for row in load_history(history_dir) if row.challenge_id != commit.challenge_id]
    draw = select(commit, catalog, policy, block_number=config["finalized_block"],
                  block_hash=config["finalized_block_hash"], used_groups=used_groups(catalog, history))
    clips: dict[str, Clip] = {}
    pending = list(draw.groups)
    while pending:
        group = pending.pop(0)
        declared = config["exclusions"].get(group)
        if declared is not None:
            if declared not in ("license_invalid", "prohibited_content"):
                raise ValueError("invalid_declared_exclusion")
            replacement = draw.reserve[0] if draw.reserve else None
            draw = exclude(draw, group, declared)
            pending.append(replacement)
            continue
        try:
            clips[group] = _render(draw, group, catalog, policy, config, root)
        except (FileNotFoundError, ValueError) as exc:
            # Only the committed reserve can replace a group, with the reason recorded.
            reason = "media_unavailable" if isinstance(exc, FileNotFoundError) else "preprocess_failed"
            replacement = draw.reserve[0] if draw.reserve else None
            draw = exclude(draw, group, reason)
            pending.append(replacement)
    reservation = reserve(history_dir, draw, catalog)
    ordered = [clips[group] for group in draw.groups]
    bundle = {"schema_version": "witness-prepared-benchmark-3",
              "catalog": catalog.model_dump(), "policy": policy.model_dump(),
              "draw": draw.model_dump(), "clips": [clip.model_dump() for clip in ordered],
              "history_dir": str(history_dir), "reservation_hash": content_hash(reservation.model_dump())}
    write_private(root / "prepared.json", bundle)
    write_private(root / "labeling.json", [{"clip_sha256": clip.task.clip_sha256, "duration": clip.task.duration,
                                             "media_path": clip.media_path} for clip in ordered])
    return {"prepared": len(ordered), "excluded": draw.exclusions, "draw_hash": content_hash(draw.model_dump()),
            "labeling_queue": str(root / "labeling.json"), "references_dir": str(root / "references")}


def load_reference(root: Path, clip: Clip, policy: Policy) -> Reference:
    path = root / "references" / (clip.task.clip_sha256 + ".json")
    if not path.is_file():
        raise ValueError("missing_human_reference:" + clip.task.clip_sha256)
    reference = Reference.model_validate_json(path.read_text())
    if reference.clip_sha256 != clip.task.clip_sha256 or reference.duration != clip.task.duration:
        raise ValueError("reference_clip_binding_mismatch")
    if reference.kind != policy.reference_kind:
        raise ValueError("reference_kind_not_accepted_by_policy")
    if len(reference.annotators) < policy.min_annotators or any(fact.origin != "annotator" for fact in reference.facts):
        raise ValueError("reference_needs_independent_annotators")
    return reference


def evaluate(config: dict) -> dict:
    required = {"root", "evaluation_factory", "hotkeys"}
    if set(config) != required:
        raise ValueError("invalid_evaluation_configuration")
    root = _private_root(config["root"])
    bundle = json.loads((root / "prepared.json").read_text())
    if bundle.get("schema_version") != "witness-prepared-benchmark-3":
        raise ValueError("wrong_benchmark_bundle")
    catalog = Catalog.model_validate(bundle["catalog"])
    policy = Policy.model_validate(bundle["policy"])
    draw = Draw.model_validate(bundle["draw"])
    reservation_path = Path(bundle["history_dir"]) / (draw.commit.challenge_id + ".reservation.json")
    if content_hash(json.loads(reservation_path.read_text())) != bundle["reservation_hash"]:
        raise ValueError("source_reservation_changed")
    cases = []
    for row in bundle["clips"]:
        clip = Clip.model_validate(row)
        cases.append(Case(**clip.model_dump(), reference=load_reference(root, clip, policy).model_dump()))
    adapters = _factory(config["evaluation_factory"])(config, policy)
    if set(adapters) != {"runner", "judge", "reviewer"}:
        raise ValueError("invalid_evaluation_factory")
    report = run_round(cases=cases, catalog=catalog, draw=draw, policy=policy, hotkeys=config["hotkeys"],
                       runner=adapters["runner"], judge=adapters["judge"], reviewer=adapters["reviewer"],
                       private_root=root)
    store_immutable(Path(bundle["history_dir"]) / (draw.commit.challenge_id + ".report.json"),
                    {"reservation_hash": bundle["reservation_hash"],
                     "report_hash": content_hash(report), "groups_evaluated": report["groups_evaluated"]})
    return {"report": str(root / "round-report.json"), "winner": report["winner"],
            "controls": report["controls"], "weights_submitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("simulate")
    demo.add_argument("--root", type=Path, required=True)
    demo.add_argument("--groups", type=int, default=8)
    acquire = commands.add_parser("acquire-archive")
    acquire.add_argument("--catalogue", type=Path, required=True)
    acquire.add_argument("--requested", type=Path, required=True)
    acquire.add_argument("--root", type=Path, required=True)
    catalog = commands.add_parser("build-catalog")
    catalog.add_argument("--manifest", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--history-dir", type=Path)
    for name in ("make-commit", "prepare", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "simulate":
        result = simulate(_private_root(str(args.root)), count=args.groups)
        print(json.dumps({"simulation_only": True, "groups": result["groups"],
                          "winner": result["report"]["winner"]}))
    elif args.command == "build-catalog":
        print(json.dumps(build_catalog(json.loads(args.manifest.read_text()), args.output,
                                       args.history_dir)))
    elif args.command == "acquire-archive":
        requested = json.loads(args.requested.read_text())
        if not isinstance(requested, dict):
            raise ValueError("invalid_archive_source_request")
        print(json.dumps(acquire_archive(args.catalogue, _private_root(str(args.root)), requested)))
    else:
        config = json.loads(args.config.read_text())
        result = (make_commit(config) if args.command == "make-commit" else
                  prepare(config) if args.command == "prepare" else evaluate(config))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
