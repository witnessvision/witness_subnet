from __future__ import annotations

import json
import subprocess

import pytest

from witness.events import content_hash
from witness.storage import sha256_file, write_private
from witness.benchmark import simulation
from witness.benchmark.archive import acquire_archive
from witness.benchmark.cli import build_catalog, evaluate, make_commit, prepare
from witness.benchmark.contract import Claim, Fact, Policy, Reference, Response
from witness.benchmark.history import load_history, reserve, used_groups
from witness.benchmark.lineage import (CLIP_GRID_S, Catalog, DrawCommit, Source, clip_spec, exclude,
                                       fingerprint, related, select)
from witness.benchmark.media import PREPROCESSOR_ID
from witness.benchmark.round import InfrastructureError, controls, decide
from witness.benchmark.reward import reward
from witness.benchmark.scoring import Assessment, Decision, score
from witness.benchmark.simulation import RUNTIME, fixture_reference, simulate, synthetic_source

ANNOTATORS = ["a" * 64, "b" * 64]


def test_private_data_root_rejects_public_checkout_and_symlink_alias(tmp_path, monkeypatch):
    from witness.benchmark import cli
    public = tmp_path / "witness_subnet"
    public.mkdir()
    monkeypatch.setattr(cli, "__file__", str(public / "witness/benchmark/cli.py"))
    alias = tmp_path / "public-alias"
    alias.symlink_to(public, target_is_directory=True)
    for target in (public, public / ".private", alias / ".private"):
        with pytest.raises(ValueError, match="outside_public_repository"):
            cli._private_root(str(target))
    assert not (public / ".private").exists()
    private = tmp_path / "validator-state" / "artifacts"
    assert cli._private_root(str(private)) == private
    assert private.stat().st_mode & 0o777 == 0o700


def claim(identifier="fact", *, start=2., end=4., description="A hand turns a valve", modality="visual"):
    return Claim(id=identifier, start=start, end=end, subject="A hand",
                 description=description, modality=modality)


def reference(*facts, salience="core"):
    return Reference(kind="human", clip_sha256="a" * 64, duration=12., annotators=ANNOTATORS,
                     facts=[Fact(claim=value, salience=salience) for value in facts])


def policy(**changes):
    values = {"judge_id": "judge:fixture", "runtime_hash": RUNTIME, "preprocessing_hash": PREPROCESSOR_ID}
    return Policy(**(values | changes))


def judged(response, decisions):
    """decisions maps prediction id -> (status, fact ids)."""
    return Assessment(judge_id="judge:fixture", decisions=[
        Decision.whole(row, decisions[row.id][0], decisions[row.id][1], "fixture")
        for row in response.claims])


def quality(ref, claims, decisions, **changes):
    response = Response(claims=claims)
    return score(ref, response, judged(response, decisions), policy(**changes))


VISUAL = claim("v")
SOUND = claim("s", description="A bell rings", modality="sound")


def test_oracle_scores_one_and_empty_scores_zero_even_when_instant():
    ref = reference(VISUAL, SOUND)
    oracle = quality(ref, [VISUAL, SOUND], {"v": ("supported", ["v"]), "s": ("supported", ["s"])})
    assert oracle["quality"] == 1.
    empty = quality(ref, [], {})
    assert empty["quality"] == 0.
    assert reward(empty["quality"], 0., 12., policy(), valid=True)["reward"] == 0.
    assert reward(1., 0., 12., policy(), valid=False)["reward"] == 0.


def test_modality_absent_from_reference_lowers_precision_but_is_not_a_zero_modality():
    result = quality(reference(VISUAL), [VISUAL, claim("x", description="Music plays", modality="sound")],
                     {"v": ("supported", ["v"]), "x": ("unresolved", [])})
    assert not result["modalities"]["sound"]["in_reference"]
    assert result["recall"] == 1. and result["precision"] == .5
    assert result["quality"] == pytest.approx(2 / 3)


def test_reviewed_fact_never_adds_a_modality_to_recall():
    music = claim("m", description="Music plays", modality="sound")
    ref = Reference(kind="human", clip_sha256="a" * 64, duration=12., annotators=ANNOTATORS,
                    facts=[Fact(claim=VISUAL, salience="core"),
                           Fact(claim=music, salience="detail", origin="review")])
    assert quality(ref, [VISUAL], {"v": ("supported", ["v"])})["recall"] == 1.
    with pytest.raises(ValueError, match="without_annotator_fact"):
        Reference(kind="human", clip_sha256="a" * 64, duration=12., annotators=ANNOTATORS,
                  facts=[Fact(claim=music, salience="detail", origin="review")])


def test_policy_clip_bounds_sit_on_the_frame_grid():
    with pytest.raises(ValueError, match="frame_grid"):
        policy(clip_min_s=10.05)


def test_duplicates_do_not_help_and_false_claims_hurt_more_than_unverifiable_ones():
    ref = reference(VISUAL, SOUND)
    base = quality(ref, [VISUAL], {"v": ("supported", ["v"])})["quality"]
    duplicate = quality(ref, [VISUAL, VISUAL.model_copy(update={"id": "v2"})],
                        {"v": ("supported", ["v"]), "v2": ("supported", ["v"])})["quality"]
    assert duplicate < base
    many = reference(*[claim(f"f{index}", description=f"Fact {index}") for index in range(10)])
    honest = [claim(f"p{index}", description=f"Fact {index}") for index in range(5)] + [
        claim(f"u{index}", description=f"Guess {index}") for index in range(5)]
    decisions = {f"p{index}": ("supported", [f"f{index}"]) for index in range(5)} | {
        f"u{index}": ("unresolved", []) for index in range(5)}
    flooded = honest + [claim(f"copy{index}", description="Fact 0") for index in range(54)]
    assert quality(many, flooded, decisions | {f"copy{index}": ("supported", ["f0"]) for index in range(54)}
                   )["quality"] < quality(many, honest, decisions)["quality"]
    extra = claim("e", description="A dog barks", modality="sound")
    unresolved = quality(ref, [VISUAL, extra], {"v": ("supported", ["v"]), "e": ("unresolved", [])})["quality"]
    contradicted = quality(ref, [VISUAL, extra], {"v": ("supported", ["v"]), "e": ("contradicted", ["s"])})["quality"]
    assert contradicted < unresolved < base


def test_claim_order_does_not_change_score():
    first, second = claim("a", start=2.), claim("b", start=2., description="A red valve", modality="visual")
    ref = reference(first, second)
    decisions = {"a": ("supported", ["a"]), "b": ("supported", ["b"])}
    assert quality(ref, [first, second], decisions) == quality(ref, [second, first], decisions)


def test_time_is_a_gate_with_tolerance_and_citations_are_bounded():
    ref = reference(VISUAL)
    near = claim("n", start=3.5, end=5.)
    assert quality(ref, [near], {"n": ("supported", ["v"])})["quality"] == 1.
    far = claim("f", start=8., end=10.)
    with pytest.raises(ValueError, match="wrong_interval"):
        quality(ref, [far], {"f": ("supported", ["v"])})
    whole_clip = claim("w", start=0., end=12.)
    with pytest.raises(ValueError, match="wrong_interval"):
        quality(ref, [whole_clip], {"w": ("supported", ["v"])})
    many = reference(*[claim(f"f{index}", description=f"Detail {index}") for index in range(4)])
    vague = claim("vague", description="Things happen")
    with pytest.raises(ValueError, match="citation_count"):
        quality(many, [vague], {"vague": ("supported", [f"f{index}" for index in range(4)])})


def test_core_facts_weigh_more_than_details():
    core, detail = claim("core"), claim("detail", description="The valve is red")
    ref = Reference(kind="human", clip_sha256="a" * 64, duration=12., annotators=ANNOTATORS,
                    facts=[Fact(claim=core, salience="core"), Fact(claim=detail, salience="detail")])
    got_core = quality(ref, [core], {"core": ("supported", ["core"])})
    got_detail = quality(ref, [detail], {"detail": ("supported", ["detail"])})
    assert got_core["recall"] == pytest.approx(1 / 1.5) and got_detail["recall"] == pytest.approx(.5 / 1.5)
    assert got_core["missed_core"] == 0 and got_detail["missed_core"] == 1


def test_speed_cannot_buy_a_real_quality_loss():
    rules = policy(quality_floor=0.)
    champion = {"quality": .6, "reward": reward(.6, 110., 12., rules, valid=True)["reward"]}
    fast_worse = {"quality": .55, "reward": reward(.55, 0., 12., rules, valid=True)["reward"]}
    assert fast_worse["reward"] > champion["reward"]
    assert not decide([(champion, fast_worse)] * 64, policy=rules, alpha=.05, seed="a" * 64)["promoted"]
    better = {"quality": .7, "reward": reward(.7, 110., 12., rules, valid=True)["reward"]}
    assert decide([(champion, better)] * 64, policy=rules, alpha=.05, seed="a" * 64)["promoted"]


def test_controls_are_blind_to_the_clip():
    generic = controls(17.5)["generic"]
    assert {row.modality for row in generic.claims} == {"visual", "speech", "text", "sound"}
    assert all(row.end == 17.5 for row in generic.claims)
    assert controls(12.)["empty"].claims == []


def source(identifier, visual, *, creator=None, parents=None, duration=60.):
    return Source(id=identifier, original_sha256=content_hash({"source": identifier}), duration=duration,
                  creator=creator, parents=parents or [], visual=visual, audio=[None] * len(visual),
                  license_url="fixture")


def frames(seed, count=6):
    return [int(content_hash({"seed": seed, "frame": index})[:16], 16) for index in range(count)]


def test_source_family_does_not_cross_train_and_eval():
    original = source("original", frames("o", 20))
    fragment = source("fragment", original.visual[4:13])  # 9 s of reused footage
    assert related(original, fragment)
    assert not related(original, source("short", original.visual[4:9]))  # 5 s is not proof
    assert related(original, source("reencoded", [value ^ 0x0001000100010001 for value in original.visual]))
    with pytest.raises(ValueError, match="crosses_splits"):
        Catalog.build([original, fragment], {"original": "train", "fragment": "eval"})
    with pytest.raises(ValueError, match="crosses_splits"):
        Catalog.build([source("a", frames("a"), creator="same"), source("b", frames("b"), creator="same")],
                      {"a": "dev", "b": "eval"})


def test_black_frames_do_not_fingerprint_or_link_unrelated_sources(tmp_path):
    path = tmp_path / "black.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x180:d=5",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5", "-c:v", "libx264",
                    "-c:a", "aac", str(path)], check=True, timeout=60)
    visual, audio = fingerprint(path)
    assert visual and all(value is None for value in visual)
    assert all(value is None for value in audio)
    assert not related(source("x", [None] * 4 + frames("x")), source("y", [None] * 4 + frames("y")))


def commit_for(catalog, rules, *, used=(), challengers=("c" * 64,)):
    return DrawCommit(challenge_id="round", catalog_hash=catalog.identity, policy_hash=rules.identity,
                      champion_hash="b" * 64, challenger_hashes=list(challengers),
                      finalized_block_after_freeze=101, used_groups_hash=content_hash(sorted(used)))


def test_draw_is_reproducible_skips_used_and_too_short_groups():
    rules = policy(screen_size=1, confirmation_size=3)
    sources = [source(f"s{index}", frames(index)) for index in range(5)] + [source("short", frames("short"), duration=9.)]
    catalog = Catalog.build(sources, {row.id: "eval" for row in sources})
    commit = commit_for(catalog, rules, used={"s0"})
    first = select(commit, catalog, rules, block_number=101, block_hash="d" * 64, used_groups={"s0"})
    assert first == select(commit, catalog, rules, block_number=101, block_hash="d" * 64, used_groups={"s0"})
    assert not {"s0", "short"} & set(first.groups + first.reserve)
    with pytest.raises(ValueError, match="entropy"):
        select(commit, catalog, rules, block_number=102, block_hash="d" * 64, used_groups={"s0"})
    with pytest.raises(ValueError, match="duplicate_round_model"):
        commit_for(catalog, rules, challengers=("b" * 64,))


def test_clip_position_and_duration_are_random_bounded_and_replacement_safe():
    rules = policy(screen_size=1, confirmation_size=8)
    sources = [source(f"s{index}", frames(index), duration=20. + index * 30) for index in range(10)]
    catalog = Catalog.build(sources, {row.id: "eval" for row in sources})
    draw = select(commit_for(catalog, rules), catalog, rules, block_number=101, block_hash="d" * 64,
                  used_groups=set())
    specs = {group: clip_spec(draw, group, catalog, rules) for group in draw.groups}
    by_id = {row.id: row for row in sources}
    for spec in specs.values():
        assert rules.clip_min_s <= spec.duration <= rules.clip_max_s
        assert 0 <= spec.start and spec.start + spec.duration <= by_id[spec.source_id].duration
        assert (spec.duration / CLIP_GRID_S).is_integer() and (spec.start / CLIP_GRID_S).is_integer()
    assert len({spec.duration for spec in specs.values()}) > 1
    replaced = exclude(draw, draw.groups[0], "media_unavailable")
    for group in draw.groups[1:]:
        assert clip_spec(replaced, group, catalog, rules) == specs[group]


def test_history_blocks_reuploads_even_under_a_new_source_id(tmp_path):
    rules = policy(screen_size=1, confirmation_size=2)
    first = [source("old", frames("old", 10)), source("other", frames("other", 10))]
    original = Catalog.build(first, {"old": "eval", "other": "eval"})
    draw = select(commit_for(original, rules), original, rules, block_number=101, block_hash="d" * 64,
                  used_groups=set())
    reserve(tmp_path / "history", draw, original)
    refreshed = Catalog.build([source("new_reupload", first[0].visual)], {"new_reupload": "eval"})
    assert used_groups(refreshed, load_history(tmp_path / "history")) == {"old", "other", "new_reupload"}


def test_round_promotes_the_better_challenger_screens_the_broken_one_and_resumes(tmp_path):
    result = simulate(tmp_path)
    report, roles = result["report"], result["roles"]
    assert result["provider_calls_paid"] == 0 and report["groups_evaluated"] == 8
    assert result["runner_calls"] == 8 + 8 + 2  # broken challenger stops after screening
    assert report["winner"] == roles["strong"] and report["controls"]["pass"]
    assert report["decisions"][roles["broken"]]["status"] == "screen_reject"
    receipts = [json.loads(path.read_text()) for path in sorted((tmp_path / "results").glob("*.json"))]
    assert any(row["review"]["added_facts"] for row in receipts)
    assert any(row["grades"][roles["strong"]]["score"]["contradicted"] for row in receipts)
    assert len({row["task"]["duration"] for row in receipts}) > 1
    assert simulate(tmp_path)["runner_calls"] == 0
    (tmp_path / "round-report.json").unlink()
    (tmp_path / "results" / "000.json").unlink()
    with pytest.raises(InfrastructureError, match="unknown_inference_outcome_requires_reconciliation"):
        simulate(tmp_path)


def test_review_budget_is_shared_between_models(tmp_path):
    from witness.benchmark.adjudication import adjudicate
    from witness.benchmark.contract import Case, Task
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    task = Task(id="task", clip_sha256="a" * 64, duration=12.)
    case = Case(task=task, source_group="g", source_id="s", source_sha256="b" * 64, source_path=str(clip),
                source_start=0., media_path=str(clip), reference=reference(VISUAL))
    rules = policy(judge_id=simulation.FakeJudge.identity, reviewer_id="fake-blind-av-reviewer-3",
                   novel_review_limit=2)
    greedy = Response(claims=[claim(f"g{index}", description=f"Greedy {index}") for index in range(5)])
    modest = Response(claims=[claim("m", description="A red triangle is visible")])
    reviewer = simulation.FakeReviewer({"A red triangle is visible"})
    reference_after, assessments, summary = adjudicate(
        case, [greedy, modest], policy=rules, judge=simulation.FakeJudge(rules), reviewer=reviewer,
        private_root=tmp_path)
    assert summary["reviewed"] == 2 and summary["added_facts"] == 1
    assert assessments[1].decisions[0].status == "supported"


def test_all_challengers_screened_champion_still_runs_every_group(tmp_path, monkeypatch):
    original = simulation.FakeRunner.__init__
    monkeypatch.setattr(simulation.FakeRunner, "__init__", lambda self, roles: original(
        self, {model_id: "broken" if role == "strong" else role for model_id, role in roles.items()}))
    result = simulate(tmp_path)
    assert result["runner_calls"] == 8 + 2 + 2
    assert result["report"]["groups_evaluated"] == 8
    assert result["report"]["winner"] == result["roles"]["champion"]


def test_resume_rejects_an_edited_case_receipt(tmp_path):
    simulate(tmp_path)
    path = tmp_path / "results" / "001.json"
    receipt = json.loads(path.read_text())
    write_private(path, {**receipt, "controls": {"empty": 0., "generic": 1.}})
    with pytest.raises(InfrastructureError, match="saved_case_receipt_changed"):
        simulate(tmp_path)


def test_lenient_judge_fails_controls_and_blocks_every_promotion(tmp_path, monkeypatch):
    def lenient(self, ref, response):
        rows = []
        for row in response.claims:
            facts = [fact.claim.id for fact in ref.facts if fact.claim.modality == row.modality]
            rows.append(Decision.whole(row, "supported" if facts else "unresolved", facts[:1],
                                       "accepts anything"))
        return Assessment(judge_id=self.identity, decisions=rows)

    monkeypatch.setattr(simulation.FakeJudge, "assess", lenient)
    report = simulate(tmp_path)["report"]
    assert not report["controls"]["pass"] and report["winner"] is None
    assert not any(row["promoted"] for row in report["decisions"].values())


def _catalog_and_commit(tmp_path, *, rules):
    media = {f"source-{index}": synthetic_source(tmp_path, index) for index in range(8)}
    manifest = [{"id": key, "path": str(path), "creator": key, "parents": [],
                 "license_url": "synthetic-local-fixture", "split": "eval"} for key, path in media.items()]
    build_catalog(manifest, tmp_path / "catalog.json")
    write_private(tmp_path / "policy.json", rules.model_dump())
    champion, strong = (content_hash({"model": name}) for name in ("champion", "strong"))
    make_commit({"catalog": str(tmp_path / "catalog.json"), "policy": str(tmp_path / "policy.json"),
                 "champion_hash": champion, "challenger_hashes": [strong],
                 "future_finalized_block": 101, "history_dir": str(tmp_path / "history"),
                 "challenge_id": "end_to_end", "output": str(tmp_path / "commit.json")})
    return media, {champion: "champion-hotkey", strong: "strong-hotkey"}


def test_prepare_renders_random_clips_then_evaluate_requires_human_references(tmp_path):
    rules = policy(judge_id=simulation.FakeJudge.identity, screen_size=2, confirmation_size=8)
    media, hotkeys = _catalog_and_commit(tmp_path, rules=rules)
    root = tmp_path / "round"
    prepared = prepare({"catalog": str(tmp_path / "catalog.json"), "policy": str(tmp_path / "policy.json"),
                        "commit": str(tmp_path / "commit.json"), "finalized_block": 101,
                        "finalized_block_hash": "a" * 64, "history_dir": str(tmp_path / "history"),
                        "source_media": {key: str(value) for key, value in media.items()},
                        "exclusions": {}, "root": str(root)})
    assert prepared["prepared"] == 8 and not prepared["excluded"]
    queue = json.loads((root / "labeling.json").read_text())
    config = {"root": str(root), "hotkeys": hotkeys,
              "evaluation_factory": "witness.benchmark.simulation:make_evaluation_adapters"}
    with pytest.raises(ValueError, match="missing_human_reference"):
        evaluate(config)
    bundle = json.loads((root / "prepared.json").read_text())
    for row in bundle["clips"]:
        index = int(row["source_id"].split("-")[-1])
        value = fixture_reference(index, row["task"]["clip_sha256"], row["task"]["duration"]).model_dump()
        write_private(root / "references" / (row["task"]["clip_sha256"] + ".json"), value)
    single = bundle["clips"][0]["task"]["clip_sha256"]
    one_annotator = json.loads((root / "references" / (single + ".json")).read_text())
    write_private(root / "references" / (single + ".json"), {**one_annotator, "annotators": ANNOTATORS[:1]})
    with pytest.raises(ValueError, match="independent_annotators"):
        evaluate(config)
    write_private(root / "references" / (single + ".json"), one_annotator)
    result = evaluate(config)
    assert result["winner"] == next(key for key, value in hotkeys.items() if value == "strong-hotkey")
    assert not result["weights_submitted"] and len(queue) == 8
    assert (tmp_path / "history" / "end_to_end.report.json").exists()


def test_archive_ingest_preserves_original_license_and_lineage_without_network(tmp_path, monkeypatch):
    import witness.benchmark.archive as acquisition

    original = synthetic_source(tmp_path, 1)
    catalogue = {"items": [{"identifier": "source-1"}], "dataset": "fixture"}
    catalogue["snapshot_hash"] = content_hash(catalogue)
    snapshot = tmp_path / "archive-snapshot.json"
    write_private(snapshot, catalogue)
    license_url = "https://creativecommons.org/licenses/by/4.0/"
    metadata = {"metadata": {"mediatype": "movies", "collection": ["opensource_movies"],
                             "licenseurl": [license_url], "creator": ["Creator B", "Creator A"]},
                "files": [{"name": "video.mp4", "size": original.stat().st_size, "length": "61"}]}
    calls = []

    def fake_request(client, url, *, output=None, expected_size=None):
        calls.append(url)
        if output is None:
            return metadata
        assert expected_size == original.stat().st_size
        output.write_bytes(original.read_bytes())

    monkeypatch.setattr(acquisition, "request", fake_request)
    monkeypatch.setattr(acquisition, "probe", lambda path: 61.)
    root = tmp_path / "archive"
    first = acquire_archive(snapshot, root, {"source-1": "eval"})
    assert first["originals"] == 1 and len(calls) == 2
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest[0]["creator"] == "Creator A | Creator B"
    assert manifest[0]["license_url"] == license_url
    assert sha256_file(root / "source-1.mp4") == sha256_file(original)
    assert not (root / "source-1.mp4").stat().st_mode & 0o077
    assert acquire_archive(snapshot, root, {"source-1": "eval"}) == first
    assert len(calls) == 2
    (root / "source-1.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="saved_archive_source_changed"):
        acquire_archive(snapshot, root, {"source-1": "eval"})


def test_annotator_drops_facts_whose_evidence_does_not_fit():
    from witness.benchmark.annotate import _fits, audio_levels
    import numpy as np
    windows = [{"index": i, "start": i * 4., "end": (i + 1) * 4.} for i in range(3)]
    packet = {"duration": 12., "frames": 24, "speech_segments": [{"start": 2., "end": 4., "text": "hello"}],
              "sound_windows": windows, "audio_levels": audio_levels(np.zeros(16000 * 12), 12.)}
    assert all(level["silent"] for level in packet["audio_levels"]) and len(packet["sound_windows"]) == 3

    def fact(modality, start, end, kind, indices, certainty="clear"):
        return {"claim": {"id": "x", "start": start, "end": end, "subject": "s", "description": "d",
                          "modality": modality}, "salience": "core", "certainty": certainty, "note": "",
                "evidence": [{"kind": kind, "indices": indices}]}
    assert _fits(fact("visual", 1., 3., "frames", [2, 4, 6]), packet) is None
    assert _fits(fact("visual", 1., 3., "frames", [20]), packet) == "cited_frame_outside_interval"
    assert _fits(fact("speech", 2., 4., "asr_segments", [0], "proposal"), packet) is None
    assert _fits(fact("speech", 0., 8., "asr_segments", [0], "proposal"), packet) == "audio_interval_outside_cited_evidence"
    assert _fits(fact("sound", 0., 12., "audio_levels", list(range(12)), "proposal"), packet) is None
    assert _fits(fact("speech", 2., 4., "frames", [4]), packet) == "wrong_evidence_kind_or_index"
    assert _fits(fact("visual", 1., 3., "frames", [2], "uncertain"), packet) == "uncertain"


def test_partial_credit_keeps_the_true_part_of_a_claim():
    from witness.benchmark.scoring import Part
    ref = reference(VISUAL)
    detailed = claim("d", description="A hand in a blue glove turns a valve")
    response = Response(claims=[detailed])
    half = Assessment(judge_id="judge:fixture", decisions=[Decision(prediction_id="d", reason="fixture", parts=[
        Part(text="a hand turns a valve", status="supported", fact_ids=["v"]),
        Part(text="the glove is blue", status="unresolved")])])
    result = score(ref, response, half, policy())
    assert result["precision"] == .5 and result["recall"] == 1.
    wrong = Assessment(judge_id="judge:fixture", decisions=[Decision(prediction_id="d", reason="fixture", parts=[
        Part(text="a hand turns a valve", status="supported", fact_ids=["v"]),
        Part(text="the glove is blue", status="contradicted", fact_ids=["v"])])])
    assert score(ref, response, wrong, policy())["precision"] == 0.
    assert wrong.decisions[0].status == "contradicted" and half.decisions[0].status == "unresolved"
