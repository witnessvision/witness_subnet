"""Provider-free end-to-end rehearsal using visibly synthetic local media."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw
from bittensor_wallet import Keypair

from witness.events import content_hash
from witness.storage import write_private
from .adjudication import MediaReview
from .contract import Case, Claim, Execution, Fact, Policy, Reference, Response, Task
from .lineage import Catalog, DrawCommit, clip_spec, select, source_from_media
from .media import PREPROCESSOR_ID, render_clip
from .round import run_round
from .scoring import Assessment, Decision, compatible

RUNTIME = content_hash({"fake_runner": 3})
PROMPT = content_hash({"synthetic_prompt": 3})
ANNOTATORS = [content_hash({"annotator": name}) for name in ("a", "b")]


def fixture_claims(index: int, duration: float) -> list[Claim]:
    return [Claim(id="card", start=0., end=duration, subject="A card",
                  description=f"Shows the number {index}", modality="text"),
            Claim(id="tone", start=0., end=duration, subject="A tone",
                  description=f"Plays at {440 + index * 50} Hz", modality="sound")]


def fixture_reference(index: int, clip_sha256: str, duration: float) -> Reference:
    return Reference(kind="human", clip_sha256=clip_sha256, duration=duration, annotators=ANNOTATORS,
                     facts=[Fact(claim=claim, salience="core") for claim in fixture_claims(index, duration)])


class FakeJudge:
    identity = "fake-exact-match-judge-3"

    def __init__(self, policy: Policy):
        self.policy = policy

    def assess(self, reference: Reference, response: Response) -> Assessment:
        decisions = []
        for claim in response.claims:
            matches = [fact.claim.id for fact in reference.facts
                       if (claim.subject, claim.description) == (fact.claim.subject, fact.claim.description)
                       and compatible(claim, fact.claim, self.policy)]
            decisions.append(Decision.whole(claim, "supported" if matches else "unresolved",
                                            matches, "synthetic_exact_match"))
        return Assessment(judge_id=self.identity, decisions=decisions)


class FakeReviewer:
    identity = "fake-blind-av-reviewer-3"

    def __init__(self, truths: set[str]):
        self.truths = truths

    def resolve(self, clip: Path, claims: list[Claim]) -> MediaReview:
        if not clip.is_file():
            raise ValueError("missing_clip")
        return MediaReview(reviewer_id=self.identity, accessed_media=True, heard_audio=True,
                           prompt_hash=PROMPT,
                           decisions={claim.id: "supported" if claim.description in self.truths
                                      else "contradicted" for claim in claims})


class FakeRunner:
    """Champion reports half the facts, ``strong`` all of them, ``broken`` invalid JSON."""

    def __init__(self, roles: dict[str, str]):
        self.roles = roles
        self.calls = []

    def __call__(self, model_id: str, task: Task, media_path: Path) -> Execution:
        if not media_path.is_file():
            raise ValueError("missing_mp4")
        self.calls.append((model_id, task.id))
        index = int(media_path.stem.split("-")[-1])
        role = self.roles[model_id]
        rows = fixture_claims(index, task.duration)
        if role == "champion":
            rows = rows[:1]
        elif role == "strong" and index == 0:
            rows += [Claim(id="false_detail", start=0., end=task.duration, subject="A card",
                           description="Shows a golden crown", modality="text"),
                     Claim(id="novel_true", start=0., end=task.duration, subject="A triangle",
                           description="A red triangle is visible", modality="visual")]
        raw = "{not json" if role == "broken" else Response(
            claims=sorted(rows, key=lambda claim: (claim.start, claim.id))).model_dump_json()
        return Execution(task_id=task.id, model_id=model_id, checkpoint_hash=model_id,
                         clip_sha256=task.clip_sha256, runtime_hash=RUNTIME,
                         preprocessing_hash=PREPROCESSOR_ID, status="ok",
                         elapsed_s=12. if role == "champion" else 8., raw=raw,
                         audio_tokens=1, video_tokens=1)


def make_evaluation_adapters(config: dict, policy: Policy) -> dict:
    """Operator factory for synthetic rounds only: first challenger strong, others broken."""
    prepared = json.loads((Path(config["root"]) / "prepared.json").read_text())
    catalog = Catalog.model_validate(prepared["catalog"])
    if (not all(source.license_url == "synthetic-local-fixture" for source in catalog.sources)
            or policy.judge_id != FakeJudge.identity):
        raise ValueError("fake_evaluator_requires_synthetic_catalog")
    commit = DrawCommit.model_validate(prepared["draw"]["commit"])
    roles = {commit.champion_hash: "champion", commit.challenger_hashes[0]: "strong",
             **{model_id: "broken" for model_id in commit.challenger_hashes[1:]}}
    reviewer = FakeReviewer({"A red triangle is visible"}) if policy.reviewer_id else None
    return {"runner": FakeRunner(roles), "judge": FakeJudge(policy), "reviewer": reviewer}


def synthetic_source(root: Path, index: int) -> Path:
    image_path = root / f"source-{index}.png"
    image = Image.new("RGB", (640, 360), "white")
    painter = ImageDraw.Draw(image)
    painter.polygon([(40, 40), (80, 40), (60, 75)], fill="red")
    for number in range(20):
        x = (index * 97 + number * 61) % 610
        y = (index * 47 + number * 83) % 330
        painter.rectangle((x, y, x + 20, y + 20), fill=(index * 23 % 255, number * 37 % 255,
                                                            (index + number) * 31 % 255))
    painter.text((280, 150), str(index), fill="black")
    image.save(image_path)
    source = root / f"source-{index}.mp4"
    if not source.exists():
        result = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error",
                                 "-loop", "1", "-framerate", "8", "-i", str(image_path),
                                 "-f", "lavfi", "-i", f"sine=frequency={440 + index * 50}:sample_rate=16000",
                                 "-t", "14", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-c:a", "aac", "-ar", "16000", "-ac", "1",
                                 "-map_metadata", "-1", str(source)],
                                capture_output=True, timeout=90)
        if result.returncode:
            raise ValueError("fake_source_render_failed")
        source.chmod(0o600)
    return source


def simulate(root: Path, *, count: int = 8) -> dict:
    """Uses no API, wallet, GPU or chain; small count keeps the rehearsal quick."""
    if count < 8:
        raise ValueError("simulation_needs_eight_groups")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    media_root = root / "media"
    media_root.mkdir(exist_ok=True, mode=0o700)
    paths = {f"source-{index}": synthetic_source(media_root, index) for index in range(count)}
    sources = [source_from_media(path, source_id=source_id, creator=f"creator-{source_id}",
                                 license_url="synthetic-local-fixture") for source_id, path in paths.items()]
    catalog = Catalog.build(sources, {source.id: "eval" for source in sources})
    reviewer = FakeReviewer({"A red triangle is visible"})
    policy = Policy(judge_id=FakeJudge.identity, reviewer_id=reviewer.identity, runtime_hash=RUNTIME,
                    preprocessing_hash=PREPROCESSOR_ID, screen_size=2, confirmation_size=count)
    roles = {content_hash({"fake_model": role}): role for role in ("champion", "strong", "broken")}
    champion, strong, broken = roles
    commit = DrawCommit(challenge_id="synthetic-round", catalog_hash=catalog.identity,
                        policy_hash=policy.identity, champion_hash=champion,
                        challenger_hashes=[strong, broken], finalized_block_after_freeze=101,
                        used_groups_hash=content_hash([]))
    draw = select(commit, catalog, policy, block_number=101, block_hash="a" * 64, used_groups=set())
    source_by_id = {source.id: source for source in sources}
    cases = []
    for group in draw.groups:
        spec = clip_spec(draw, group, catalog, policy)
        index = int(spec.source_id.split("-")[-1])
        clip = media_root / f"clip-{index}.mp4"
        digest = render_clip(paths[spec.source_id], clip, start=spec.start, duration=spec.duration)
        task = Task(id=f"task-{index}", clip_sha256=digest, duration=spec.duration)
        cases.append(Case(task=task, source_group=group, source_id=spec.source_id,
                          source_sha256=source_by_id[spec.source_id].original_sha256,
                          source_path=str(paths[spec.source_id]), source_start=spec.start,
                          media_path=str(clip), reference=fixture_reference(index, digest, spec.duration)))
    runner = FakeRunner(roles)
    miners = [Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob", "//Eve")]
    hotkeys = {model_id: key.ss58_address for model_id, key in zip(roles, miners)}
    report = run_round(cases=cases, catalog=catalog, draw=draw, policy=policy, hotkeys=hotkeys,
                       runner=runner, judge=FakeJudge(policy), reviewer=reviewer, private_root=root)
    result = {"simulation_only": True, "provider_calls_paid": 0, "groups": count,
              "roles": {role: model_id for model_id, role in roles.items()},
              "runner_calls": len(runner.calls), "report": report}
    write_private(root / "simulation-result.json", result)
    return result
