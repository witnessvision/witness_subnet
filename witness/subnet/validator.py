"""Witness validator round construction, scoring, aggregation, and CLI."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import random
import secrets
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import typer
import uvicorn

from witness import freeze
from witness.gen import generate
from witness.recompose.generator import generate_recomposition
from witness.score import score_reconstruction
from witness.score.scorer import DEFAULT_Q_MIN
from witness.tools.server import Budget, create_app

from .chain import BittensorChainAdapter, ChainAdapter, InMemoryChainAdapter, MinerEndpoint, WeightSubmission
from .protocol import WitnessTask


DEFAULT_BUDGET: dict[str, int | float] = {
    "visual_tokens": 100_000,
    "audio_seconds": 120.0,
    "transcript_chars": 20_000,
}
RELATIVE_GATE_FLOOR = 0.35
RELATIVE_GATE_RATIO = 0.8
ZERO_Q_MIN = {1: 0.0, 2: 0.0, 3: 0.0}
SCORED_RECONSTRUCTION_FIELDS = (
    "events",
    "dialogue",
    "shots",
    "on_screen_text",
    "audio_events",
    "intentional_errors",
    "qa",
)
log = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_LOCK = REPOSITORY_ROOT / "data" / "benchmark_v1.5.lock.json"
DEFAULT_LOCKED_SCENE_ROOT = REPOSITORY_ROOT / "data/scenes/hidden-v15"


@dataclass(slots=True)
class ValidatorConfig:
    round_root: Path = Path("rounds")
    scene_count: int = 3
    programmatic_share: float = 0.25
    tiers: tuple[int, ...] = (1, 2, 3)
    pool_manifest: Path | None = None
    source_scenes: tuple[Path, ...] = ()
    budget: dict[str, int | float] = field(default_factory=lambda: dict(DEFAULT_BUDGET))
    deadline_s: float = 180.0
    ema_alpha: float = 0.3
    tool_host: str = "0.0.0.0"
    tool_port: int = 8765
    tool_public_url: str | None = None
    burn_uid: int | None = 0
    burn_rate: float = 0.0
    round_interval_s: float = 60.0
    benchmark_lock: Path | None = DEFAULT_BENCHMARK_LOCK
    locked_scene_root: Path = DEFAULT_LOCKED_SCENE_ROOT
    allow_unlocked: bool = False
    score_version: str = "1.5"
    transcript_source: str = "legacy_labels"

    def validate(self) -> None:
        if self.score_version not in {"1.5", "1.6-candidate", "1.7-candidate", "1.8"}:
            raise ValueError("unknown score_version")
        if self.score_version != "1.5" and not self.allow_unlocked:
            raise ValueError("candidate scoring requires explicit allow_unlocked diagnostic mode")
        if self.transcript_source not in {"legacy_labels", "asr", "none"}:
            raise ValueError("unknown transcript_source")
        if self.transcript_source == "asr" and not self.source_scenes:
            raise ValueError("ASR rounds require source_scenes with precomputed observations")
        if self.scene_count < 1:
            raise ValueError("scene_count must be at least one")
        if not 0 <= self.programmatic_share <= 1:
            raise ValueError("programmatic_share must be between zero and one")
        if not self.tiers or any(tier not in (1, 2, 3) for tier in self.tiers):
            raise ValueError("tiers must contain only 1, 2, or 3")
        if self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0 <= self.burn_rate <= 1:
            raise ValueError("burn_rate must be between zero and one")
        if self.tool_port < 0 or self.tool_port > 65535:
            raise ValueError("tool_port must be between 0 and 65535")


@dataclass(frozen=True, slots=True)
class RoundScene:
    scene_id: str
    directory: Path
    truth: dict[str, Any]
    seed: int
    kind: str


def load_and_verify_benchmark_lock(
    lock_path: Path | None,
    scene_root: Path,
    *,
    allow_unlocked: bool,
) -> dict[str, Any] | None:
    """Load the optional v1.5 lock and refuse altered code or hidden scenes."""
    if lock_path is None:
        return None
    if not lock_path.is_file():
        if allow_unlocked:
            return None
        raise RuntimeError("benchmark lock missing; diagnostic use requires allow_unlocked")
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("lock root must be an object")
        roots = [scene_root]
        # The canonical freeze covers both partitions; only hidden is sampled.
        if lock_path.resolve() == DEFAULT_BENCHMARK_LOCK.resolve():
            roots.append(REPOSITORY_ROOT / "data/scenes/dev-v15")
        problems = freeze.verify_lock(value, roots)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        value = None
        problems = [f"invalid benchmark lock: {type(exc).__name__}: {exc}"]
    if problems and not allow_unlocked:
        preview = "; ".join(problems[:5])
        if len(problems) > 5:
            preview += f"; and {len(problems) - 5} more"
        raise RuntimeError(f"benchmark v1.5 lock verification failed: {preview}")
    if problems:
        log.warning(
            "Continuing with an unlocked benchmark because allow_unlocked is set (%d problems)",
            len(problems),
        )
    return value


class InProcessToolServer:
    """Run the existing metered FastAPI app in a bounded background thread."""

    def __init__(
        self,
        scene_root: Path,
        log_dir: Path,
        *,
        host: str,
        port: int,
        public_url: str | None,
        transcript_source: str = "legacy_labels",
    ) -> None:
        self.scene_root = scene_root
        self.log_dir = log_dir
        self.host = host
        self.port = port
        self.public_url = public_url.rstrip("/") if public_url else None
        self.transcript_source = transcript_source
        self.server: uvicorn.Server | None = None
        self.thread: threading.Thread | None = None
        self.socket: socket.socket | None = None
        self.internal_url = ""
        self.advertised_url = ""

    def __enter__(self) -> "InProcessToolServer":
        app = create_app(self.scene_root, log_dir=self.log_dir, allow_session_creation=False,
                         transcript_source=self.transcript_source)
        self.store = app.state.store
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(2048)
        actual_port = int(listener.getsockname()[1])
        config = uvicorn.Config(app, log_level="warning", lifespan="off")
        self.server = uvicorn.Server(config)
        self.socket = listener
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [listener]},
            name="witness-tool-server",
            daemon=True,
        )
        self.thread.start()
        deadline = time.monotonic() + 10.0
        while not self.server.started and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self.server.started:
            self.__exit__(None, None, None)
            raise RuntimeError("Witness tool server did not start")
        self.internal_url = f"http://127.0.0.1:{actual_port}"
        self.advertised_url = self.public_url or self.internal_url
        return self

    def __exit__(self, *_args: object) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=10)
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass


def seed_commitment(scene_id: str, seed: int, nonce: str) -> str:
    """Bind a private seed without enabling dictionary lookup of fixture seeds."""
    material = f"witness:seed-commitment:v1:{scene_id}:{seed}:{nonce}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def public_task_spec(scene: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "duration": float(scene["duration"]),
        "fps": int(scene["fps"]),
        "tier": int(scene["difficulty"]),
        "schema_version": str(scene.get("schema_version", "1.0")),
        "qa": [
            {"id": str(item["id"]), "q": str(item["q"])}
            for item in scene.get("qa", [])
        ],
    }


def reconstruction_hash(reconstruction: Mapping[str, Any]) -> str:
    scored_projection = {
        key: reconstruction.get(key, {} if key == "qa" else [])
        for key in SCORED_RECONSTRUCTION_FIELDS
    }
    encoded = json.dumps(
        scored_projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_relative_gate(records: list[dict[str, Any]]) -> None:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scene.setdefault(str(record["scene_id"]), []).append(record)
    for scene_records in by_scene.values():
        responded = [record for record in scene_records if record["responded"]]
        best_quality = max((float(record["quality"]) for record in responded), default=0.0)
        threshold = max(RELATIVE_GATE_FLOOR, RELATIVE_GATE_RATIO * best_quality)
        for record in scene_records:
            absolute_minimum = DEFAULT_Q_MIN[int(record.get("tier", 1))]
            effective_threshold = max(threshold, absolute_minimum)
            passed = bool(record["responded"] and float(record["quality"]) >= effective_threshold)
            record["gate"] = {
                "floor": RELATIVE_GATE_FLOOR,
                "ratio": RELATIVE_GATE_RATIO,
                "best_quality": round(best_quality, 12),
                "threshold": round(effective_threshold, 12),
                "absolute_minimum": absolute_minimum,
                "passed": passed,
            }
            record["score_before_duplicates"] = round(
                float(record["quality"]) * float(record["efficiency_factor"]) if passed else 0.0,
                12,
            )


def apply_duplicate_sharing(records: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if not record["responded"]:
            continue
        digest = reconstruction_hash(record["reconstruction"])
        record["reconstruction_hash"] = digest
        groups.setdefault((str(record["scene_id"]), digest), []).append(record)
    for group in groups.values():
        count = len({int(record["uid"]) for record in group})
        for record in group:
            record["duplicate_count"] = count
            record["score"] = round(float(record["score_before_duplicates"]) / count, 12)
    for record in records:
        record.setdefault("duplicate_count", 0)
        record.setdefault("score", 0.0)


def aggregate_miner_scores(
    uids: Iterable[int],
    records: list[dict[str, Any]],
    scene_count: int,
    previous_ema: Mapping[str, float],
    alpha: float,
) -> tuple[dict[int, float], dict[int, float], set[int]]:
    totals = {int(uid): 0.0 for uid in uids}
    responders: set[int] = set()
    for record in records:
        uid = int(record["uid"])
        totals[uid] = totals.get(uid, 0.0) + float(record["score"])
        if record["responded"]:
            responders.add(uid)
    round_scores = {uid: totals.get(uid, 0.0) / scene_count for uid in totals}
    ema: dict[int, float] = {}
    for uid, score in round_scores.items():
        prior = previous_ema.get(str(uid))
        ema[uid] = score if prior is None else alpha * score + (1.0 - alpha) * float(prior)
    return round_scores, ema, responders


def build_weight_vector(
    uids: list[int],
    ema_scores: Mapping[int, float],
    responders: set[int],
    *,
    burn_uid: int | None,
    burn_rate: float,
) -> list[float]:
    weights = [0.0] * len(uids)
    uid_to_index = {uid: index for index, uid in enumerate(uids)}
    positive = {
        uid: max(0.0, float(ema_scores.get(uid, 0.0)))
        for uid in uids
        if uid in responders and float(ema_scores.get(uid, 0.0)) > 0
    }
    total = sum(positive.values())
    valid_burn = burn_uid is not None and burn_uid in uid_to_index
    if total <= 0:
        if valid_burn:
            weights[uid_to_index[int(burn_uid)]] = 1.0
        return weights
    miner_share = 1.0 - burn_rate if valid_burn else 1.0
    for uid, score in positive.items():
        weights[uid_to_index[uid]] = miner_share * score / total
    if valid_burn and burn_rate:
        weights[uid_to_index[int(burn_uid)]] += burn_rate
    return weights


class WitnessValidator:
    def __init__(
        self,
        chain: ChainAdapter,
        config: ValidatorConfig,
        *,
        dry_scene_lookup: dict[str, Path] | None = None,
    ) -> None:
        config.validate()
        self.chain = chain
        self.config = config
        self.dry_scene_lookup = dry_scene_lookup
        self.scorer = score_reconstruction
        if config.score_version == "1.6-candidate":
            from witness.score_v16 import score_reconstruction as scorer
            self.scorer = scorer
        elif config.score_version == "1.7-candidate":
            from witness.score_v17 import score_reconstruction as scorer
            self.scorer = scorer
        elif config.score_version == "1.8":
            from witness.score_v18 import score_reconstruction as scorer
            self.scorer = scorer
        self.scoring_identity = {
            "version": config.score_version,
            "transcript_source": config.transcript_source,
            "code_sha256": {
                "scorer": hashlib.sha256(Path(inspect.getfile(self.scorer)).read_bytes()).hexdigest(),
                "validator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                **{name: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
                   for name, path in (("contract", "witness/contract.py"), ("metering", "witness/tools/metering.py"))},
            },
            "mode": "diagnostic" if config.allow_unlocked or config.score_version != "1.5" else "legacy",
            "lock_verification": "required" if config.benchmark_lock is not None and not config.allow_unlocked else "disabled_or_diagnostic",
        }
        self.benchmark_lock = load_and_verify_benchmark_lock(
            config.benchmark_lock,
            config.locked_scene_root,
            allow_unlocked=config.allow_unlocked,
        )
        self.scoring_identity["benchmark_lock_sha256"] = (
            hashlib.sha256(config.benchmark_lock.read_bytes()).hexdigest()
            if config.benchmark_lock is not None and config.benchmark_lock.is_file() else None
        )

    async def run_round(self) -> dict[str, Any]:
        previous = self._load_ema()
        block_hash = self.chain.current_block_hash()
        round_id = self._round_id(block_hash)
        round_dir = self.config.round_root.resolve() / f"round_{round_id}"
        private_root = round_dir / "private" / "scenes"
        log_dir = round_dir / "private" / "session_logs"
        private_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        private_root.parent.chmod(0o700)
        scenes = self._prepare_scenes(private_root, block_hash)
        commitment_nonce = secrets.token_hex(32)
        commitments = {
            scene.scene_id: seed_commitment(scene.scene_id, scene.seed, commitment_nonce)
            for scene in scenes
        }
        nonce_path = private_root.parent / "seed-commitment.json"
        self._write_json(nonce_path, {"nonce": commitment_nonce})
        nonce_path.chmod(0o600)
        self._write_json(round_dir / "commitments.json", {
            "round_id": round_id, "scheme": "witness:seed-commitment:v1",
            "scenes": commitments,
        })
        if self.dry_scene_lookup is not None:
            self.dry_scene_lookup.update({scene.scene_id: scene.directory for scene in scenes})
        endpoints = self.chain.miner_endpoints()
        records: list[dict[str, Any]] = []
        started_at = datetime.now(timezone.utc).isoformat()

        with InProcessToolServer(
            private_root,
            log_dir,
            host=self.config.tool_host,
            port=self.config.tool_port,
            public_url=self.config.tool_public_url,
            transcript_source=self.config.transcript_source,
        ) as tool_server:
            for scene in scenes:
                ordered = list(endpoints)
                random.Random(f"{commitment_nonce}:{scene.scene_id}").shuffle(ordered)
                for endpoint in ordered:
                    session_id = tool_server.store.create(
                        scene.scene_id, Budget(**self.config.budget)
                    ).session_id
                    task = WitnessTask(
                        task_id=f"{round_id}:{scene.scene_id}:{endpoint.uid}",
                        tool_base_url=tool_server.advertised_url,
                        session_id=session_id,
                        scene_id=scene.scene_id,
                        seed_commitment=commitments[scene.scene_id],
                        budget=dict(self.config.budget),
                        task_spec=public_task_spec(scene.truth),
                        deadline_s=self.config.deadline_s,
                    )
                    response: WitnessTask | None = None
                    error: str | None = None
                    try:
                        response = await self.chain.query(
                            endpoint, task, timeout=self.config.deadline_s
                        )
                    except Exception as exc:  # one miner cannot abort the round
                        error = f"{type(exc).__name__}: {exc}"
                    with tool_server.store.lock:
                        completed_session = tool_server.store.sessions.pop(session_id)
                        cost = completed_session.cost.as_dict()
                    response_status = (
                        response.trace_summary.get("status")
                        if response is not None and isinstance(response.trace_summary, dict)
                        else None
                    )
                    responded = response is not None and response_status not in {
                        "busy",
                        "deadline_exceeded",
                        "error",
                        "degraded",
                    }
                    if response is not None and not responded and error is None:
                        error = f"miner status: {response_status}"
                    reconstruction = (
                        response.reconstruction
                        if response is not None and isinstance(response.reconstruction, dict)
                        else {}
                    )
                    report = self.scorer(
                        scene.truth,
                        reconstruction,
                        cost,
                        q_min=ZERO_Q_MIN,
                    )
                    records.append(
                        {
                            "uid": endpoint.uid,
                            "hotkey": endpoint.hotkey,
                            "scene_id": scene.scene_id,
                            "tier": int(scene.truth["difficulty"]),
                            "session_id": session_id,
                            "responded": responded,
                            "score_version": self.config.score_version,
                            "error": error,
                            "quality": report["quality"],
                            "efficiency_factor": report["cost"]["efficiency_factor"],
                            "cost": cost,
                            "family_scores": report["family_scores"],
                            "diagnostics": report.get("diagnostics", []),
                            "reconstruction": reconstruction,
                            "trace_summary": response.trace_summary if response else None,
                        }
                    )

        apply_relative_gate(records)
        apply_duplicate_sharing(records)
        uids = [endpoint.uid for endpoint in endpoints]
        round_scores, ema_scores, responders = aggregate_miner_scores(
            uids, records, len(scenes), previous, self.config.ema_alpha
        )
        weights = build_weight_vector(
            uids,
            ema_scores,
            responders,
            burn_uid=self.config.burn_uid,
            burn_rate=self.config.burn_rate,
        )
        artifact = {
            "schema_version": "1.0",
            "scoring_identity": self.scoring_identity,
            "observation_budget": dict(self.config.budget),
            "round_id": round_id,
            "block_hash": block_hash,
            "validator_hotkey": self.chain.validator_hotkey,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "seed_commitment_nonce_revealed": commitment_nonce,
            "scene_seeds_revealed": [
                {
                    "scene_id": scene.scene_id,
                    "seed": scene.seed,
                    "seed_commitment": commitments[scene.scene_id],
                    "kind": scene.kind,
                    "tier": int(scene.truth["difficulty"]),
                    "input_sha256": {
                        name: hashlib.sha256((scene.directory / name).read_bytes()).hexdigest()
                        for name in ("scene.json", "video.mp4", "observations/transcript.json")
                        if (scene.directory / name).is_file()
                    },
                }
                for scene in scenes
            ],
            "miners": {
                str(uid): {
                    "round_score": round(round_scores[uid], 12),
                    "ema_score": round(ema_scores[uid], 12),
                    "responded": uid in responders,
                    "scenes": [record for record in records if int(record["uid"]) == uid],
                }
                for uid in uids
            },
            "weights": {"uids": uids, "values": [round(value, 12) for value in weights]},
            "weight_submission": WeightSubmission("prepared").as_dict(),
            "ema_updated": False,
        }
        # Preserve the scored round and submission intent before network I/O.
        # A crash here leaves an explicit unresolved intent, not a lost round.
        self._write_json(round_dir / "round.json", artifact)
        try:
            submission = self.chain.set_weights(uids, weights)
            if not isinstance(submission, WeightSubmission):
                raise TypeError("chain adapter returned no submission evidence")
        except Exception as exc:
            # A transport exception may follow broadcast: do not label it rejected
            # or automatically resubmit the same signed action.
            submission = WeightSubmission("unknown", error_type=type(exc).__name__)
        artifact["weight_submission"] = submission.as_dict()
        self._write_json(round_dir / "round.json", artifact)
        # EMA describes scored observations, independently of chain acceptance.
        self._save_ema(ema_scores)
        artifact["ema_updated"] = True
        self._write_json(round_dir / "round.json", artifact)
        return artifact

    async def run_forever(self) -> None:
        while True:
            try:
                await self.run_round()
            except Exception:
                log.exception("Witness round failed; retrying after the configured interval")
            await asyncio.sleep(self.config.round_interval_s)

    def _round_id(self, block_hash: str) -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = hashlib.sha256(
            f"{block_hash}:{self.chain.validator_hotkey}:{now}".encode("utf-8")
        ).hexdigest()[:10]
        return f"{now}-{suffix}"

    def _prepare_scenes(self, root: Path, block_hash: str) -> list[RoundScene]:
        scenes: list[RoundScene] = []
        source_scenes = list(self.config.source_scenes)
        programmatic_indexes = self._programmatic_indexes(block_hash)
        locked_sources = self._locked_sources(block_hash)
        locked_indexes: dict[int, int] = {}
        for index in range(self.config.scene_count):
            seed = secrets.randbits(64)
            tier = self.config.tiers[index % len(self.config.tiers)]
            scene_token = secrets.token_hex(12)
            destination = root / f"scene_{scene_token}"
            if source_scenes:
                source = source_scenes[index % len(source_scenes)].resolve()
                if not (source / "scene.json").is_file() or not (source / "video.mp4").is_file():
                    raise ValueError(f"invalid source scene directory: {source}")
                destination.mkdir(parents=True)
                shutil.copy2(source / "scene.json", destination / "scene.json")
                shutil.copy2(source / "video.mp4", destination / "video.mp4")
                if self.config.transcript_source == "asr":
                    (destination / "observations").mkdir()
                    shutil.copy2(source / "observations/transcript.json", destination / "observations/transcript.json")
                kind = "fixture"
            elif index in programmatic_indexes:
                generate(seed, tier, destination)
                kind = "programmatic"
            elif locked_sources:
                candidates = locked_sources.get(tier) or [
                    source
                    for locked_tier in sorted(locked_sources)
                    for source in locked_sources[locked_tier]
                ]
                locked_index = locked_indexes.get(tier, 0)
                source = candidates[locked_index % len(candidates)]
                locked_indexes[tier] = locked_index + 1
                destination.mkdir(parents=True)
                shutil.copy2(source / "scene.json", destination / "scene.json")
                shutil.copy2(source / "video.mp4", destination / "video.mp4")
                kind = "locked_hidden"
            else:
                if self.config.pool_manifest is None:
                    raise ValueError("pool_manifest is required for recomposed scenes")
                generate_recomposition(seed, tier, self.config.pool_manifest, destination)
                kind = "recomposed"
            truth = json.loads((destination / "scene.json").read_text(encoding="utf-8"))
            scenes.append(RoundScene(destination.name, destination, truth, int(truth["seed"]), kind))
        return scenes

    def _locked_sources(self, block_hash: str) -> dict[int, list[Path]]:
        if not self.benchmark_lock:
            return {}
        entries = self.benchmark_lock.get("scenes", {})
        if not isinstance(entries, dict):
            return {}
        scene_root = self.config.locked_scene_root.resolve()
        sources: dict[int, list[Path]] = {}
        for scene_id, entry in sorted(entries.items()):
            source = scene_root / scene_id
            if not (source / "scene.json").is_file() or not (source / "video.mp4").is_file():
                continue
            tier = int(entry.get("tier", 0)) if isinstance(entry, dict) else 0
            sources.setdefault(tier, []).append(source)
        for tier, tier_sources in sources.items():
            random.Random(
                f"{block_hash}:{self.chain.validator_hotkey}:locked:{tier}"
            ).shuffle(tier_sources)
        return sources

    def _programmatic_indexes(self, block_hash: str) -> set[int]:
        count = int(round(self.config.scene_count * self.config.programmatic_share))
        if self.config.programmatic_share > 0:
            count = max(1, count)
        if self.config.programmatic_share < 1 and self.config.scene_count > 1:
            count = min(self.config.scene_count - 1, count)
        indexes = list(range(self.config.scene_count))
        random.Random(f"{block_hash}:{self.chain.validator_hotkey}:mix").shuffle(indexes)
        return set(indexes[:count])

    @property
    def _ema_path(self) -> Path:
        return self.config.round_root.resolve() / "ema.json"

    def _load_ema(self) -> dict[str, float]:
        if not self._ema_path.is_file():
            return {}
        raw = json.loads(self._ema_path.read_text(encoding="utf-8"))
        scores = raw.get("scores", {}) if isinstance(raw, dict) else {}
        identity = raw.get("scoring_identity") if isinstance(raw, dict) else None
        legacy = self.config.score_version == "1.5" and self.config.transcript_source == "legacy_labels"
        if scores and identity != self.scoring_identity and not (identity is None and legacy):
            raise ValueError("EMA scoring identity differs; use a new round_root for the new scoring regime")
        return {str(uid): float(value) for uid, value in scores.items()}

    def _save_ema(self, scores: Mapping[int, float]) -> None:
        self._write_json(
            self._ema_path,
            {
                "schema_version": "1.0",
                "alpha": self.config.ema_alpha,
                "scoring_identity": self.scoring_identity,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "scores": {str(uid): value for uid, value in scores.items()},
            },
        )

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _parse_tiers(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item not in (1, 2, 3) for item in result):
        raise typer.BadParameter("tiers must be a comma-separated subset of 1,2,3")
    return result


app = typer.Typer(add_completion=False, help="Run the Witness validator")


@app.command()
def run(
    netuid: int = typer.Option(1, envvar="WITNESS_NETUID"),
    network: str = typer.Option("finney", envvar="WITNESS_NETWORK"),
    wallet_name: str = typer.Option("default", "--wallet", envvar="WITNESS_WALLET"),
    wallet_hotkey: str = typer.Option("default", "--hotkey", envvar="WITNESS_HOTKEY"),
    wallet_path: str | None = typer.Option(None, envvar="WITNESS_WALLET_PATH"),
    scenes: int = typer.Option(3, min=1),
    programmatic_share: float = typer.Option(0.25, min=0, max=1),
    tiers: str = typer.Option("1,2,3"),
    pool_manifest: Path | None = typer.Option(None, exists=True, dir_okay=False),
    scene: list[Path] | None = typer.Option(None, exists=True, file_okay=False),
    round_root: Path = typer.Option(Path("rounds")),
    tool_host: str = typer.Option("0.0.0.0"),
    tool_port: int = typer.Option(8765, min=0, max=65535),
    tool_public_url: str | None = typer.Option(None, envvar="WITNESS_TOOL_PUBLIC_URL"),
    deadline_s: float = typer.Option(180.0, min=1),
    ema_alpha: float = typer.Option(0.3, min=0.000001, max=1),
    burn_uid: int | None = typer.Option(0),
    burn_rate: float = typer.Option(0.0, min=0, max=1),
    interval_s: float = typer.Option(60.0, min=0),
    benchmark_lock: Path | None = typer.Option(DEFAULT_BENCHMARK_LOCK, dir_okay=False),
    locked_scene_root: Path = typer.Option(DEFAULT_LOCKED_SCENE_ROOT, file_okay=False),
    allow_unlocked: bool = typer.Option(
        False,
        help="Continue despite a present benchmark lock mismatch",
    ),
    dry_run: bool = typer.Option(False, help="Use the local base miner and no chain"),
    score_version: str = typer.Option("1.5", help="1.5, 1.6-candidate, 1.7-candidate or 1.8"),
    transcript_source: str = typer.Option("legacy_labels", help="legacy_labels, asr or none"),
    once: bool = typer.Option(False, help="Run one round and exit"),
) -> None:
    """Run continuous rounds, or one complete local round with --dry-run."""
    source_scenes = tuple(scene or ())
    if dry_run and not source_scenes:
        source_scenes = (Path("data/scenes/synthetic/scene_101"), Path("data/scenes/synthetic/scene_202"))
    config = ValidatorConfig(
        round_root=round_root,
        scene_count=scenes,
        programmatic_share=1.0 if dry_run and not pool_manifest else programmatic_share,
        tiers=_parse_tiers(tiers),
        pool_manifest=pool_manifest,
        source_scenes=source_scenes,
        deadline_s=deadline_s,
        ema_alpha=ema_alpha,
        tool_host="127.0.0.1" if dry_run else tool_host,
        tool_port=0 if dry_run else tool_port,
        tool_public_url=None if dry_run else tool_public_url,
        burn_uid=burn_uid,
        burn_rate=burn_rate,
        round_interval_s=interval_s,
        benchmark_lock=benchmark_lock,
        locked_scene_root=locked_scene_root,
        allow_unlocked=allow_unlocked,
        score_version=score_version,
        transcript_source=transcript_source,
    )
    chain: ChainAdapter
    lookup: dict[str, Path] | None = None
    if dry_run:
        from .miner import WitnessMiner

        fake = InMemoryChainAdapter()
        base = WitnessMiner()
        fake.add_miner(0, base.forward)
        chain = fake
    else:
        if config.tool_host in {"0.0.0.0", "::"} and not config.tool_public_url:
            raise typer.BadParameter("--tool-public-url is required for a public bind")
        chain = BittensorChainAdapter(
            netuid=netuid,
            network=network,
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
            wallet_path=wallet_path,
        )
    validator = WitnessValidator(chain, config, dry_scene_lookup=lookup)
    artifact = asyncio.run(validator.run_round()) if dry_run or once else None
    if artifact is None:
        asyncio.run(validator.run_forever())
    else:
        typer.echo(json.dumps({"round_id": artifact["round_id"], "weights": artifact["weights"]}))


if __name__ == "__main__":
    app()
