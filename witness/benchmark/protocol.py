"""Mainnet v2 wire contract. Consensus needs only finalized chain data.

Result commitments are 126 ASCII bytes (91 binary bytes, unpadded base64url).
Scores are unsigned 16-bit fixed point; consensus uses these exact values.
Large receipts and media are optional audit/display data, never consensus input.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

from witness.events import content_hash

RESULT_PREFIX = "wr2|"
MAX_COMMITMENT_BYTES = 128
SCALE = 65535
WINDOW_EPOCHS = 2
# Explicit mainnet transition: windows 0..8 retain the finalized v2 policy.
MEDIA_RECOVERY_WINDOW = 9
TEN_VIDEO_WINDOW = 12
RESUME_WINDOW = 13
TEN_VIDEO_POLICY = "3caf2020574fcc790fbe1d14dca4c9749f6916edc87a3f66217cb9c258a64341"
FIVE_VIDEO_POLICY = "4623a70ac9c406d8af341390cb25e7d22b29cada7b1b78929d6303cbc27bd239"
LEGACY_POLICY = "d063b9d0f4bd2d659f175823c3d720be32b2d7f7d7311dd07514fa5ad3ddd918"
BURN_FRACTION = .7
CONTROLS_OK, REJECTED, BASELINE, EARLY_STOP = 1, 2, 4, 8
_RESULT = struct.Struct(">IH32s12s32s4HB")


def encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def decode_bytes(value: str, size: int) -> bytes:
    try:
        data = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid_commitment_encoding") from exc
    if len(data) != size or encode_bytes(data) != value:
        raise ValueError("noncanonical_commitment")
    return data


def quantize(value: float) -> int:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid_score")
    return int(math.floor(value * SCALE + .5))


def weight_vector(king: dict | None, snapshot: dict) -> dict:
    """One policy for replay, live submission and display; never guess a burn UID."""
    burn = snapshot.get('burn_uid')
    uid = snapshot['uids'].get(king['hotkey']) if king else None
    king = king if uid is not None else None
    allocations = {}
    if burn is not None:
        allocations[burn] = BURN_FRACTION if king else 1.
        if king:
            allocations[uid] = allocations.get(uid, 0.) + (1. - BURN_FRACTION)
    pairs = sorted(allocations.items())
    return {'king': king, 'uids': [u for u, _ in pairs], 'weights': [w for _, w in pairs],
            'burn_uid': burn, 'king_uid': uid,
            'burn_fraction': BURN_FRACTION if king else 1.,
            'king_fraction': 1. - BURN_FRACTION if king else 0.,
            'source': 'burn_unavailable' if burn is None else 'scores' if king else 'burn_no_king'}


def weights_match(observed, vector: dict) -> bool:
    """Compare normalized proportions, not SDK max-scaled raw u16 values."""
    expected = dict(zip(vector['uids'], vector['weights']))
    actual = dict(observed or [])
    if not expected or set(actual) != set(expected) or len(actual) != len(observed or []):
        return False
    total, target = sum(actual.values()), sum(expected.values())
    return total > 0 and target > 0 and all(
        abs(actual[u] / total - w / target) <= 2 / SCALE for u, w in expected.items())


@dataclass(frozen=True)
class Result:
    window: int
    uid: int
    model_id: str
    policy: str
    evidence: str
    quality: int
    reward: int
    king_quality: int
    king_reward: int
    flags: int = CONTROLS_OK

    def __post_init__(self):
        if not 0 <= self.window < 2**32 or not 0 <= self.uid < 2**16:
            raise ValueError("invalid_window_or_uid")
        for value, length in ((self.model_id, 64), (self.policy, 24), (self.evidence, 64)):
            if len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid_digest")
        if self.flags & ~15 or any(type(x) is not int or not 0 <= x <= SCALE for x in
                                 (self.quality, self.reward, self.king_quality, self.king_reward)):
            raise ValueError("invalid_result")
        if self.reward > self.quality or self.king_reward > self.king_quality:
            raise ValueError("reward_exceeds_quality")
        if self.flags & REJECTED and (self.quality or self.reward):
            raise ValueError('rejected_result_must_score_zero')
        if self.flags & EARLY_STOP and (self.flags & (BASELINE | REJECTED) or not self.flags & CONTROLS_OK):
            raise ValueError('invalid_early_stop_flags')

    @property
    def commitment(self) -> str:
        raw = _RESULT.pack(self.window, self.uid, bytes.fromhex(self.model_id), bytes.fromhex(self.policy),
                           bytes.fromhex(self.evidence), self.quality, self.reward,
                           self.king_quality, self.king_reward, self.flags)
        return RESULT_PREFIX + encode_bytes(raw)

    @classmethod
    def parse(cls, value: str) -> "Result":
        if not value.startswith(RESULT_PREFIX) or len(value.encode()) > MAX_COMMITMENT_BYTES:
            raise ValueError("not_a_v2_result")
        w, uid, model, policy, evidence, q, r, kq, kr, flags = _RESULT.unpack(decode_bytes(value[4:], _RESULT.size))
        return cls(w, uid, model.hex(), policy.hex(), evidence.hex(), q, r, kq, kr, flags)

    def public(self) -> dict:
        return {**asdict(self), **{key: getattr(self, key) / SCALE for key in
                                 ("quality", "reward", "king_quality", "king_reward")}}


def policy_identity() -> str:
    """Same on CPU followers and evaluators. Never includes wallet, GPU ID or seed."""
    here = Path(__file__).parent
    files = ("reward.py", "scoring.py", "contract.py", "adjudication.py", "judge.py", "annotate.py",
             "media.py", "pod_runtime.py", "pod_audio.py", "pod_setup.sh", "protocol.py", "ledger.py",
             "submission.py", "triggers.py", "evaluator.py", "pool.py", "duel.py", "runner.py", "stopping.py",
             "execution.py")
    catalog = here / "data" / "catalogue-v2.json"
    hashes = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in files}
    hashes.update({str(p.relative_to(here)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((here / "pod_env").glob("*.txt"))})
    return content_hash({"protocol": "witness-mainnet-2", "files": hashes,
                         "catalogue": hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.exists() else None,
                         "media_recovery": "deterministic-reserve-v1-from-window-9",
                         "ten_video_window": TEN_VIDEO_WINDOW, "five_video_policy": FIVE_VIDEO_POLICY,
                         "resume_window": RESUME_WINDOW, "ten_video_policy": TEN_VIDEO_POLICY,
                         "execution": "durable-first-answer-per-clip-v1", "attempt_budget_s": 1800,
                         "window_epochs": WINDOW_EPOCHS, "videos": 10, "clips": 2, "references": 2,
                         "clip_timeout_s": 60,
                         "early_stop": "finite-batch-90-one-look-after-three-generalized",
                         "burn_fraction": BURN_FRACTION,
                         "label_model": "gpt-6-luna", "judge_model": "gpt-5.6-terra",
                         "margin": .02, "controls_max": .05, "quality_floor": .05})


def decide(results: list[dict], *, window: int, policy: str, king: dict | None,
           candidates: dict[str, dict], snapshot: dict) -> dict:
    """Paired stake-weighted improvement; duplicate evaluators never multiply stake.

    candidates maps hotkey-bound challenge digests to submissions admitted before opening.
    A baseline record is required from the same evaluator and window. First terminal
    result wins; later amendments cannot provide another attempt with the same hotkey.
    """
    records, baselines, used, terminal_panels = {}, {}, set(), {}
    for row in sorted(results, key=lambda r: (r["block"], r["hotkey"], r["value"])):
        try:
            result = Result.parse(row["value"])
        except ValueError:
            continue
        voter = row["hotkey"]
        if (result.window != window or result.policy != policy[:24]
                or voter not in snapshot["validators"] or not snapshot["validators"][voter] > 0):
            continue
        if result.flags & BASELINE:
            baselines.setdefault((voter, result.model_id), result)
            continue
        candidate = candidates.get(result.model_id)
        if candidate is None or candidate["uid"] != result.uid:
            continue
        key = (voter, candidate["hotkey"])
        if key in used:
            continue
        if not result.flags & EARLY_STOP:
            used.add(key)
        if not result.flags & CONTROLS_OK:
            continue
        if result.flags & EARLY_STOP:
            if king is None:
                continue
        else:
            terminal_panels.setdefault(voter, set()).add(result.model_id)
        # A definitive rejection is a measured zero, including its stake in the
        # denominator. Dropping it would let a small positive minority dominate.
        votes = records.setdefault(result.model_id, {})
        if voter not in votes or not result.flags & EARLY_STOP:
            votes[voter] = result
    aggregates, early_losses = {}, []
    for model_id, votes in records.items():
        valid = {}
        for voter, result in votes.items():
            if king is None and (len(candidates) != 2 or terminal_panels.get(voter) != set(candidates)):
                continue
            if king:
                baseline = baselines.get((voter, king["model_id"]))
                if (baseline is None or baseline.flags & REJECTED or not baseline.flags & CONTROLS_OK
                        or (baseline.quality, baseline.reward) != (result.king_quality, result.king_reward)):
                    continue
            valid[voter] = result
        if not valid:
            continue
        stakes = {v: Fraction(str(snapshot["validators"][v])) for v in valid}
        total = sum(stakes.values())
        quality = sum(stakes[v] * r.quality for v, r in valid.items()) / total / SCALE
        reward = sum(stakes[v] * r.reward for v, r in valid.items()) / total / SCALE
        delta = sum(stakes[v] * (r.reward - r.king_reward) for v, r in valid.items()) / total / SCALE
        if quality <= Fraction(5, 100) or delta <= Fraction(2, 100):
            early_losses += [{'hotkey': v, 'value': r.commitment} for v, r in valid.items()
                             if r.flags & EARLY_STOP]
        if quality <= Fraction(5, 100):
            continue
        aggregates[model_id] = {"quality": quality, "reward": reward, "delta": delta,
                                "bounded": any(r.flags & EARLY_STOP for r in valid.values()),
                                "evaluators": sorted(valid), "stake": float(total)}
    inconclusive = [m for m, a in aggregates.items() if a['bounded'] and a['delta'] > Fraction(2, 100)]
    eligible = [m for m, a in aggregates.items() if candidates[m]['hotkey'] in snapshot['uids']
                and not a['bounded']
                and (king is None or a["delta"] > Fraction(2, 100))]
    # A bootstrap needs terminal results for two distinct immutable submissions.
    if king is None and not any(len(panel) == 2 for panel in terminal_panels.values()):
        eligible = []
    metric = "delta" if king else "reward"
    chosen = min(eligible, key=lambda m: (-aggregates[m][metric], candidates[m]["block"],
                                         candidates[m]["hotkey"])) if eligible else None
    # A partial upper bound must never crown a challenger or hide a possibly
    # stronger challenger. Its evaluator resumes the missing videos if needed.
    if inconclusive and (chosen is None or any(aggregates[m][metric] >= aggregates[chosen][metric]
                                              for m in inconclusive)):
        chosen = None
    winner = candidates[chosen] if chosen else king
    if winner and winner["hotkey"] not in snapshot["uids"]:
        winner = None
    return {"window": window, **weight_vector(winner, snapshot),
            "inconclusive": sorted(inconclusive),
            "early_losses": early_losses,
            "aggregates": {m: {k: float(v) if isinstance(v, Fraction) else v for k, v in a.items()}
                           for m, a in aggregates.items()}}
