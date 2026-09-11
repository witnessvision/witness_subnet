# Witness scoring and advanced validator operation

For the CPU-only SN20 setup, start with the [validator guide](validator.md).

## Scoring and artifacts

For each session, validator cost is snapshotted from the server-owned store
under its lock when the task closes. JSONL logs retain the audit trail. Miner
cost claims are never used. Production **Witness Scorer v1.1.0** uses a fixed,
inclusive quality threshold in every tier:

```text
reward before duplicates = quality * efficiency if quality >= 0.4 else 0
```

There is no partial band or peer-relative threshold in this version.
Missing, invalid or degraded responses receive zero reward.
Sessions are created directly by the validator store; the public tool server
rejects `POST /session`. Sessions are removed when their task finishes.

Identical canonical reconstruction hashes from multiple UIDs divide each
matching score by the number of UIDs sharing it. Scene scores are averaged, then
folded into `ema.json`. Non-responders receive zero weight. Positive eligible
EMA scores are normalized; the configured burn share is added to the burn UID.
If no miner has a positive eligible score, the full vector goes to the valid burn
UID (or remains zero if none is configured).

### Winner takes all with a reserved burn share

Select `--weight-policy winner-takes-all --burn-rate 0.70 --burn-uid BURN_UID`
to reserve 70% of this validator's weight for the burn destination and assign
the entire remaining 30% to one miner. The equivalent environment setting is
`WITNESS_WEIGHT_POLICY=winner-takes-all`. Start verification with
`--no-set-weights`; the allocation policy does not enable submissions.

The mainnet preset uses aggregation/weight policy v1.1.0: the mean of the last
five round rewards, then an EMA with alpha 0.2. `--score-window` and `--ema-alpha`
configure these values. The first positive window mean initializes the EMA
directly; zero rounds count toward the mean. See the [validator guide](validator.md)
for the formula, persisted history and upgrade instructions. Advanced mode without
`--score-window` retains policy v1.0.0 and its single-round EMA.

Both policies rank by EMA of scored rewards. A winner must have a valid response
and positive mean reward in the current round as well as a positive EMA. The burn
UID cannot win. Exact EMA ties go to the lowest UID, independently of query order.
Nonresponders and miners with zero current reward cannot win using old scores.
If nobody qualifies, 100% goes to the burn UID. Missing burn targets and invalid
scores fail closed. WTA EMA history binds each UID to its hotkey; re-registered
UIDs and histories lacking this binding start without inherited scores.

Use a fresh round root when activating this policy after a validator code change;
do not silently overwrite diagnostic EMA identities. Scorer v1.0.0 retains its
quality and reward formulas. The weight policy is recorded separately in each
round with the winner, burn fraction and tie rule. Historical proportional
allocation remains available with its existing behavior.

WTA submissions also persist `weight-submission.json` before network I/O. A
prepared, ambiguous or unconfirmed submission blocks subsequent submitting
rounds until reconciled. Read-only rounds leave this guard intact. A finalized
commitment still requires separate verification of revealed weights.

Before activation, verify the burn UID belongs to the current subnet owner,
`RecycleOrBurn=Burn`, the validator permit and current weight constraints. Evaluate
the intended registered miner population using validated private scenes. Stop
any separate full-burn writer before enabling scored submissions, so only one
process controls this hotkey's weights. With commit/reveal enabled, wait for the
new weights to appear in finalized storage and reconcile the next epoch's
incentives. The 70/30 vector is this validator's vote; other validators and Yuma
Consensus determine the actual miner allocation. Calculated weight is not proof
of a paid reward.

Completed rounds are stored below `--round-root` as `round_<id>/round.json`.
After scoring, the artifact records scene seeds, reports and a `prepared` weight
submission before network I/O. It then records `simulated`, `submitted`,
`included`, `finalized`, `rejected`, or `unknown` from the adapter evidence.
The pinned SDK is asked to wait for inclusion/finalization with one attempt.
A transport exception remains `unknown`; it is not safe to assume rejection
or automatically rebroadcast it. Raw signing payloads and provider messages
are not stored.

Finalization of a commitment does not prove weights are active after reveal;
`weights_applied` remains unknown until separately verified on chain. EMA stores
scored observations independently of submission success. Private scenes and
session logs remain under the round's `private/` directory.

Metered cost is benchmark cost, not currency: visual patches, requested audio
seconds, and returned transcript characters. The scorer's efficiency factor is
`max(0, 1 - 0.30 * total_cost / cost_ref)`.
## Scoring and observation identity

The default scorer is production `1.1.0`; select it explicitly with
`--score-version 1.1.0`. It does not require candidate diagnostic mode.
Historical scoring remains explicitly selectable, including `1.0.0`, `1.5` and
`1.9-candidate`; research versions require `--allow-unlocked`.
Pass `--benchmark-lock PATH` to verify a particular frozen corpus. An explicit
missing or altered lock still fails closed unless `--allow-unlocked` is given.
`--transcript-source asr` requires explicitly supplied
`--scene` directories with valid `observations/transcript.json` sidecars. Missing
or stale observations fail before querying miners. `none` disables transcript
evidence; `legacy_labels` retains the historical label-derived hints.

Round artifacts record the scorer, validator, contract and metering hashes, the
benchmark-lock hash when present, observation budget and per-scene media/label/ASR
hashes. EMA files carry the same scoring identity. Changing it requires a fresh
round root or an explicitly verified migration; the validator refuses to blend incompatible reward histories before
creating a round or querying miners. Legacy unversioned EMA files remain readable
only with the historical 1.5/legacy-label configuration.

### Production scorer v1.1.0

Quality and metered efficiency are unchanged from v1.0.0. Every scene with
quality at least `0.4` receives `quality * efficiency` before duplicate sharing;
scenes below `0.4` receive zero. All assigned scenes, including failures and
below-threshold scenes, remain in the round mean. The threshold is independent
of tier and other miners. Empty responses still fail this gate on the current
synthetic workload; this is not a guarantee about every possible future corpus.

`gate.passed` identifies the inclusive fixed gate and `gate.reward_factor` is
always zero or one. `score_before_duplicates` and the final `score` expose the
complete reward calculation. Upgrading from v1.0.0 requires new history; keep
old artifacts and reconcile pending submissions before changing the round root.
Scorer and aggregation versions are separate identities.

### Historical production scorer v1.0.0

**Witness Scorer v1.0.0** is the first production release of the contract used in
the mainnet evaluation. It promotes `1.9-candidate` without changing numerical
quality, efficiency, partial rewards, relative competition or duplicate sharing.
The version refers to the scoring contract, independently of package and corpus
versions. Changes to numerical rules require a new scorer version.

Quality retains the frozen v1.8 family weights:

| Family | Weight |
| --- | ---: |
| Events | 0.25 |
| Dialogue | 0.20 |
| Shots | 0.10 |
| On-screen text | 0.05 |
| Audio events | 0.05 |
| Intentional errors | 0.15 |
| QA | 0.20 |

Every miner's round report includes `metrics`: mean reconstruction `quality`,
`continuous_score` (quality times observation efficiency, before gating and
duplicate sharing), final `reward`, full-threshold and positive-reward rates,
valid response rate, component scores and difficulty slices. Failed responses
count as zero in the means; `scenes` is the denominator. Compare quality on the
same held-out source videos and budgets, independently of reward version.

The production partial-reward policy uses unchanged v1.8 quality. The full-credit threshold
remains `T = max(tier minimum, 0.35, 0.8 * best responding quality)`, with tier
minima 0.70/0.65/0.60. Partial credit starts at `F = max(0.35, T - 0.15)`:

```text
factor = clamp((quality - F) / (T - F), 0, 1)
reward before duplicates = quality * efficiency * factor
```

Below the floor reward is zero; at or above the original threshold reward is
unchanged. `gate.passed` still means the full threshold was reached, while
`gate.reward_factor` exposes partial credit. Invalid/missing responses remain
zero and duplicate sharing still applies. `1.9-candidate` preserves this same
reward calculation and its historical version label. Production v1.0.0 remains explicitly selectable with its original numerical behavior.

`scoring_identity.version` records `1.0.0`. A production scorer can still evaluate
diagnostic inputs: `--allow-unlocked` keeps `scoring_identity.mode=diagnostic`.
Promoting the contract does not relabel old round receipts, certify a corpus or
turn calculated weights into paid emissions. Weight submission remains controlled
separately by `--no-set-weights` and burn configuration.

For an equivalent-policy EMA migration, preserve the old file and its identity,
verify the old/new rules against the stored records, and explicitly bind the
unchanged EMA scores to the new code identity. A version match alone is not
sufficient: the normal validator never auto-migrates incompatible EMA files.

Compare stored v1.8 rounds with v1.0.0 without inference or modifying the originals:

```bash
python -m witness.reward_report rounds/round_example/round.json --out comparison.json
```

Multiple round paths may be supplied if their scoring identities match. Relative
competition and duplicate sharing are recomputed separately within each round;
reports preserve source hashes and separate old/new reward from unchanged quality.
Use distinct output paths and fresh round roots when changing scoring identity.

### Five-scene rounds aligned to subnet epochs

The default round now contains five scenes (`--scenes 5`). Use
`--epoch-aligned --no-set-weights` to run the full forward/scoring path without
submitting its calculated weights. Epoch scheduling follows finalized
`SubnetEpochIndex`, not a fixed wall-clock timer or legacy block modulo formula.
The first launch runs once in the current epoch; subsequent rounds wait for a new
epoch. State under `round-root/epoch-state.json` prevents replay after restart.
A lock excludes concurrent schedulers sharing that round root. A failed round
waits for the next epoch, and epochs crossed during a long round are skipped.
`--epoch-poll-s` defaults to 12 seconds. Without `--epoch-aligned`, the existing
finish-then-sleep `--interval-s` behavior remains available.

Burn maintenance can run independently in `--burn-only` mode while the evaluation
process uses `--no-set-weights`. The evaluation process needs no axon. Running
both is not permission to enable scoring-derived weight submissions.
