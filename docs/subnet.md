# Witness protocol overview

Witness tasks expose metered video observations and accept structured miner
reconstructions. Start with [local setup](workflows.md), then follow the
[miner](miner.md) or [validator](validator.md) onboarding guide.

This release is a research preview. The historical default scorer has known
reward weaknesses; independent evaluation and production reward validation
remain unresolved.

## Round protocol

1. The validator prepares each scene privately. New programmatic/recomposed
   scenes use an OS-random 64-bit seed and an unrelated opaque scene token.
   Block hash and public validator hotkey alone cannot reconstruct either.
   Copied fixtures/locked scenes retain their actual source seed for replay.
2. Before asking any miner, the validator writes `commitments.json` and sends
   each scene's `seed_commitment` in `WitnessTask`. All miners see the same
   commitment for the same scene. Its precise SHA-256 input is UTF-8:
   `witness:seed-commitment:v1:<scene_id>:<decimal_seed>:<nonce>`.
   A fresh 32-byte random nonce, encoded as lowercase hex, prevents dictionary
   lookup of low-entropy fixture seeds. It stays in the private round directory
   (0700), in `seed-commitment.json` (0600), until responses are scored.
3. Each miner gets its own metered session, tool URL, deadline, budget and public
   task specification: duration, fps, tier, scene schema version and QA questions
   without answers. The scene schema version identifies the task format.
   Public `POST /session` is disabled for validator rounds. Tool URLs on the
   miner proxy are restricted to its assigned session.
4. The miner submits reconstruction and response status. The validator snapshots
   cost while closing the session under lock. Degraded/failed model responses
   receive no reward. Completed round artifacts retain the true scene seed,
   nonce reveal and commitment, permitting replay and comparison with the
   commitment received before the response. No automatic public artifact
   publication is implemented.

The commitment binds the revealed seed and scene identity. It does not by
itself prove fair seed sampling, correct labels or honest validator execution.
Keep exact media, source manifests, code and dependency identities with a round.


## Observations and scoring

Frame batches and individual frames come from decoded video; audio comes from
its decoded waveform. Transcript behavior is selected explicitly: `asr`, `none`, or historical
`legacy_labels`. The historical default supplies label-derived hints, not
independent ASR. See [Observation sources](observations.md). Metadata responses
include duration; observation responses carry server-owned running cost.

Visual metering is `ceil(width/14) * ceil(height/14) * frames`; audio seconds and
transcript characters are recorded separately. The scorer sums these three
channels as its declared budget proxy, not a dollar bill or hardware FLOP count.
Observer tokens/calls and wall time are additional reported measurements.

The frozen v1.5 scorer uses weights events .25, dialogue .20, shots .10, text .05,
audio .05, intentional errors .15 and QA .20. Its false-positive caps and
permissive QA credit are known gaming failures. Candidate versions are selected
explicitly and are not certified with an independent corpus. Native visible-text
annotation completeness remains unresolved.

The validator threshold is `max(q_min[tier], 0.35, 0.8 * best_scene_quality)`,
with unchanged tier minima 0.70/0.65/0.60. Relative competition cannot lower
those minima. Above the gate, quality is multiplied by
`max(0, 1 - 0.30 * cost / cost_ref)`, where `cost_ref` is a full 1-fps 640x360
visual read. Duplicate scored reconstructions share reward; round means feed
an EMA and normalized weights. An in-memory round tests this plumbing without
submitting real chain weights.

## Evidence boundaries

The default mixed configuration can reuse locked v1.5 scenes alongside fresh
programmatic scenes. This is implementation behavior, not approval to use the
contaminated locked corpus as permanent production reward evidence. Repeated
exposure permits memorization. Randomized generator choices also do not prove
absence of an exploitable grammar.

Independent evaluation requires source-lineage separation, validated annotations,
fixed controls and uncertainty estimates at the independent-group level. See
[evaluation requirements](evaluation-v2.md). Scoring version and observation
source are part of the evaluation identity and must not be mixed in one history.
