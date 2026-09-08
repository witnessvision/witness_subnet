# Independent evaluation requirements

These requirements describe evidence needed for a release claim. They do not
indicate that a corpus or release has passed validation.

## Decision and claim

Decide whether a frozen candidate improves controlled video reconstruction over
a strong FullRead reference at comparable observation budgets, with supported
quality, efficiency and operational reliability. General natural-video
understanding is not established by this controlled task.

The source/uploader/reused-footage connected group is the independent unit.
Frames, clips, seeds, tiers and repeated runs from one group are correlated.
Unknown base-model pretraining exposure remains a limitation. Historical sources,
trained adapters and inspected evaluations remain development-only.

## Data gates before partition sealing

1. Inventory historical exposure, source aliases, media hashes and uploader
   identity. Resolve missing identity and cross-source reused footage. Perceptual
   screening is a review aid; absence of a match is not proof of independence.
2. Assign connected groups to development, calibration and reserved final roles
   before deriving clips or foreign inserts. Record the assignment and every
   source dependency. Inserts must belong to the same partition as their host.
3. Validate licenses, media integrity and actual task observability. Native text,
   audible speech/sounds and source cuts must have complete in-scope labels. A
   public scope rule must be perceptually identifiable; private overlay IDs or
   inaccessible generator metadata cannot determine what a miner must report.
   Do not silently treat absent annotations as negative labels.
4. Use decoded-audio ASR or no transcript for independent experiments. Pin ASR
   model and video hashes and preserve failed observations. Legacy label-derived
   hints are an explicitly named development ablation only.
5. Validate at least 12 independent final groups, each with all three tiers
   (36 valid scenes minimum), plus disjoint development and calibration groups.
   Invalid samples are excluded for documented non-performance reasons before
   inference. Insufficient valid groups is INCONCLUSIVE; do not lower the count.

## Freeze and selection

Develop prompts, policies, detectors and optional adapters on development groups
only. Calibrate selection once on the separately assigned calibration groups.
Replacement adapters start from the pinned clean base, not a historically
contaminated adapter. All attempts, selection criteria and inference failures
remain in the record.

Before opening final answers, seal media/labels, partition lineage, scorer and
task schema, model revision/adapter, miner source, dependencies, tool policy,
budgets and ASR identities. Candidate scoring versions are for development;
validity issues must be resolved before release evaluation.
Retain the corresponding FullRead implementation and exact configuration.

Run the selected miner and FullRead once per final scene/configuration. Fix
empty, random, transcript-only, source-prior, flooding and duplicate controls
before final scoring; distinguish no-observation attacks from controls that
consume metered evidence. Oracle controls require full quality on every valid
sample. A perfect-label duplicate stress test is not a no-observation attack.

## Endpoints and verdicts

Use equal weight per independent group and paired group bootstrap with seed
20260905 and 10,000 replicates. Report quality and gate pass rate by tier,
observation cost, latency, inference cost and operational failure rate. Missing,
error and degraded attempts contribute zero operational quality and zero reward;
complete-only quality is a separately labelled diagnostic. A retry preserves its
original attempt and does not silently replace the denominator.

- Integrity: all required hashes, provenance, partition separation, sample counts
  and label observability must pass. Proven leakage is FAIL; missing evidence is
  INCONCLUSIVE.
- Quality: the lower 95% group-bootstrap bound for each tier must reach the
  unchanged 0.70 / 0.65 / 0.60 floors.
- Comparison: the lower 95% paired quality-difference bound must be nonnegative,
  and the upper 95% bound on the ratio of paired mean visual-token use must be
  below 1 for an efficiency claim. Zero denominators are INCONCLUSIVE. Report
  audio, transcript and inference costs too; visual savings alone do not establish
  lower total cost.
- Gaming: every fixed no-observation attack must earn zero reward on every final
  scene. Report other controls by their declared information access. Known
  rewarded shortcuts are FAIL, even when the model's mean quality improves.
- Runtime: failed tasks earn zero reward. Unresolved metering or endpoint failures
  make readiness INCONCLUSIVE. Validator receipt finalization does not prove
  effective on-chain weights.

Promotion requires every required gate PASS. Failed final evidence cannot become
a tuning set and retain its sealed status; a later promotion needs fresh groups.
