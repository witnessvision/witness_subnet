# Evaluation v3: preparation and activation

This release changes scoring. All validators must use the same published policy
hash and activation window. It must not be installed as an unversioned patch to
an ongoing round. The answer and commitment wire formats remain unchanged.

## Scoring

Reward is `quality * (0.9 + 0.1 * speed)`, with the existing 60-second maximum
per clip. Quality remains the F1 of precision and recall. Ten videos, two clips
per video, independent Luna annotations, and architecture preprocessing remain.

Recall weights annotated events by salience (`core=1`, `detail=0.5`), rather than
averaging modalities equally. Each annotation retains its own denominator.
Repeated overlapping identical labels are collapsed. Supported propositions
citing the same fact/event share precision credit; paraphrases and interval
fragments cannot multiply that fact's capacity. Recall uses union coverage of
intervals with the existing one-second tolerance. Precision discounts unsupported
extensions outside the evidence interval. Ambiguous distinct labels remain
separate unless citation evidence links their equivalence across annotations.

Up to eight deduplicated unresolved visual/text propositions per response are
selected by a clip-bound hash, without model identity. Terra reviews frames and
must cite visible evidence. The code derives coverage from cited frames
(half a frame interval on either side), rather than trusting generated times.
Invalid frame evidence receives no additional credit and is recorded. It cannot confirm speech or sound. Missing evidence
is unresolved, not contradiction. Reviewed facts repair precision only; they
never become recall targets or retroactively change a published king baseline.
Provider failure is infrastructure failure, never a miner zero.

## Throughput

Four concurrent grading workers overlap API work with local GPU inference.
The API adapter has a process-wide four-call limit and single-flight request
locking in addition to durable budget reservations and cache receipts.

The local GPU worker flushes each answer, pauses after the first three videos,
and waits for an explicit continue/stop decision while keeping weights loaded.
Draw order, not completion order, controls early stopping. Complete answers and
complete-video grades are persisted for resumption. Remote SSH backends retain
the existing bounded batch interface. No second GPU model runs concurrently.

Phase wall time and summed call work are reported separately: overlapping
provider calls must not be added together to claim duel duration.

## Activation and rollback

1. Finish the current evaluation using the old release. Back up chain, trigger,
   and budget databases with SQLite's backup API; keep the old image and launch
   command. Never copy wallet secrets into release artifacts.
2. Choose the next unopened window and record the exact old/new policy hashes.
   Test opening a copy of the database with that schedule.
3. The CLI requires `--eval-upgrade-window N --previous-policy OLD_HASH` for every evaluator
   and follower. A fresh follower needs the same schedule before replaying.
4. Start the prepared release before window N. It follows chain state and sets
   existing weights, but does not evaluate windows before N. The ledger refuses
   an upgrade that changes already-opened windows or rewrites a stored upgrade schedule.
   Later upgrades retain the earlier schedule via `--prior-policy-upgrade`. The
   window-31 native-FP8 / window-32 admission release keeps the original
   scoring and FP8 execution path before activation, so `--hotkey-reset-upgrade`
   continues window-30 evaluations.
5. Verify the first new baseline and duel: matching policy, terminal commitment,
   finalized consensus/weights, API health and phase timing. Keep queue freezing,
   strict miner retries, compression, authentication and sandbox controls.

Before any new-policy result is published, rollback may restore the backup and
old release and replay finalized history. Once new-policy results exist, stop
new evaluations but retain the version-aware follower/weights process. Do not
rewrite those commitments or silently reinterpret the window with old scoring;
prepare a corrective release under another explicit future-window transition.

## vLLM and Omni quantization follow-up

This is a separate runtime qualification step, not a claim that FP4 is already
supported by the deployed validator. Preserve submitted weight bytes; validators
must not automatically quantize miners' models or silently change precision.

- Add a pinned, isolated vLLM/Omni thinker-only environment, with audio+video input
  and text output. Keep trusted loaders, safetensors, no remote code, no network,
  no credentials, no arbitrary engine arguments from miner manifests.
- Qualify Qwen2.5-Omni and Qwen3-Omni separately, including audio/video alignment,
  actual precision, output validity, GPU memory and end-to-end clip latency.
- Distinguish FP8, INT4 W4A16, NVFP4 W4A16 and native NVFP4 W4A4. The documented
  Qwen3-Omni W4A4 recipe uses Blackwell SM100+. The current Ada GPU must undergo
  separate backend qualification; generic quantization support is insufficient.
- Keep encoders and other excluded components in their documented precision.
  Quantization configuration and scales must be part of the immutable manifest.
- A format/backend not qualified on the validator is an infrastructure capability
  failure, not a consumed miner attempt. Default remains the existing runtime
  until a real audiovisual clip passes in the selected sandbox/backend.
- Add a private miner export/calibration utility only after selecting the exact
  supported checkpoint format. Use non-evaluation calibration data and preserve
  originals. No new GPU rental or model submission is part of this preparation.

Primary references:
[Omni ModelOpt formats and hardware](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/quantization/modelopt/),
[vLLM model support](https://docs.vllm.ai/en/stable/models/supported_models/).

## Operator burn override and qualification

`BURN_ALL=1` selects 100% burn regardless of the current king. It does not
change consensus or scoring. `--set-weights` independently enables chain writes;
omitting it means no weight transactions, and previously applied chain weights
remain in effect. The override is exposed in intended-weights status. Invalid
BURN_ALL values fail startup. For burn-only maintenance, keep --set-weights enabled with BURN_ALL=1.
The normal consensus vector is recorded as without_burn_override for inspection
but is never submitted while the override is active.
