# Mainnet v2 activation

## Shared checkpoint

Validators following the current SN20 mainnet protocol use:

| Parameter | Value |
| --- | --- |
| Network / netuid | Finney / 20 |
| Activation block | `9168259` |
| Finalized block hash | `0x085952610b9e139f00fa3b2fda5b6afa61240bf89dbb8c7f4663e70bab62a4bf` |
| Actual subnet epoch at activation | `25402` |
| Protocol source release | `053cee682b8ebf41ee101dc0df8d0a8330c60a6c` |
| Policy digest | `d063b9d0f4bd2d659f175823c3d720be32b2d7f7d7311dd07514fa5ad3ddd918` |

Pass `--activation-block 9168259 --activation-epoch 25402` and use a fresh
ledger directory when starting a new follower. Current CLI invocations also
require the complete schedule shown in [Validator setup](validator.md).
Window 36 switches the text judge and bounded visual reviewer to `gpt-6-luna`
(low effort), matching the existing labeler. No second-stage Terra call or fallback
is used. Both king and challenger are scored with Luna on the new round's clips;
previous scores and report hashes are not rewritten. Shared labeler/judge blind
spots remain possible; this cost change is not a claim of equal judging accuracy.

Use `--eval-upgrade-window 36 --previous-policy
94e218007681deafedea9c71cd42b779783fb137cca51a4d3f2015cdcf399cc1`
and nest the full window-31 upgrade (including its window-32 admission reset and
window-30 predecessor) in `--prior-policy-upgrade`, as in the guide. Do not pass
`--hotkey-reset-upgrade` at the new top level. The new policy digest is
`31806d75b12ebe00574ec9d4dd6afe633cf126c5cf826c7d8fefe60d12260bfb`.
Install before window 36 opens; the evaluator waits for that boundary, while
chain synchronization and weights continue. Already-opened windows cannot be
migrated.

At window 32, every hotkey gets one new admission opportunity. The existing king,
closed decisions, commitments, and archived admissions/results remain intact.
All old queue entries expire, including pending or withdrawn submissions. Miners
must publish a fresh commitment at or after the first finalized block of window 32;
old commitments do not automatically re-enter. As usual, a submission can only be
evaluated in a later window whose queue was locked after that commitment.
This is a **one-time reset**, not unlimited hotkey reuse. All validators/followers
must use the same release and activation schedule. The scoring formula is unchanged;
the runtime retains its original FP8 path until window 31. The admission reset
still occurs only at window 32.

These are shared protocol coordinates, not a certificate that
all real acceptance checks below have completed. The allocation is 100% burn
before a king is selected, then 0% burn and 100% to the king.

The implementation and synthetic/local integration tests are not an activation
certificate. Do not announce GPU architecture support or enable production
writes until the corresponding checks below have measured evidence.

## Release and protocol

1. Review the exact public source allowlist and secret scan, including the status
   export in this repository. Include only the documented public packages;
   exclude credentials, non-public data and host-specific configuration.
   Preserve the managed Git identity and pre-push guard.
2. Run `.venv/bin/python -m pytest -q` on the exact release; it includes the
   subnet tests as described in the README. Test separately installed display
   applications independently of validator activation.
3. Freeze the release/policy digest, public catalogue, activation block and actual
   subnet epoch. All validators use the same checkpoint. Do not point a v2 ledger
   at legacy HF/opinion commitments or silently change its policy on restart.
4. Confirm the selected RPC can replay finalized historical `CommitmentOf` state
   from that checkpoint, with sufficient throughput and retention. Losing history
   cannot be repaired from the latest commitment or dashboard reports.

## Required real acceptance evidence

- **Each advertised architecture:** exact pinned image/dependencies load a valid
  full model and answer audiovisual clips on the chosen GPU. Verify model load,
  audio alignment, response format, warmup, memory, per-clip deadline enforcement
  and hardware receipts. Keep it out of `enabled_architectures` until this passes;
  the shipped architecture list is a loader contract, not a qualification record.
- **Authenticated download:** a real miner endpoint accepts an eligible signed
  validator request, rejects a wrong key/receiver/replay/low-alpha request,
  resumes interrupted files and matches all hashes. Default gate: 100,000 alpha.
- **Sampling and scoring:** ten real public sources, two fresh clips each and two
  references each, correct control scores, one cached king evaluation and several
  queued challengers on the same window batch. Record actual throughput and costs;
  do not infer them from mock tests or historical unrelated runs.
  Verify the 60-second clip cap, cancellation on window closure and attempt-budget
  expiry on the real GPU, including during load/warmup. Exercise a three-video
  statistical cut, a later full continuation prompted by another evaluator,
  and an unresolved partial at window close. Measure false-cut frequency against
  complete ten-video outcomes; synthetic combinatorics are not model calibration.
- **Chain rehearsal:** publish all compact records on an authorized test network,
  overwrite latest commitments, restart followers, replay historical records and
  recover identical decisions. Test failures and the unknown-extrinsic recovery
  path without double-signing.
- **Weights:** verify target-chain burn/allocation limits, permits, current rate
  limits and commit/reveal settings. Observe submitted extrinsic and then the
  finalized applied 100% burn vector before the first king, then 0% burn / 100% king
  (compare normalized u16 weights, not individual 65535 values). The writer reads
  the current required weights version from finalized chain state. RPC success
  is not proof that weights have been applied.
- **Operations:** restart queue/outbox safely; kill the test evaluator and confirm
  its owned inference children terminate while the chain loop remains responsive.
  For the optional rented-GPU backend, also verify the provider watchdog stop and
  billing. Test optional budget exhaustion with a busy queue.
- **Dashboard:** inspect the real closed evaluation at 1920×1080 and 390px, including
  media/answers and averages; verify live, stale, missing and open-window states.

The mainnet gate remains closed until these real checks are recorded. Provisioning,
paid calls, wallet actions, publication and deployment require their own scoped
operator authorization; installing the package grants none of them.

## Rollout and recovery

Start CPU followers without `--set-weights`; inspect catch-up and intended
weights. Start an authorized evaluator on the prepared GPU. For the optional
RunPod backend, also start its independent watchdog. Once
scores, decisions and actual chain constraints have been verified, enable result
publication, then weight submission on the agreed activation schedule.

Use a dedicated service account, private state directories (`0700`), private env
and key files (`0600`), and one writer per hotkey. Supervise the validator; if using
the RunPod watchdog, supervise it separately. Keep RPC/SSH/API timeouts and alert on stale synchronization,
unknown extrinsic outcome, failed watchdog requests and budget exhaustion.

Before upgrading, stop the writer cleanly and preserve its entire state. A policy
change requires a coordinated protocol migration/checkpoint; do not roll back
code against a newer ledger. Never erase consumed hotkeys or pending submissions
to recover availability. An ambiguous signed transaction must be reconciled from
chain receipts/storage before retry. Keep dashboard failures independent of the
consensus service.

### Native FP8 runtime

The SALMONN FP8 loader additionally requires `accelerate==1.15.0`. Provision it
in the existing SALMONN environment before enabling quantized submissions:

```bash
envs/salmonn/bin/python -m pip install --no-deps accelerate==1.15.0
envs/salmonn/bin/python -c 'import accelerate; assert accelerate.__version__ == "1.15.0"'
```

This supplies the optional model-loading dependency missing from the historical
SALMONN environment; it does not upgrade Torch or the other pinned packages.
The setup uses system-site packages, so an installation that works on one host
can still lack this dependency on another. Qualify a **complete quantized-model
load in the actual inference sandbox**, not only standalone FP8 kernel arithmetic.
Keep active-window protocol lockfiles unchanged when repairing a missing
provisioning dependency. Include this prerequisite when recreating the runtime.

From window 31, SALMONN W8A8 compressed-tensors linears with symmetric per-tensor
FP8 E4M3 weights and dynamic per-tensor FP8 inputs use validator-owned GPU kernels.
Large prefill batches use CUDA scaled matrix multiplication; single-token decode
uses a fused vector kernel. BF16 modules, unaligned shapes and other quantization
schemes retain their previous path. This does not assert Omni FP4 support.
Quantizing every layer is not necessarily faster: miners should measure their
checkpoint, particularly small attention projections. Model loading/compilation
remains bounded; the per-clip deadline remains 60 seconds.

Operators who already staged the superseded reset-only release can explicitly
replace its **unopened** schedule by adding
`--replace-pending-policy 5a464a4591b76f478fa41a704f7eb1eeed836c31db96a28d949b8b1779c89b72`.
This preserves historical policy coordinates and refuses any affected opened
window. Fresh followers do not need this replacement flag.

For a `noexec` inference sandbox, build Triton's CPU launchers **without any miner
model, media or credentials mounted**, using the pinned SALMONN Python environment:

```bash
envs/salmonn/bin/python witness/benchmark/pod_fp8.py --prepare-extensions native_triton
```

This provisioning step needs a GPU and an executable temporary build directory.
Copy the resulting `native_triton/` directory beside `pod_fp8.py` in the pinned
worker image, owned by root and read-only. The inference worker then links its
cache entries to these trusted image files; `/tmp` stays `noexec`. GPU kernels
still compile into the temporary cache. Do not accept extension bundles from miners.
