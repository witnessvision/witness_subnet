# Mainnet v2 activation

The implementation and synthetic/local integration tests are not an activation
certificate. Do not announce GPU architecture support or enable production
writes until the corresponding checks below have measured evidence.

## Release and protocol

1. Review the exact public source allowlist and secret scan, including the web
   application in this repository. Include only the documented public packages;
   exclude credentials, non-public data and host-specific configuration.
   Preserve the managed Git identity and pre-push guard.
2. Run `.venv/bin/python -m pytest -q` on the exact release; it includes both
   subnet and web tests. Install the `web[dev]` package from this checkout as
   described in the README. The dashboard API contract remains shared.
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
- **Sampling and scoring:** five real public sources, two fresh clips each and two
  references each, correct control scores, one cached king evaluation and several
  queued challengers on the same window batch. Record actual throughput and costs;
  do not infer them from mock tests or historical unrelated runs.
  Verify the 60-second clip cap, cancellation on epoch change and attempt-budget
  expiry on the real GPU, including during load/warmup. Exercise a three-video
  statistical cut, a later full continuation prompted by another evaluator,
  and an unresolved partial at window close. Measure false-cut frequency against
  complete five-video outcomes; synthetic combinatorics are not model calibration.
- **Chain rehearsal:** publish all compact records on an authorized test network,
  overwrite latest commitments, restart followers, replay historical records and
  recover identical decisions. Test failures and the unknown-extrinsic recovery
  path without double-signing.
- **Weights:** verify target-chain burn/allocation limits, permits, current rate
  limits and commit/reveal settings. Observe submitted extrinsic and then the
  finalized applied 100% burn vector before the first king, then 70% burn / 30% king
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
