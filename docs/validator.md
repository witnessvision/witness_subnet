# Validator: CPU follower by default

A follower replays finalized on-chain submissions and evaluation scores and
computes the same single king as every validator using that protocol release.
It does not need GPU, ffmpeg, API keys, model downloads or dashboard connectivity.
An evaluator additionally runs a private GPU worker and publishes its measured
scores. Both modes allocate 100% to the owner burn UID before the first king,
then 70% to burn and 30% to the single king. A deregistered king returns the
allocation to 100% burn. Applied weights are checked as normalized proportions.

## Activation and follower

All operators must agree on **release/policy hash, activation block and actual
subnet epoch at that block**. These values are mandatory; no production
checkpoint is silently selected. Use a historical-state-capable chain endpoint.

```bash
.venv/bin/witness-validator --hotkey PUBLIC_SS58 \
  --activation-block BLOCK --activation-epoch EPOCH \
  --root /var/lib/witness-validator --status-port 8099
```

This opens no wallet and performs no chain writes. Inspect
`intended-weights.json`, `queue.json`, synchronization progress and policy hash.
To authorize weights, replace `--hotkey` with `--wallet-name NAME
--wallet-hotkey HOTKEY` and add `--set-weights`. `--publish-results` is a separate
flag available only to evaluators. Defaults are Finney SN20 and a 12-second
poll interval. `--once` performs one bounded replay step (up to 64 blocks), not
necessarily a complete historical synchronization.

No weights or evaluation start until replay catches the finalized head. The
chain thread continues while the GPU runs; a separate serialized writer handles
commitments and weight extrinsics. Weight failures do not advance the success
checkpoint. Restarting during a send marks its outcome unknown and reconciles
observed weights before another signature. An unresolved pending commit/reveal
requires operator reconciliation; do not erase that record to force a retry.
Run only one writer for a hotkey, with one durable root outside release folders.
A filesystem lock protects the same root; it cannot protect two different hosts.

## Evaluator configuration

Install `requirements-evaluator.txt` and ffmpeg. The following is a **local GPU**
configuration, stored privately outside the checkout:

```json
{
  "provider": "openai",
  "enabled_architectures": ["qwen2.5-omni"],
  "api_daily_limit_usd": null,
  "gpu": {"backend": "local", "workspace": "/var/lib/witness-gpu"}
}
```

If your host cannot reach the Archive frontend, `source_urls` may name a private
JSON file mapping catalogue identifiers to their resolved official Archive HTTPS
download URLs. Resolve these from the canonical download redirects. Only Archive
subdomains and the same item/file path are accepted; file size and media checks
still apply. Refresh stale URLs deliberately. This changes neither the 1,000-video
catalogue nor the private draw, and does not enable arbitrary download hosts.

Set `enabled_architectures` only after those loaders pass your real GPU qualification;
no architecture is enabled by default. An unenabled architecture defers without
consuming the submission. The examples assume Qwen2.5-Omni has passed.

### Optional remote backends

These are alternatives, not requirements of the single-machine installation.
An SSH backend uses `backend: ssh`, `host`, `port`, `ssh_key` and `workspace`.
A RunPod backend uses an explicit compute budget:

```json
{
  "provider": "openai",
  "enabled_architectures": ["qwen2.5-omni"],
  "gpu": {
    "backend": "runpod",
    "ssh_key": "/etc/witness/gpu_key",
    "workspace": "/workspace",
    "gpu_type_ids": ["NVIDIA A40"],
    "daily_cap_usd": 10,
    "idle_stop_s": 900,
    "ready_timeout_s": 900,
    "container_disk_gb": 40,
    "volume_gb": 150
  }
}
```

The accompanying `.pub` file supplies the SSH public key. Optional
`network_volume_id` plus `data_center_id` prefer that volume; fallback pods have
separate disks. The backend may stop or replace its own compute; put durable
records and verified weights on the validator CPU host. Pin/verify SSH host keys
before production use; existing host-key changes are rejected.

Credentials go in a private `0600` env file: `OPENAI_API_KEY` for OpenAI or
`GM_API_KEY` for `provider: saygm`, plus `RUNPOD_API_KEY` only for RunPod. Wallet
secrets never belong there. Luna labeling and Terra judging are versioned policy,
not per-validator tuning knobs. They share one persistent spending ledger. `api_daily_limit_usd: null` (the
evaluator default) records consumption without a monetary cap. Set a non-negative
USD amount to impose your own UTC daily cap. It never changes scoring. A saved
budget policy cannot change silently; migrate it explicitly and preserve its ledger. RunPod's
compute cap is additional; storage, network and provider settlement can add costs.

```bash
.venv/bin/witness-validator --mode evaluator \
  --activation-block BLOCK --activation-epoch EPOCH \
  --root /var/lib/witness-validator --config /etc/witness/evaluator.json \
  --env /etc/witness/evaluator.env --wallet-name NAME --wallet-hotkey HOTKEY \
  --status-port 8099
```

Starting evaluator mode can spend GPU/API budget even without chain write flags.
A signing hotkey is needed for authenticated miner downloads. The miner's default
100,000-alpha gate can prevent lower-stake evaluators from downloading; those
validators can still follow scores and set weights. Add `--publish-results` and
`--set-weights` only when those writes are authorized and activation checks pass.

## One GPU machine, one startup command

Install the subnet package on a CUDA machine with the runtime image's
pinned base dependencies. Prepare the GPU environments once, before activation:

```bash
.venv/bin/python -m pip install -e '.[evaluator]'
.venv/bin/witness-gpu-setup --workspace /var/lib/witness-gpu
.venv/bin/witness-validator --mode evaluator \
  --activation-block BLOCK --activation-epoch EPOCH \
  --root /var/lib/witness-validator --config /etc/witness/evaluator.json \
  --env /etc/witness/evaluator.env --wallet-name NAME --wallet-hotkey HOTKEY \
  --publish-results --set-weights --status-host 0.0.0.0 --status-port 8099
```

Use the local GPU configuration above. This process exports validator status and runs
chain tracking and evaluation independently. Model and media subprocesses remain
isolated and cancellable. It never rents or stops a GPU and needs no RunPod key,
SSH worker or separate GPU service. Process supervision for reboot recovery is
optional. Put persistent state and encrypted wallet files on durable storage;
never rely on a cloud container's ephemeral disk. Serve the status port over HTTPS.

Other GPU jobs must acquire the same `/var/lib/witness-gpu/gpu.lock` with `flock`.
The evaluator acquires that lock for the whole paired attempt and checks for
existing GPU processes. If occupied it defers without a miner penalty. This is
cooperative scheduling: do not launch an uncoordinated GPU workload during a duel.
The validator never terminates another operator's job.

From window 13 the 1,800-second attempt budget starts before shared preparation
and downloads; completed clips, labels, first model answers and grades survive
retries. Work continues between epochs of the same two-epoch window. Local
media/inference processes are cancelled at expiry or window closure, with a five-second kill
backstop; network requests can take their bounded timeout to unwind. GPU setup
is an installation step, not part of a production duel. A 60-second model answer
limit is distinct from model loading, labeling and judging time.

Model acquisition uses at most two concurrent signed 4 MiB requests per miner,
within the existing server limit. HTTP/1.1 connections are reused when supported;
HTTP/1.0 peers remain compatible. Only contiguous completed ranges are appended
to partial files, so retries resume without holes even if ranges arrive out of
order. Interrupted or truncated transfers are retried; complete files must still
match their committed hashes. Slow initial downloads can span multiple attempts
and windows: the per-attempt limit is not a total download or duel limit.

Planned epoch/window cancellation and an exhausted attempt budget resume without
the exponential failure backoff. Real infrastructure failures still back off.
Dispatch filters to the current window's eligible candidates before applying
coldkey FIFO, so an ineligible submission cannot block an eligible sibling.
On restart, queued legacy timeout/cancellation rows receive one immediate retry;
their attempt counts, downloaded bytes and finalized usage remain intact.
Retryable failures, including socket timeouts, disconnected peers and temporary
server errors, retain a cooldown capped at five finalized blocks. Normal resumed
turns do not turn a single failure into the maximum historical-attempt backoff.
Restart also caps older queued backoffs; exhausted acquisition budgets remain
parked. The cumulative acquisition budget below still limits repeated failures.
`download-budgets/<model>.error.json` records only the exception type and, when
available, a fixed transport code or HTTP status; it omits authentication and URLs.

Miners must send each requested file range with lossless Zstandard compression. The
validator decompresses it before writing, resumes at original byte offsets, and
checks the same committed file hashes. Compressed and decoded sizes and decoder
memory are bounded; malformed, oversized or concatenated frames are rejected.
A one-byte authenticated probe checks support before batch preparation or inference,
also for cached challengers. Raw/unsupported encodings and invalid frames exclude
the challenger from the operational queue without a score or consumed hotkey. The
local exclusion survives restarts; one bounded five-second recovery probe per worker
pass retries entries after 25 finalized blocks. Updating the server requeues the
same binding. Other eligible submissions, including coldkey siblings, can proceed.
Previously finalized results and the current king remain intact.

The client requires `zstd, identity;q=0`; tiny/incompressible ranges must still use
Zstandard. Encoded frames allow at most 64 KiB of overhead over the requested
original size. This changes neither model precision nor evaluation scores.

Package limits apply to the original, decompressed bytes: 24 GB for SALMONN2 Pro
and Qwen2.5 Omni, and 44 GB for Qwen3 Omni (decimal GB; only explicitly qualified
architectures are enabled). The committed manifest is checked before requesting
model files. Each request has a 30-second absolute deadline, additionally bounded
by the remaining attempt budget, including peers that keep HTTP headers alive
by slowly sending bytes. Cancellation closes the owned request socket. A miner
cannot bypass the package limit by compressing oversized weights into a small
response; each decoded range is at most 4 MiB and must match its original length.

Each model has a persistent 90-minute budget for active acquisition, shared by
all retries and windows on this evaluator. Queue waiting time is excluded;
manifest transfer, file transfer and local verification during acquisition count.
An individual turn remains bounded by 15 minutes and the enclosing attempt's
remaining time. On normal exit, only elapsed acquisition time is charged. A
crash conservatively charges the reserved turn (at most 15 minutes), so repeated
restarts cannot reset the limit. Accounting starts when a model first reaches
the budget-aware downloader; historical unrecorded transfer time is not inferred.

Exhausting the budget parks acquisition for operator review with
`DownloadBudgetExceeded`. The partial cache and hotkey reservation remain intact;
no zero score or consumed hotkey is created. A parked entry does not block an
eligible sibling sharing its coldkey. Ordinary restarts do not unpark it. Keep
`download-budgets/` with the other validator state; deleting these records resets
resource accounting and must not be used as an automatic retry mechanism.

Known host video-decoder resource exhaustion is an infrastructure failure: defer
and retry rather than scoring the miner's answer as zero. On hosts exposing many
CPU cores, bound the service's CPU affinity and numerical-library thread counts
so decoder/model thread pools cannot exhaust the process limit. Qualify the real
clips under those service limits before publishing scores.

For incident recovery, `evaluation_hold_windows: [WINDOW_ID]` in the evaluator
configuration prevents new attempts in the listed windows while chain replay,
the dashboard and weight submission continue. Holds do not amend chain history,
publish replacement scores or consume hotkeys, and do not affect later windows.
If an infrastructure-corrupted bootstrap result was already finalized, do not
complete its pair: leave that window inconclusive and retry in a later window.

Evaluators automatically remove completed challenger weight caches after the
evaluation's consensus window closes. Cleanup runs between attempts and removes
both the downloaded weights and their staged GPU copies. It preserves the current
king, results awaiting publication, and retryable or inconclusive challenges,
including partial downloads. Replaced kings are also eligible once no longer
needed by a pending challenge. Scores, reports, clips and chain history remain
available. Cleanup retries on failure and after restart; it never rents or starts
a GPU just to remove files. An offline GPU copy is removed when reachable again.
This is not a disk quota: allow space for the king and pending downloads as well
as evaluation media and environments.

## Continuous queue and recovery

- One immutable first submission per hotkey; coldkey ownership comes from the
  finalized submission block. Rotation is durable, with FIFO within each owner.
- The batch, candidates and king are fixed per two-epoch window. New submissions
  wait for the next window. Pending infrastructure retries keep their hotkey
  reservation and use the new window's batch after expiry.
- Scores are written to a durable outbox. A hotkey is consumed after its terminal
  score is observed finalized, not merely after local inference or an RPC reply.
  Bootstrap consumption is atomic for its complete pair. Copied commitments bind
  distinct hotkey identities; duplicate weights are rejected only after authenticated
  acquisition.
- Model code is never downloaded or executed. Only pinned validator loaders run
  allowlisted verified weights. Clip deadlines kill the owned inference worker;
  hardware identity changes invalidate the paired comparison for that window.
- Preserve `chain.sqlite3`, `triggers.sqlite3`, outbox, sampling secret, models,
  budget ledgers and pending extrinsic records together. Restoring only a queue
  or deleting a budget file changes the meaning of the state.

## Optional RunPod backend watchdog

Run the watchdog as a **separate supervised process**, with the same root,
configuration and private env. It never starts or deletes compute. Without
`--stop`, it only reports the action it would take.

```bash
.venv/bin/witness-gpu-watchdog --root /var/lib/witness-validator \
  --config /etc/witness/evaluator.json --env /etc/witness/evaluator.env \
  --interval 30 --stale-seconds 300 --stop
```

It checks ownership, heartbeat, startup deadline and daily spend. Before asking
the provider to stop a pod, it writes `gpu-watchdog-stop.json`; this latch blocks
evaluator restarts of compute. Failed requests remain visible and retryable.
After an intervention, reconcile actual provider status/billing, stop the
validator, settle its GPU accounting and deliberately clear the latch before
resuming. Keep the watchdog running independently of validator crashes. A stop
request is not confirmation of stopped billing; verify provider state. Internal
checks run every 15 seconds; neither check can guarantee an exact monetary cap
under provider/network failure.

## Dashboard and operator state

`--status-port 8099` defaults to loopback. It exports this validator's status
at `/queue.json` and finalized evidence at
`/api/evaluations/{validator}/{window}/{model}` and
`/api/media/{validator}/{window}/{digest}.mp4`. Place it behind the operator's
chosen HTTPS proxy when sharing it. The status server serves no HTML or assets.
Reports and clips remain unavailable while their window is open. Report hashes
and media hashes are checked before serving evidence; references, API logs,
model weights and sampling secrets remain private.

Optional external display applications can consume these endpoints. Validators
may send signed public status with `--telemetry-url https://DISPLAY/api/telemetry`.
Signatures bind the hotkey to public status; receiving applications must reject
stale/replayed updates and verify permitted-validator membership. Telemetry never
supplies consensus decisions. No wallet keys or private labels are sent.

Useful private files: `last-error.json`, `worker-error.json`,
`commitment-error.json`, `weight-send.json`, `last-weights.json`,
`watchdog-status.json`, `gpu.json`. Errors are type-only to avoid leaking provider
payloads. Keep these files out of the public repository.

## Prefetch and performance evidence

The owned evaluator can download one additional challenger while the current
paired evaluation runs. It must belong to the frozen window panel; late
submissions cannot be prefetched into the active evaluation. Acquisition uses
an independent 900-second turn, existing cumulative time/package limits,
mandatory zstandard and final manifest/file hash verification. Before starting,
free space must cover two maximum-size packages plus 20 GB reserve. Prefetch
never runs a model, publishes a score, consumes a hotkey, or changes queue order.
Window closure or shutdown cancels it. Foreground acquisition waits for that
model's prefetch to unwind, preventing two writers to the same partial files.

Private `performance/<window>/<model>.json` records each challenge's cumulative
active seconds, elapsed wall time since its first attempt (including retries),
and the latest attempt's phase timings. `performance-latest.json` and structured
`duel_timing` log events provide a concise operational view. Preparation/labels,
download or prefetch wait, cache verification, GPU calls and judge-provider
time are measured separately; nested phases must not be added as a total.
Per-clip measured latency, status, reused-answer markers and p50/p95/max are
recorded. GPU call time includes setup, transfer, load and warmup; subtracting
new clip latency gives aggregate overhead, not an isolated load measurement.
Ready-for-publication is distinct from finalized chain acknowledgment.
A separate `<model>.ack.json` records the actual finalized commitment block
and elapsed wall time until this validator observed that acknowledgment.
Timing diagnostics never feed scoring or consensus.
