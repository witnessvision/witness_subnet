# Validator: CPU follower by default

A follower replays finalized on-chain submissions and evaluation scores and
computes the same single king as every validator using that protocol release.
It does not need GPU, ffmpeg, API keys, model downloads or dashboard connectivity.
An evaluator additionally runs a private GPU worker and publishes its measured
scores. Both modes use the same consensus code and one-winner weight vector.

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
  "gpu": {"backend": "local", "workspace": "/var/lib/witness-gpu"}
}
```

Set `enabled_architectures` only after those loaders pass your real GPU qualification;
no architecture is enabled by default. An unenabled architecture defers without
consuming the submission. The examples assume Qwen2.5-Omni has passed.

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
not per-validator tuning knobs. They have separate persistent budget ledgers,
each capped at 9 USD per UTC day; this is a limit, not a cost estimate. RunPod's
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

## Independent RunPod watchdog

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

`--status-port 8099` defaults to loopback. Serve the public read-only projection
behind the operator's chosen HTTPS proxy. Multiple sources may be displayed:

```bash
.venv/bin/witness-dashboard \
  --source http://EVALUATOR_A:8099/queue.json \
  --source http://EVALUATOR_B:8099/queue.json --port 8098
```

The first configured source supplies the chain decision shown on the page.
If unavailable, that decision is unknown; the dashboard never elects a majority
king. Additional sources contribute their own queues and evaluation detail.
The separate web application accepts `WITNESS_VALIDATOR_SOURCES` as a JSON array.

The API exposes `/api/subnet`, paginated/searchable `/api/hotkeys`, and closed
`/api/evaluations/{validator}/{window}/{model}` reports with hash-bound media.
Reports and clips remain unavailable while their window is open. Responses,
quality, latency, time reward and clip/video/total averages are displayed after
close. Reference facts, API logs, weights and sampling secrets remain private.
Unavailable values stay unknown. Decision, submitted and applied weights are
shown separately; commit/reveal can delay the last of these.

Useful private files: `last-error.json`, `worker-error.json`,
`commitment-error.json`, `weight-send.json`, `last-weights.json`,
`watchdog-status.json`, `gpu.json`. Errors are type-only to avoid leaking provider
payloads. Keep these files out of the public repository.
