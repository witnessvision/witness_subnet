# Witness validator

The validator generates or copies a private scene batch, starts the metered tool
server, opens one session per miner and scene, sends `WitnessTask` synapses, and
sets weights from the EMA of round scores. Video bytes and `scene.json` never
enter the synapse. Only duration, fps, tier, QA IDs/questions, budget, the opaque
scene/session IDs, and the tool URL are public.

## Prerequisites and local verification

Complete the [local quickstart](workflows.md), including its generated scene and
no-chain round with the empty base miner. Dependencies are declared in `pyproject.toml`.

For network operation, obtain the intended network, subnet UID, registration
requirements and approved scoring configuration from the subnet operator.
Configure a validator wallet hotkey, private scene storage and a tool endpoint
reachable by miners. This research preview does not establish production reward
readiness. CLI defaults are not a public deployment announcement.

## Network configuration

Set `WITNESS_NETUID`, `WITNESS_NETWORK`, `WITNESS_WALLET` and `WITNESS_HOTKEY`
for the intended network. The principal operator settings are:

| Setting | Purpose |
| --- | --- |
| `--tool-host`, `--tool-port` | Observation server bind address and port |
| `--tool-public-url` / `WITNESS_TOOL_PUBLIC_URL` | Miner-reachable observation API URL |
| `--scene` | Explicit scene directory; repeat for additional scenes |
| `--pool-manifest` | Source manifest for recomposed scenes |
| `--round-root` | Private round artifacts and scoring history |
| `--deadline-s` | Per-task response deadline |
| `--score-version` | Versioned scoring contract |
| `--transcript-source` | `asr`, `none` or historical `legacy_labels` |
| `--ema-alpha` | Current-round contribution to the moving average |
| `--once` | Exit after one round |

Inspect the complete interface before configuring a network round:

```bash
.venv/bin/witness-validator --help
```

A public tool URL is required when binding a wildcard address. It must route to
the configured tool port. The validator creates sessions locally; miners receive
only their assigned session URLs. Private labels and round files must not be
served through that endpoint.

Use validated scenes and a matching benchmark identity. Source recomposition can
invoke a hosted speech service on a cache miss; configure credentials and a call
cap before enabling that path. The local dry run makes no hosted model calls.

## Scoring and artifacts

For each session, validator cost is snapshotted from the server-owned store
under its lock when the task closes. JSONL logs retain the audit trail. Miner
cost claims are never used. The relative gate is:

```text
quality >= max(tier_minimum, 0.35, 0.8 * best_quality_for_scene)
score = quality * efficiency_factor
```

The frozen tier minima are 0.70 / 0.65 / 0.60 for tiers 1 / 2 / 3. A
relative gate cannot weaken them. Degraded Observer responses receive no reward.
Sessions are created directly by the validator store; the public tool server
rejects `POST /session`. Sessions are removed when their task finishes.

Identical canonical reconstruction hashes from multiple UIDs divide each
matching score by the number of UIDs sharing it. Scene scores are averaged, then
folded into `ema.json`. Non-responders receive zero weight. Positive eligible
EMA scores are normalized; the configured burn share is added to the burn UID.
If no miner has a positive eligible score, the full vector goes to the valid burn
UID (or remains zero if none is configured).

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

The default remains historical scoring 1.5. Local diagnostic rounds can select
`--score-version 1.7-candidate --allow-unlocked`; this is not certification of the
candidate scorer or corpus. `--transcript-source asr` requires explicitly supplied
`--scene` directories with valid `observations/transcript.json` sidecars. Missing
or stale observations fail before querying miners. `none` disables transcript
evidence; `legacy_labels` retains the historical label-derived hints.

Round artifacts record the scorer, validator, contract and metering hashes, the
benchmark-lock hash when present, observation budget and per-scene media/label/ASR
hashes. EMA files carry the same scoring identity. Changing it requires a fresh
round root; the validator refuses to blend incompatible reward histories before
creating a round or querying miners. Legacy unversioned EMA files remain readable
only with the historical 1.5/legacy-label configuration.
