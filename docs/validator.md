# Run a Witness MP4 validator

Witness production uses **signed MP4 transport 5.2 on Finney SN20**. It sends five
shared clips per round to every registered endpoint advertising an IP and port.
A remote evaluator compares miner events with private human annotations. A
separate writer requests **70% burn / 30% to the eligible EMA winner** after
activation; incomplete comparisons request **100% burn**.

The validator needs CPU, FFmpeg and **one funded API credential**. No local GPU
or inference model is required. The legacy `witness-validator --mainnet` command
still runs synthetic scenes and full burn; it does **not** start MP4 production.
Its instructions remain in the [legacy guide](validator-legacy.md).

## 1. Install and prepare private state

Use Linux, Python 3.11+, FFmpeg/ffprobe, a registered SN20 hotkey with a validator
permit, and a Finney RPC. Allow outbound access to announced miner ports, the API
provider and video source. This flow does not use the old inbound observation
server on port 8765.

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv ffmpeg
git clone https://github.com/witnessvision/witness_subnet.git
cd witness_subnet
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Run under a dedicated service user with private directories (`0700`). Keep the
hotkey in its wallet directory and the coldkey private key off the server. Use
your established secure wallet-unlock mechanism under that user; never put wallet
passwords or key material in JSON, shell arguments or API env files.

The example paths below must exist and be accessible by the service user. Keep
state outside Git and release directories. Preserve the scheduler database,
reports, caches and budget ledger across upgrades.

## 2. Choose the evaluator

| `judge_provider` | Entry required in the private env file | Default model / effort |
| --- | --- | --- |
| `saygm` (default) | `GM_API_KEY` | `gpt-5.6-luna` / `low` |
| `openai` | `OPENAI_API_KEY` | `gpt-5.6-luna` / `low` |

Set the selected credential in a private file such as
`/var/lib/witness-validator/provider.env`, using a secret manager or secure
editor, and set permissions to `0600`. The JSON `env` field points to that file.
The production evaluator reads it explicitly: an exported key alone or a `.env`
in the checkout does not satisfy this entrypoint. No second key is required.

Both adapters use Responses API structured output. A key alone does not establish
model access or credit: verify a successful evaluation with the selected account
before launch. HTTP 401/429, exhausted budget and invalid judge output leave
evaluation pending. No automatic provider fallback or retry occurs; these failures
are not bad miner responses.

Provider, model, effort, prompt, adapter and calibration define the ranking series.
Changing them requires matching calibration and a **new round-state root**. Keep
the authoritative budget ledger unchanged when changing providers or keys.

The current judge is `fields-only-v2`. Luna judges each field; code derives the
overall relation, so a second model-generated summary cannot contradict those
fields. Every required field must still be valid. Upgrading from `all-fields-v1`
requires fresh calibration, a separate cache and a new ranking root; do not restart
this version against the old scheduler identity. See [judge migration](production-v5.2.md#evaluator-and-budget).

The shipped policy allows **$9 per validator per UTC day**, including calibration
and reevaluation. Its companion miner allocation is $1/day. The fixed role limits
bound one deployment's combined spend to $10; this is not a subnet-wide allowance.
Use one ledger for every paid validator process on the host, outside release/cache
directories. Reservations survive cancellation and restarts. Do not reset the
ledger or create another one to bypass limits.
See [budget accounting](production-v5.2.md#evaluator-and-budget).

## 3. Provision the catalog and calibration

**Cloning the repository and adding a key is not sufficient.** Private annotation
catalogs, source manifests, calibration cases/results and competitive miner code
are not distributed here. Provision these evaluation assets through the operator's
private data process before starting the service.

The source adapter expects an available ActivityNet annotation catalog and the
pinned mirror's `metadata.jsonl` and `README.md`. `mirror_root` contains that
metadata; videos are fetched during preparation into a bounded cache. The adapter
checks the revision/hashes in `witness.sources.activitynet_mirror.PIN`. An arbitrary
MP4 folder is not a compatible catalog. Review source access and terms before use;
keep original identifiers, names and annotations accessible only to the validator.

The catalog must include its `availability` evidence and only eligible, available
originals. `witness.sources.available_catalog` derives it from the annotation
catalog, pinned mirror and recorded permanent source failures. Do not filter using
a miner's index or performance. The production operator's frozen catalog has 6,123
originals; `catalog_count` must match **your approved artifact**.

Compute its canonical hash and count, not a hash of the file bytes:

```bash
.venv/bin/python - /private/catalog.json <<'PY'
import json, sys
from witness.events import content_hash
catalog = json.load(open(sys.argv[1]))
print(json.dumps({"catalog_hash": content_hash(catalog),
                  "catalog_count": len(catalog["originals"])}))
PY
```

Calibrate the selected evaluator with 300 distinct labeled cases using
`witness.events_evaluation.calibrate` and `JudgeConfig.build`. Load the selected
key with `witness.providers.load_key`; pass the same `budget_path` and judge
configuration as production. This makes paid calls. Persist the returned report
privately. The case schema is enforced by `calibrate`; generic examples are in the
public evaluation tests. Real cases and miner-specific tests remain private.

Require accuracy at least 95%, contradiction acceptance at most 2%, matching
`evaluator_id`/`prompt_hash`, and `passed: true` from the actual run. Retain
provenance, uncertainty intervals and case dependence. Never fabricate a passing
report or reuse another evaluator's result. Indexed benchmark concordance does
not establish general video understanding.

## 4. Configure and run evaluation

Create `/var/lib/witness-validator/validator.json` with this structure. Replace
identity, paths, catalog hash/count and starting epoch with verified values.
`start_after_epoch` is the last epoch to skip for a **new** root; thereafter the
persisted cursor controls starts. Set it to the current finalized SN20 epoch to
start at the next epoch. Never erase history to replay earlier epochs.

```json
{
  "root": "/var/lib/witness-validator/v5.2/rounds",
  "network": "finney",
  "wallet": "MY_WALLET",
  "wallet_hotkey": "MY_HOTKEY_NAME",
  "wallet_path": "/var/lib/witness-validator/wallets",
  "expected_hotkey": "MY_VALIDATOR_SS58",
  "catalog": "/var/lib/witness-validator/data/catalog.json",
  "catalog_hash": "CANONICAL_CATALOG_HASH",
  "catalog_count": 6123,
  "mirror_root": "/var/lib/witness-validator/data/mirror",
  "start_after_epoch": 0,
  "env": "/var/lib/witness-validator/provider.env",
  "judge_provider": "saygm",
  "judge_model": "gpt-5.6-luna",
  "judge_effort": "low",
  "judge_cache": "/var/lib/witness-validator/v5.2/judge-cache",
  "budget_path": "/var/lib/witness-validator/budget/daily.sqlite3",
  "calibration": "/var/lib/witness-validator/v5.2/calibration.json"
}
```

Run from the installed checkout under the service user:

```bash
.venv/bin/python -m witness.subnet.production \
  --config /var/lib/witness-validator/validator.json
```

**This process cannot write weights.** Keep the existing full-burn writer during
validation. Use your service supervisor with an absolute interpreter/config path,
a stable working directory, `UMask=0077` and `StandardInput=null`. Configure hotkey
access before unattended launch. The evaluator has no `--once` or
`--no-set-weights` flag.

Inspect `latest.json` and `rounds/*.json` under `root`; `scheduler.sqlite3` is
authoritative. Check `complete`, each miner's `planned`/`sent`/`valid`/`scored`,
latencies, F1, score and winner. Every announced endpoint gets five planned tasks,
including incompatible miners. Five sends do not imply five valid responses.
There are four concurrent sends globally and one per miner, with no automatic
retries, restart duplicates or bursts to catch up missed epochs.

Invalid/absent miner responses score zero. Evaluation failures leave comparisons
incomplete and prevent EMA updates. Saved signed responses and semantic decisions
support offline reproduction without resending. See the
[wire, scoring and ranking contract](production-v5.2.md).

## 5. Activate one weight writer

Keep 100% burn until calibration, three consecutive complete rounds and operational
acceptance pass. The [activation contract](production-v5.2.md#weight-handover) defines
the 15-response quality/latency gate, cancellation/restart probes and the narrowly
scoped operator exception. Do not mark unperformed checks passed.

Prepare a private activation JSON containing `reports` (three complete reports),
`calibration` (the matching report), `own_hotkey` (the deployment candidate's hotkey)
and `operational_probes_passed` backed by saved evidence. Validators do not need to
run a competitive miner themselves, but this writer still requires a qualified
candidate in the activation evidence. That candidate receives no priority in ranking.

Save this separate writer config as `/var/lib/witness-validator/weights.json`:

```json
{
  "root": "/var/lib/witness-validator/v5.2/weights",
  "activation": "/var/lib/witness-validator/v5.2/activation.json",
  "scheduler": "/var/lib/witness-validator/v5.2/rounds/scheduler.sqlite3",
  "network": "finney",
  "wallet": "MY_WALLET",
  "wallet_hotkey": "MY_HOTKEY_NAME",
  "wallet_path": "/var/lib/witness-validator/wallets",
  "expected_hotkey": "MY_VALIDATOR_SS58"
}
```

Stop the superseded writer and disable its restart/timer. Reconcile its pending
commitments, then start the replacement with secure wallet access:

```bash
.venv/bin/python -m witness.subnet.production_weights \
  --config /var/lib/witness-validator/weights.json
```

**This command submits weights.** Use one writer per hotkey. The implementation
requires systemd and checks that `witness-burn.service` is inactive or failed;
also stop any writer with a different service name. The exclusive lock protects
only processes sharing the same writer root. Configure the supervisor not to
restart exit code 78, which requires operator reconciliation.

A complete round with an eligible winner requests 70% burn / 30% winner. A running
round waits normally; incomplete, unavailable or stale comparisons request full
burn. Hotkey EMA uses alpha 0.2, with lower UID breaking a tie. The writer rechecks
registration and the owned burn destination.

Decisions and receipts persist under `weights/submissions/`; restarts do not
resubmit them. All pending commitments must drain before the next submission:
timelock reveals can arrive out of order. A closed round therefore does not
guarantee a newly active vector in the same epoch. Inspect
`weights/observation.json` for source status, desired vector and pending commits.
An ambiguous submission stops; preserve its receipt and reconcile chain state
before restarting. Do not delete the journal to bypass the guard.

## 6. Verify active weights and consensus

At one **finalized block**, check the validator hotkey/UID/permit, owned burn
destination and `RecycleOrBurn`, actual `Weights`, native reveal events and pending
timelocked/legacy commitments. A finalized commitment is not proof of active
weights. Normalize integer weights by their row sum; the chain need not store
literal 70 and 30.

Then inspect `Consensus`/`Incentive` for miners and `ValidatorTrust`/`Dividends`
for your validator, alongside other permitted validators' weights. These metrics
come from the latest mechanism step; newly revealed weights can be newer than
that step. Your 70/30 vote does not guarantee a 70/30 subnet payout. Yuma combines
stake-weighted validator opinions; see [Bittensor's consensus documentation](https://www.bittensor.com/docs/internals/consensus).

## Upgrade without losing evidence

Pin, install and test a release before switching services. Preserve the old
scheduler, evaluation artifacts, activation evidence and submission receipts.
Keep the budget ledger in place. An evaluator identity change requires a fresh
calibration and ranking root; a documentation update does not. Never run old and
new writers concurrently. Reconcile pending commitments before a handover or
rollback to full burn.

[Production contract](production-v5.2.md) · [Legacy validator](validator-legacy.md) ·
[MP4 miner integration](production-v5.2.md#wire-contract)
