# Production MP4 5.2

For installation, provider credentials, private assets and complete evaluator/writer
configuration examples, start with [Run a Witness MP4 validator](validator.md).

The public package contains the validator, provider adapters, scorer, transport
and an empty miner serving adapter. Private datasets, evaluation inputs/results
and competitive miner implementations are never part of this package.

## Wire contract

`witness.mp4_v5_2` retains event schema 5.0, annotation scorer 5.0.0 and latency
reward 5.1.0. Experimental `witness.mp4` remains transport 5.1.

`POST /events` sends the full MP4 with `Content-Type: video/mp4`, its exact
`Content-Length` (at most 128 MiB), and no content encoding. `X-Witness-Task`
contains base64 canonical JSON with transport version `5.2`, netuid, both SS58
hotkeys, a random 128-bit task ID, issue/expiry times, complete clip SHA-256 and
size, and `EventsTaskSpec`. No original names, dataset IDs, source offsets,
annotations or source URLs are sent. Clips have relative timestamps and stripped
metadata. Recognizable audiovisual content can still reveal a source.

`X-Witness-Signature` is base64 of the validator hotkey signature over
`b"witness-mp4-request-5.2\0POST\0/events\0" + canonical_task_json`.
The receiving miner verifies the intended miner, network and permitted validator.

A successful response is at most 2 MiB. `X-Witness-Response-SHA256` hashes the
complete response bytes. `X-Witness-Response-Signature` is base64 of the miner
hotkey signature over
`b"witness-mp4-response-5.2\0POST\0/events\0" + SHA256(canonical_task_json).digest()
+ SHA256(response_body).digest()`.
The validator verifies it with the hotkey selected from the registered endpoint.
Redirects, duplicate JSON fields and compressed response bodies are rejected.

`create_app(handler, hotkey=..., validator_hotkeys=..., netuid=..., state=...)`
wraps an async `handler(temporary_clip_path, task)`. No model is supplied by the
base miner. Requests are consumed persistently, including failed requests.
Errors contain fixed codes. There is no annotation feedback endpoint.

The external 180-second deadline covers upload and receipt of the full body.
The server's 170-second deadline starts before upload. Disconnect/deadline cancels
the handler; subprocess handlers must terminate their full process group.
Evaluation has a separate 300-second limit per response.

## Validator and ranking

Run `python -m witness.subnet.production --config /private/validator.json`.
Required configuration includes `root`, wallet identity/path and expected hotkey,
`catalog`, `catalog_hash`, `catalog_count`, `mirror_root`, `start_after_epoch`,
`env`, `judge_cache`, `budget_path` and `calibration`. Source manifests and outputs
must remain accessible only to the validator. The optional network setting defaults
to Finney and the chain genesis and SN20 identity are checked.

All registered endpoints with an announced IP and port are selected without a
compatibility filter. A round prepares five distinct random originals and sends
the same five clips to each endpoint. At most four sends run globally and one
task runs per miner. SQLite records round, clip ordinal and hotkey before sending;
requests are never retried. A restart does not recover old epochs with a burst.
Ambiguous interrupted requests and unfinished evaluations leave an incomplete
round. Saved responses are retained for explicit reevaluation without resending.

Absent, incompatible and invalid miner responses receive zero. Provider/budget
failures have no score and do not count as miner failures. Only a complete
comparison updates ranking. Scores are reproduced from stored semantic decisions.

`score = F1 * (0.7 + 0.3 * max(0, 1 - seconds/180))`.
Each complete round updates hotkey EMA as `0.2 * round_mean + 0.8 * previous_ema`,
starting at zero. A new hotkey does not inherit the old UID's EMA. Eligible miners
have a positive scored response in the complete round. Highest EMA wins; lower
UID breaks a tie. The writer rechecks registration before submission.

## Evaluator and budget

Defaults are `judge_provider: saygm`, `judge_model: gpt-5.6-luna`,
`judge_effort: low`. SayGM uses `GM_API_KEY`; `openai` uses `OPENAI_API_KEY`.
Only the selected key is loaded. Requests use Responses API structured output,
`store=false`, no tools and no automatic retry or fallback. Provider, model,
effort, output limit, adapter, prompt and calibration identify the ranking series.
Changing the evaluator requires new calibration and a new ranking state directory.
Cache identities include provider and the complete request configuration.

The production judge uses `fields-only-v2`: the model returns exactly one relation
for every supplied event field, and the validator derives the overall relation
with precedence `contradiction > uncertain > unbacked > supported`. It does not
ask the model for a redundant global relation. Missing, extra or invalid field
decisions still leave evaluation incomplete. Raw provider outputs stay in the
request cache; scorer decisions retain every field unchanged. The strict v1
`ApiJudge` and its prompt remain available for historical experiments.

Upgrading from `all-fields-v1` requires new calibration, a separate judge cache and
a fresh ranking root. Keep prior responses and reports unchanged. An explicit
reevaluation may rebuild the new series from complete sets of saved responses,
using their original request identities and measured latencies; recompute every
valid response and start EMA from zero. Never copy old scores/EMA or relabel an old
calibration. API/budget failures during reevaluation remain incomplete.

The fixed allocations are $9/day UTC for the validator and $1/day UTC for the
miner. Each role has exactly one authoritative ledger on its host, outside
release, provider and cache directories. Calibration and saved-response
reevaluations run on the validator host using its same ledger. Do not create
another ledger for a second provider, API key, process or release. The two fixed
ceilings bound the combined spend to $10; reallocations are not automatic.

Atomic SQLite reservations use integer nanodollars and synchronous commits.
Unsettled requests retain their full reservation across restarts. Settlement is
charged to the UTC day of reservation. SayGM settles from `usage.cost_nano_usd`;
OpenAI usage is valued at the pinned token rates. Unexpected settlement overruns
are recorded as debt and fail the call. Price ceilings must be reverified before
a model/rate migration. See [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
and the [SayGM model catalogue](https://api.saygm.com/v1/models).

## Weight handover

The evaluator cannot write weights. Since September 15, 2026, the separate
`witness.subnet.production_weights` writer enforces **100% burn** irrespective of
miner performance, evaluator availability and old activation evidence. The
production pause has no automatic expiry and no configuration override. A later
reviewed release is required to resume miner rewards.

Burn operation requires no calibration, catalog, scheduler or provider key. The
old `activation` and `scheduler` config entries are accepted but are not read by
the writer. Existing promotion helpers remain available for historical evidence
inspection; their result does not authorize miner allocation during this pause.
Preserve the old activation artifacts, scores and ranking series.

Stop the superseded writer and retain its submission journal/root. The new writer
holds the existing exclusive lock, rechecks the subnet-owner burn destination,
`RecycleOrBurn` and validator permit, and waits for all prior timelocked and legacy
commitments to drain. An already committed 70/30 vector can still reveal during
handover; changing software does not cancel it. Respect the chain submission rate
limit and verify the subsequent burn reveal.

One durable burn decision is produced per epoch, including while evaluation is
running, unavailable or incomplete. Policy-specific decision IDs prevent old
winner receipts from suppressing the burn update; restarts do not duplicate a
recorded decision. Ambiguous submissions stop for reconciliation. Preserve all
receipts, including the previous activation gate. The current policy is stored in
`weight-policy.json`; desired and observed vectors are in `observation.json`.

A submitted/finalized commitment does not prove active weights. At a finalized
block verify reveal events, the exact row `[[burn_uid, 65535]]` and pending commit
state. Other validators must update independently; one published release or one
validator's burn does not establish subnet-wide adoption. See the
[validator update guide](validator.md#apply-the-temporary-burn-update).
