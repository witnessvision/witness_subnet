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

The evaluator cannot write weights. Existing full burn remains active during
validation. `witness.subnet.production_weights` is the separate replacement writer.
It requires a private activation artifact with the Luna calibration, three
consecutive complete round reports and the completed operational probes. Promotion
requires 300 calibration cases with accuracy at least 95% and contradiction
acceptance at most 2%, plus 15/15 valid on-time responses for the deployment
candidate, mean F1 at least 0.98 and mean score at least 0.96.

An explicit operator instruction may waive only that initial candidate quality
threshold. Record `operator_authorization` with `policy: 70_burn_30_winner`,
`waive_initial_own_quality: true`, `authorized_at` and a nonempty `reason` in the
private activation artifact. Preserve the original reports and failed gate.
Calibration, complete comparisons, dispatch counts, valid on-time responses and
operational probes cannot be waived by this option. It does not give the candidate
priority: the complete-round hotkey EMA still selects the winner.

Stop the superseded writer and reconcile its pending commitments before activation.
The new writer holds an exclusive lock, rechecks subnet-owner burn destination
and validator permit, and never stacks a policy behind an unknown commitment.
After old commitments drain, each closed round produces one durable submission
record. All pending commitments must drain before the next policy is submitted:
timelock reveals can arrive out of submission order, including within one epoch.
Restarts never resend a recorded decision. A running round waits
for its result; a stalled round, failed preparation or incomplete comparison
requests full burn. Registration changes also invalidate an earlier winner.
Without a complete current comparison and valid winner it requests full burn.
An ambiguous submission stops for reconciliation. A submitted/finalized commit
does not prove active weights: verify reveal events, actual weights and pending
commit state before reporting activation. Operational regression requires rollback
and reconciliation before returning to the previous full-burn service.

Calibration reports must include provenance and uncertainty intervals. Shared
templates/originals are dependent observations. Accuracy on indexed content
measures benchmark concordance; it does not establish general video understanding.
