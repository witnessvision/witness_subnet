# Full-clip transport 5.1

The direct HTTP adapter in `witness.mp4` delivers one complete MP4 per task.
It does not require chain registration or wallet access. Event responses retain
schema `5.0`. Annotation quality retains scorer `5.0.0`; reward policy `5.1.0`
adds the validator-measured latency component described below.

## Request

Send `POST /events` with `Content-Type: video/mp4`. The body is the standalone
clip, with audio when present. `Content-Length` must match its signed size and
must not exceed 128 MiB. Content encoding and redirects are not supported.

`X-Witness-Task` is base64 of the canonical JSON `ClipTask`. It contains only:

- transport version `5.1`, a random 128-bit task ID, issue and expiry times;
- SHA-256 and byte length of the entire MP4;
- the public `EventsTaskSpec`: duration, frame rate, audio availability, Spanish
  response language, schema, and response/deadline limits.

`X-Witness-Signature` is base64 of an Ed25519 signature over
`b"witness-mp4-request-5.1\0POST\0/events\0" + canonical_task_json`.
The server pins the validator's public key. Use TLS or an authenticated tunnel
to protect transport and authenticate the server.

The clip must use relative time starting at zero. Preparation removes source
metadata, chapters, filenames and URLs. No annotations, dataset IDs, original
offsets or reference-derived hints are part of the request. Visible or audible
content can still identify the source.

## Miner integration

Pass an async `handler(temporary_clip_path, task)` to `create_app`, together with
the pinned validator key and a private state directory. The base adapter has no
model or inference implementation. The handler returns the existing
`{"schema_version":"5.0","events":[...]}` object.

The miner can sample the full clip freely. Frame/audio observation call budgets
do not apply to this transport. Event timestamps are seconds relative to the
received clip. The miner must evaluate only its observations and independently
acquired resources; validator references and task manifests remain private.

## Limits and failure behavior

- One active request per server; concurrent requests receive `429`.
- The 170-second internal deadline starts before upload and covers processing.
- The client enforces 180 seconds from dispatch through the complete response.
- Responses are strictly validated and limited to 2 MiB. The
  `X-Witness-Response-SHA256` header hashes every byte of the response body.
- SQLite consumes each task ID once, including failed requests, across restarts.
  There are no automatic retries.
- Upload interruption, client disconnect or deadline cancels the handler. A
  subprocess-based implementation should use `run_process`, which terminates
  its process group. Temporary clips are removed when the request ends.
- Errors contain fixed codes only; they do not echo private filenames or input.

Scoring happens after receipt and does not consume the miner's deadline. Every
planned/sent/valid/failed task must be counted; a missing response is not a
successful task. Existing observation-based and chain transports remain
available under their existing contracts.

## Epoch scheduling and reward

`witness.subnet.mp4_validator.MP4Validator` accepts an injected read-only chain
view, private clip provider and private evaluator. It sends five clips from
distinct originals to one explicitly configured registered hotkey, sequentially,
once per observed epoch. The provider constructs anonymous media; references
remain on the validator. Acquisition may cross epochs: the scheduler claims the
epoch observed when the clips are ready, without backfilling missed epochs.

SQLite persists round/task claims before dispatch. An interrupted task is
recorded as unknown and never resent; unsent tasks can resume. A process lock
prevents two scheduler owners. An epoch crossed during a round is consumed,
and failed acquisition consumes the current epoch too. Reports distinguish
planned, dispatched, valid, rejected, expired, interrupted and scored tasks,
and separate miner and evaluator latency. This adapter cannot write weights.

For an on-time, valid response, `speed = max(0, 1 - elapsed_s / 180)` and
`reward = F1 * (0.70 + 0.30 * speed)`. The clock stops when the complete body is
received, before validator parsing or scoring. Missing, invalid or late responses
earn zero; failed evaluation contributes zero to the reported lower bound.
Raw F1, precision and recall remain separate from this reward. This policy does
not activate payments or change the separate burn service.
