# Observation API and sources

## Assigned sessions

A network miner uses the `tool_base_url` and `session_id` from its task. The
validator creates the session; public session creation is disabled for rounds.
Observation routes are below `/s/{session_id}/`:

| Route | Evidence |
| --- | --- |
| `meta` | Public scene metadata and current cost |
| `frame` | One decoded image |
| `frames` | A ZIP of images and a timestamp manifest |
| `audio` | Decoded PCM WAV for a requested interval |
| `transcript` | Transcript entries for a requested interval |
| `search_transcript` | Matching transcript entries |

Use `witness/tools/client.py` for the client request interface. Respect the
assigned visual, audio and transcript budgets independently. Requests exceeding
a channel budget return HTTP 429 without charging that request. JSON responses
include cumulative cost; binary responses use the `X-Witness-Cost` header.

Visual units are `ceil(width / 14) * ceil(height / 14)` per frame. Audio is
metered in requested seconds and transcripts in returned text characters.
Local processing and provider inference charges are separate from these units.

## Transcript modes

Historical v1.5 transcript hints were generated from answer labels. Preserve
that behavior only for explicit historical development comparisons. New
independent evidence must use `asr` or `none`, never `legacy_labels`.

The metered server and validator accept `transcript_source`: `legacy_labels`,
`asr`, or `none`. Round identities pin the observed transcript sidecar and media.
Changing observation sources requires a separate scoring history.

## Audio-derived transcripts

Validators must prepare transcript sidecars from decoded audio before choosing
`asr`. Each `observations/transcript.json` must record its schema version,
`source: decoded_audio`, video SHA-256, model-file hashes and timestamped entries.
See `witness/tools/transcripts.py` for validation requirements. Do not derive
observations from `scene.json` or answer labels.

Start a validator round with `--transcript-source asr`. Window and search endpoints read the same ASR
artifact and charge returned characters. Speakers remain unknown because the
ASR does not perform diarization. `none` returns empty transcript observations;
miners can still use their metered audio channel.

Missing, stale or invalid ASR artifacts fail server startup. They never trigger
a fallback to label-derived hints. Removing hints does not repair incomplete
labels for original source speech or native text: those require reviewed data
under the new public task contract before any final evaluation is certified.
