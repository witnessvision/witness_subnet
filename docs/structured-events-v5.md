# Structured events v5

Schema `5.0` and quality scorer `5.0.0` define a diagnostic annotation-concordance
benchmark. Previous schemas and scorers remain available. v5 never submits
weights. An independently operated burn service remains the sole weight writer.

## Request and response

The signed `WitnessTask` retains the existing observation capability and includes
a strict `task_spec`: `schema_version`, `task_type=structured_events`, measured
`duration` (60–120 seconds), native `fps`, `has_audio`, `response_language=es`,
`max_response_bytes=2097152`, `max_events=2048`, and `miner_deadline_s=170`.
Task, scene and session identifiers are random. No dataset identity, source URL,
label vocabulary, question, answer option or reference-derived field is sent.
Visible media can still reveal its origin. Retrieval and memorization are allowed;
their performance measures this benchmark, not generalization.

```json
{
  "schema_version": "5.0",
  "events": [
    {
      "timestamp": 12.5,
      "actor": "la persona que lleva la cámara",
      "action": "abre",
      "objects": ["la puerta"],
      "details": []
    }
  ]
}
```

Events are ordered point timestamps relative to the clip in `[0,duration)`.
All five event fields are required. Text must be nonempty Spanish; semantic
evaluation accepts correct translations and paraphrases. Extra keys, free
summaries, nonfinite numbers, duplicate JSON keys, incomplete JSON and oversized
responses are rejected. Actor/action text and each object/detail are evaluated.
The total wire envelope and the complete miner response are limited to 2 MiB.
The canonical SHA-256 response hash covers reconstruction and trace. The only
v5 trace field is a bounded status; model metadata or arbitrary diagnostics are
not accepted. Existing SDK signatures cover all application request fields.

## Observations and deadlines

The existing session API serves native-time observations. Budget equals 2 fps at
640×360 and two passes through the audio; no reference-derived transcripts are
available. Audio retains native sample rate and channels. Original container/stream metadata and chapters are removed
from frame and audio observations. Sessions close at
completion or expiry. Decoder processes are cancellable; HTTP errors do not
reflect paths or input data. An external 180-second clock covers sending through
receipt of the complete body. The miner has 170 seconds including local work;
clients do not retry. Evaluation is separate and limited to 300 seconds.

Run observation services and evaluation launchers with standard input connected
to `/dev/null` (for systemd services, `StandardInput=null`). FFmpeg can interpret
inherited input as interactive commands, including changes to the diagnostic
logging used to recover native timestamps. Verify the running process's input
descriptor when launching through SSH or a wrapper script.

The base miner is deliberately empty and model independent. Subclasses should
implement asynchronous `reconstruct`; blocking inference must run in a bounded
process so timeout or disconnect kills its descendants before releasing capacity.

## Scoring and limitations

References retain their human annotation time semantics. Point narrations use
absolute point error. ActivityNet Captions descriptions retain their original
`start`/`end` intervals; prediction error is distance to that interval (zero inside),
not distance to an invented action onset or midpoint. A reference cannot mix time
semantics. No sound labels are manufactured. A semantic evaluator classifies every text field as
`supported`, `contradiction`, `unbacked` or `uncertain`. Any contradictory detail
makes the whole event contradictory; any unbacked detail prevents full support.
All reference/prediction pairs within 5 seconds of the annotated point or interval
require a saved judgment. An interval score measures weaker temporal localization
than point annotations; shifts within a long interval do not reduce concordance.
The deterministic scorer uses maximum-cardinality one-to-one matching; exact
duplicate submissions cannot acquire additional labels and count against precision.

`precision = matched / predictions`, `recall = matched / references`, and
`F1 = 2*matched / (predictions + references)`. Empty predictions score zero.
Temporal errors are reported separately. Unmatched contradictions, unbacked
events, evaluator uncertainty, competing matches and missing temporal references
are separate diagnostics. “Unbacked” is not a claim of physical fabrication.
Consumption declarations do not affect scoring; miners never share matching credit.

### Optional quality and latency reward, scorer 5.1.0

The direct MP4 benchmark reports quality scorer `5.0.0` unchanged and the
versioned `5.1.0` reward alongside it:

`speed = max(0, 1 - validator_elapsed_s / 180)`

`reward = F1 * (0.70 + 0.30 * speed)`

The validator measures upload through receipt of the complete response body.
Source acquisition, clip preparation and semantic evaluation are outside this
clock. At equal quality a faster response earns more; an empty or wholly
incorrect response earns zero even when immediate. A perfect response at 90 seconds
earns 0.85. Invalid, missing or late responses (180 seconds or later) earn zero.
The time component is gated by quality, so it is at most 0.30 and cannot compensate
for unsupported content. Reports retain F1, precision, recall, measured latency,
both reward components and the reward version. This diagnostic score never writes
chain weights or changes the separate 100% burn policy.

Reports bind reference, response, decision, prompt and evaluator identities.
Replaying saved decisions reproduces the arithmetic without resampling a model.
Quality remains provisional until the exact evaluator passes 300 frozen
calibration cases at ≥95% expected decisions and ≤2% contradiction acceptance.
The decision unit is the event/reference relation. All text fields must be
classified and determine that relation; exact agreement about which field bears
a role mismatch is an additional diagnostic, not a separate quality gate.
Wilson 95% intervals and denominators accompany both rates; dependence between
cases sharing a source/template must be disclosed. Any unresolved judgment keeps
its score provisional even with a calibrated evaluator.

## Diagnostic operations

`EventsValidator` accepts a private job provider and evaluator callback. It targets
exactly one currently registered hotkey and schedules five different originals
per observed finalized epoch, serially. The SQLite journal commits dispatch intent
before network handoff and prevents retries after crashes. Ambiguous interrupted
sends remain `interrupted_unknown`; they never count as completed. Remaining
unattempted tasks can continue after restart. Epochs crossed by a round are
recorded and consumed; missed epochs are not replayed in bursts.

Feedback contains only aggregate numeric results. Private artifacts distinguish
planned, sent, completed, rejected, expired and uncertain sends, observation calls,
feedback delivery, miner latency and evaluator latency. Missing responses and
missing evaluations never qualify acceptance. Burn and live chain weights require
separate operational verification; a diagnostic `burn_rate` field is not proof.

Activation requires dataset provenance and licensing, split isolation, all checks,
development mode selection followed by a frozen evaluation, and reversible
deployment. The production acceptance target is 15 valid responses over three
consecutive rounds, ≤180 seconds each, indexed F1 ≥0.90, reproducibility,
restart/cancellation evidence and 100% burn before/during/after. This document is
a contract, not evidence that those gates have passed.

The initial corpus is ActivityNet Captions, using the original public training and
validation descriptions (test descriptions are withheld). Validation's alternate
annotation set is preserved rather than counted as extra simultaneous labels.
Only accessible public source videos are eligible. Clips remain 60–120 seconds;
windows with partially cut descriptions are rejected, not relabeled. Originals
and known uploader channels stay in one partition. ActivityNet does not identify
all participants, so this separation cannot certify participant independence.
Publish availability, exclusion, duration, activity, and partition counts alongside
results. The downloadable corpus is not assumed to equal all catalogue entries.
