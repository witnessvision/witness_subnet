# Grounded video benchmark 3.0 (local candidate)

Schema `3.0` / scorer `3.0.0` is an explicitly selected local diagnostic
candidate. The default scorer remains `1.1.0`. This document specifies behavior;
it does not certify general video understanding or announce a network upgrade.
The base miner remains an empty, model-independent implementation.

When explicitly selected with `--mainnet --score-version 3.0.0`, the preset
requires a reviewed `--pool-manifest` and generates three interaction worlds
and two natural tasks per round. It retains the five-round window, EMA 0.2 and
existing 70% burn / 30% winner policy (full burn when nobody is eligible).
Use a fresh round-history directory when switching scorer versions.

## What the tasks measure

A full round contains both content slices:

- Fresh interaction worlds: identify the visible actor, object, action, recipient
  and destination, reconstruct intervals, follow repeated interactions in order,
  and connect spoken instructions to what subsequently happens in the image.
  Actions are `carry`, `push`, `lift`, `touch`, and `pass`. A badge identifies each
  actor; changing colors, camera position and unrelated movements are distractors.
  Interaction counts, durations and intervening gaps vary; the task has no fixed
  action calendar that a miner can fill in without locating events in the image.
  Carrying/lifting raises the object away from its ground shadow; pushing/touching
  leaves it at ground height. Carrying/pushing changes its zone, while lifting/
  touching leaves it in place. Passing changes the supporting actor before the
  recipient moves the object. These are schematic visual action definitions.
- Reviewed natural action episodes: identify observable actions, tools and affected
  objects in real footage, then reconstruct their newly composed temporal order.
  Public questions provide action descriptions with fresh codes such as `K3`,
  including plausible alternatives that never occur. Brief gray images separate
  episodes; ordinary camera cuts within an episode do not add another episode.
  Each episode retains its complete reviewed source interval, with playback
  normalized to an independently sampled duration of 5–7 seconds. Source interval
  length therefore cannot disclose its action code. This measures action and
  order recognition on time-normalized footage, not native-speed motor dynamics.

The tasks do not score hidden intentions, material properties, or invisible
causes. Natural labels need recorded visual review and source provenance. A
single review is not independent human adjudication. Evaluation partitions must
separate sources and uploaders before constructing clips or temporal variants.
Do not inherit a full interval's annotation into an arbitrary shorter excerpt:
the named object or tool may be hidden throughout that excerpt. Review labels at
the observable level; context must not imply an invisible tool, color or intent.

The natural source library is finite. New temporal compositions prevent replaying
an entire previously submitted answer, but cannot prevent a miner from recognizing
known source clips in a labeled reference library. Report previously seen and new
sources separately. A repeatedly exposed library cannot support claims about
unseen video, even if its generated task seeds are fresh.

## Reconstruction contract

The task exposes only duration, FPS, tier, schema version and question ID/text.
Seeds, labels, source paths, composition plans, supporting event IDs and speech
validation samples are private. Observe the video through the metered API.

Submit an object with `events`, `qa`, and any applicable auxiliary families:

```json
{
  "events": [
    {
      "start": 2.9,
      "end": 4.5,
      "actor": "A",
      "action": "pass",
      "object": "mug",
      "target": "right",
      "recipient": "B"
    }
  ],
  "qa": {
    "history_mug": "A pass to B right > C touch right"
  }
}
```

The example illustrates field syntax, not a complete task answer. In interaction
worlds all five semantic fields are required. Use `recipient: null` for actions
other than `pass`. A unique entity ID or its unambiguous description is accepted.
Actor `A` is an identity, not a removable article. Bare initial colors/shapes are
not identity aliases; actor colors can change during the video. A destination is `left`,
`center`, or `right`, measured after the action. Include every actual interaction;
approaches without contact are not interactions.

Natural episode events require only `start`, `end` and the public `action` code.
Return each requested ordered history as a string separated by `>` or a list of
strings. Unknown alternatives, reordered histories and incomplete histories do
not receive exact-answer credit. Correct answers must also be supported by the
corresponding correctly reconstructed events; contradictory events cannot back an
otherwise correct summary. Submitted event IDs are unnecessary and confer no
credit.

Seconds and `start_frame`/`end_frame` are supported. When both are supplied they
must agree within half a video frame. Intervals must be finite, nonempty and
inside the video. Matching requires the declared semantic fields plus either
interval IoU at least `0.5` or both boundary errors at most `0.35` seconds. One
prediction cannot recover multiple truth events.

Auxiliary dialogue/text require exact normalized text and matching intervals;
audio events and intentional errors require their declared semantic content and
point timing. Absent truth families contribute no free credit. Reconstruction
payloads are limited to 262,144 serialized bytes and 512 entries per family.
Malformed predictions fail without aborting the whole round. Invalid private
labels or failed media validation invalidate a sample instead of penalizing a
miner for an unanswerable task.

## Quality, reward and shared credit

Compute precision/recall F1 within each nonempty family. Base family weights are
`events=0.45`, `qa=0.30`, `dialogue=0.10`, `shots=0.05`,
`on_screen_text=0.04`, `audio_events=0.03`, `intentional_errors=0.03`, normalized
over families present in the sample. A question receives credit only when its
exact ordered answer and all supporting events match. The history group is
required in both slices; the audio-grounding group is additionally required in
interaction worlds.

Each of the three history questions is a complete sequence. Passing that group
therefore requires at least two fully correct, supported histories. Event F1 is
also reported so partial recognition remains visible even when no reward is earned.

Quality is the minimum of weighted family F1, recovered semantic credit mass,
overall claim precision, event F1, and accuracy in every required question group.
Invented content in absent families and answers to nonexistent questions count
against overall precision; leaving those families empty grants no free credit. Extra alternatives count
against precision. High auxiliary scores cannot compensate for absent events or
failed required questions.

The existing fixed reward formula remains:

- `quality < 0.4`: zero reward.
- `quality >= 0.4`: `quality * efficiency`, before sharing with other miners.
- Efficiency follows the existing authoritative observation-cost formula.

Crossing `0.4` does not assign score `1`. Client-reported costs are not authoritative.

Each recovered truth fact has a fixed normalized credit weight. Eligible miners
sharing that fact divide its weight. A miner's pre-sharing reward is multiplied
by the fraction of its recovered credit remaining after this sharing. This also
covers partial copies, omitted events, aliases, changed ordering of event rows,
and irrelevant metadata. Total shared reward is bounded by one per scene, up to
floating-point rounding. This is credit conservation, not proof of independent
computation or complete resistance to multiple identities controlled by one owner.
In particular, if one identity obtains another's correct answer externally, its
lower measured observation cost can still outrank the original under
winner-takes-all. Sharing alone does not close collusion or prove that the miner
personally performed the computation. Do not describe this candidate as a complete
anti-copying protocol.

Within a full hybrid round the score is the **minimum of the two content-slice
means**. Missing responses count as zero using the attempted-scene denominator.
A miner cannot retain a high full-round score by skipping the natural slice.
Single-slice diagnostics remain possible and are explicitly identified in scoring
metadata. Existing rolling means, EMA and burn policy then apply unchanged.
Historical scorer versions retain their historical aggregation behavior.

## Validity and adversarial evaluation

Before serving a grounded sample, validate decoded media, frame count, duration,
private-label consistency and the applicability of every question. Natural
samples bind reviewed annotations and source bytes to the generated episode
pixels. Interaction worlds validate visible object positions, physical action
constraints and decoded speech against the exact synthesized PCM.

Matched interaction controls keep questions, initial/final decoded images and
speech identical while changing actions. Lossless encoding of the synthetic
video avoids residual compression differences in otherwise identical endpoints.
Matched natural controls retain the exact decoded frame multiset and endpoints,
but reorder episodes. The paired answers must consequently differ.

A validation report must separately state results for empty/question-only
outputs, transcript-only information, endpoints, shuffled order, wrong semantic
results, mass alternative guesses, actor movement without action grounding, and
exact/partial copy multiplication. Clearly label any control granted oracle
intervals, true histories or auxiliary labels. Oracle correctness alone does not
establish perceptual solvability; include decoded-media evidence and an actual
visual baseline. Report source/world independent units and all attempted tasks,
including failures, not just successful clips.

Freeze benchmark and controls before candidate tuning. Freeze the candidate
before fresh held-out worlds and source-disjoint natural evaluation. Keep private
miners, training libraries, reviewed media, operational notes and run artifacts
outside the public repository. A local test pass is neither a mainnet deployment
nor evidence of agreement among network validators.

## Local smoke check

With the README dependencies installed, choose fresh output directories and run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from witness.grounded import generate
generate(102, 1, Path("data/grounded-demo"))
PY
.venv/bin/witness-validator --dry-run --scene data/grounded-demo --scenes 1 \
  --score-version 3.0.0 --allow-unlocked --transcript-source none \
  --no-set-weights --round-root rounds/grounded-demo
```

The empty base miner earns zero. This checks one procedural task, media validation,
transport and scoring locally; it neither evaluates a competitive miner nor
constitutes a full hybrid benchmark. Natural-source annotation pools remain private.
