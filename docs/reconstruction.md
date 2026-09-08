# Witness miner reconstruction contract (v1.0)

`reconstruction.json` is the miner's only scored output. Its root is a JSON
object with the following shape. Unknown fields are ignored; missing scored
families are treated as empty.

```json
{
  "schema_version": "1.0",
  "events": [{"frame": 180, "actor": "the orange circle", "action": "pick_up", "object": "the purple book"}],
  "dialogue": [{"speaker": "B", "start": 20.0, "end": 22.3, "text": "I picked up the red book."}],
  "shots": [{"start_frame": 0, "end_frame": 240}],
  "on_screen_text": [{"start_frame": 120, "end_frame": 192, "role": "sign", "text": "ROOM 204"}],
  "audio_events": [{"frame": 408, "kind": "door_slam"}],
  "intentional_errors": [{"frame": 300, "type": "continuity", "object": "the purple book"}],
  "qa": {"temporal_1": "the orange circle", "count_1": "2"}
}
```

Frames are canonical. Point items may provide decimal `t` instead. Text
intervals may provide decimal `start`/`end`. Dialogue uses decimal seconds
because synthesized speech ends need not fall on a frame boundary; it may use
`start_frame`/`end_frame` as a lower-precision alternative. QA must be an object
keyed by the `qa[].id` values from the task specification.

Miners must identify actors and objects with their observable
`visual_description` from scene contract v1.2, such as `the orange circle` or
`the purple book`; internal IDs such as `A` and `obj1` are not visible miner
inputs. The scorer retains ID support for validator or oracle use. Descriptions
are case-insensitive, ignore English articles, punctuation, and repeated
whitespace. A color alone (`orange`) or shape/kind alone (`circle`, `book`) is
accepted only when it identifies exactly one entity of that family in the
scene. Ambiguous partial descriptions never resolve.

Malformed miner output never aborts validator scoring. A malformed family item
(including a missing/invalid timestamp, wrong type, negative interval, or
unresolvable entity) is omitted from matching and counted as a false positive
for that family. The report's `diagnostics` array records its family, input index,
and reason. A non-object reconstruction root or non-object `qa` produces quality
and final score zero and includes a top-level `invalid_reason`.

For scene contract v1.3 through v1.5, `events[].action` also accepts edit actions `cut`,
`speed_change`, `freeze`, `repeat`, `insert_foreign`, `mirror`, and `tint`, plus
the exact overlay anchors `dialogue`, `text`, and `audio`. These are scored by
the same temporal event matcher as the v1.2 synthetic actions. Source provenance
and per-shot source intervals remain validator truth; miners do not need to copy
them into `reconstruction.json`.

Scene v1.5 adds renderer-only style and observability metadata to each truth text
item. Miners still submit only role, text, and interval. Normal text is guaranteed
to last at least 0.8 s, be at least 18 px tall at 640×360, and intersect the
declared 4 fps grid at 320×180. Only an item explicitly marked as a tier-3 flash
may use the exception, and it lasts at least 125 ms. Scripted dialogue records a
conservative TTS/source SNR lower bound of at least 12 dB; source speech remains
unscored background.

## Scoring compatibility

The scene schema, reconstruction schema and scorer version are separate
identities. The selected scorer determines matching and reward behavior; do not
infer it from the reconstruction's `schema_version`.

- [v1.5](benchmark-v1.5.md) defines the historical frozen rules, including
  interval-matched dialogue with word-error credit. It has known reward weaknesses.
- [v1.6 candidate](scoring-v16-candidate.md) changes false-positive and QA credit.
- [v1.7 candidate](scoring-v17.md) changes temporal, duplicate and public-identity matching.
- [v1.8 development](benchmark-v18.md) changes dialogue speaker handling.

Use public questions and observable entity descriptions to construct a response.
The validator supplies cost from its own session records. Unknown or incomplete
reference annotations are evaluation limitations, not permission to infer private
labels or omit observable evidence solely to fit a development corpus.
