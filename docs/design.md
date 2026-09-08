# Scene format reference

This document describes validator-side scene data. Miners implement the
[public task protocol](subnet.md) and [reconstruction contract](reconstruction.md);
they do not receive `scene.json` or need to reproduce a particular generator.

Programmatic scenes support deterministic local integration checks. Recomposition
adds controlled edits to source footage. Renderer-generated labels must still be
validated against observable media, and incomplete native annotations limit
what evaluation claims a dataset can support.

## scene.json (v1.5 contract)

`scene.json` stores the declared reference annotations. Frame numbers and audio sample indices are
canonical; decimal seconds are derived conveniences. Visual intervals are
half-open (`start_frame <= frame < end_frame`) so a cut or state transition has
one unambiguous first frame. Audio intervals use the equivalent half-open sample
rule. The renderer consumes this contract rather than a separate hidden script.

The original v0 keys remain. v1.1 added `schema_version`, `renderer_version`,
`duration_frames`, `timebase`, stable IDs, initial positions/visibility, canonical
frame/sample coordinates, explicit camera crops, evidence frames for QA,
provenance, and `validation_checks`. v1.2 adds observable entity identity fields
and structured QA `references`. Events record enough transition data to
reconstruct state (`from_position`, `to_position`, `end_frame`, colors, or drop
position). These additions make exact media validation possible without changing
the JSON format or removing v0 fields.

v1.3 is the backward-compatible real-video extension. It adds a root `source`
object with the selected pool item's `id`, URL, title, uploader, license,
duration, portable pool path, and manifest hash. `foreign_sources` is empty
unless an intentional foreign-shot error is present. Recomposition shots
add `segment_id`, source interval, playback rate, transform flags, and one of the
`source`, `repeat`, `freeze`, or `foreign` kinds. Existing synthetic scenes remain
valid v1.2 documents.

v1.4 keeps those fields and removes the fixed recomposition structure. For each
seed it samples the base-segment count (tier 1: 2–5, tier 2: 4–9, tier 3: 6–12),
frame-level durations that do not land on whole seconds, and independent edit
presence and placement. Dialogue count/timing/text, audio events, text overlays
and positions, and semantic-error presence/placement are also seeded. The tier
is disclosed but implies only difficulty ranges; it does not determine a cut
count, edit inventory, error inventory, or timestamp.

v1.5 keeps the edit-plan contract and makes exact-truth overlays perceptually
fair. Each `on_screen_text` item declares one of six real-world styles
(`lower_third_bar`, `corner_caption`, `centered_title`, `subtitle_strip`,
`watermark`, or `price_tag_ui_chip`), its system font, box/color/opacity,
outline, shadow, and an `observability` record. Normal text lasts at least 0.8 s,
uses a font size of at least 18 px at 640×360, and names a frame on the 4 fps
grid that the independent validator also checks after downsampling to 320×180.
Only tier 3 may use the explicit `tier_3_flash` exception, and those flashes
last at least 125 ms. Dialogue source mixing varies the base gain by seed and
records a conservative per-line TTS-to-source SNR lower bound of at least 12 dB.
Foreign inserts prefer another source with the same per-video pool query tag;
when no such tag or candidate exists, selection falls back to any other source.

```json
{
  "schema_version": "1.2", "renderer_version": "witness-0.1.0",
  "seed": 123, "duration": 32.0, "fps": 24, "resolution": [640, 360],
  "debug_labels": false,
  "duration_frames": 768,
  "timebase": "frame timestamps are canonical; intervals are [start_frame, end_frame)",
  "difficulty": 1,
  "summary": "...",
  "actors": [{"id": "A", "name": "Ari", "visual_description": "the blue circle", "name_grounded_by": null, "color": "blue", "shape": "circle", "initial_position": [-60, 252]}],
  "objects": [{"id": "obj1", "name": null, "visual_description": "the red mug", "name_grounded_by": null, "kind": "mug", "color": "red", "initial_position": [330, 270], "initial_visible": true}],
  "shots": [{"id": "shot1", "start_frame": 0, "end_frame": 300, "start": 0.0, "end": 12.5, "camera": "wide", "crop": [0, 0, 640, 360]}],
  "events": [{"frame": 74, "t": 3.083333, "actor": "A", "action": "pick_up", "object": "obj1"}],
  "dialogue": [{"speaker": "B", "start_frame": 221, "start_sample": 203044, "end_sample": 244939, "start": 9.208345, "end": 11.108345, "text": "The meeting is at six.", "tts": {"engine": "espeak-ng", "voice": "en-us", "rate": 150}}],
  "on_screen_text": [{"id": "clock", "start_frame": 96, "end_frame": 192, "start": 4.0, "end": 8.0, "text": "5:45", "role": "clock", "bbox": [536, 18, 624, 48]}],
  "audio_events": [{"frame": 360, "t": 15.0, "start_sample": 330750, "end_sample": 342878, "kind": "door_slam", "relation": "off_screen"}],
  "intentional_errors": [{"frame": 341, "t": 14.208333, "type": "continuity", "object": "obj1", "before": "red", "after": "blue"}],
  "qa": [{"id": "temporal_1", "q": "What did the blue circle do after picking up the red mug?", "a": "moved right", "type": "temporal", "evidence_frames": [120], "references": [{"entity_type": "actor", "id": "A", "surface": "the blue circle", "grounded_by": "visual_description"}]}],
  "audio": {"sample_rate": 22050, "channels": 1, "tts_engine": "espeak-ng"},
  "validation_checks": [{"kind": "hard_cut", "frame": 300, "minimum_mean_difference": 8.0}],
  "provenance": {"generator": "witness.scene.build_scene", "seed": 123, "scene_sha256": "..."}
}
```

A recomposed v1.5 scene adds records such as:

```json
{
  "schema_version": "1.5",
  "source": {"id": "youtube-id", "url": "https://www.youtube.com/watch?v=...", "title": "...", "uploader": "...", "license": "Creative Commons Attribution license (reuse allowed)", "duration": 877.0, "path": "videos/youtube-id.mp4", "manifest_sha256": "..."},
  "foreign_sources": [],
  "shots": [{"id": "shot_1", "kind": "source", "segment_id": "segment_2", "source_id": "youtube-id", "source_start": 83.2, "source_end": 94.2, "playback_rate": 1.0, "mirror": false, "tint": null, "start_frame": 0, "end_frame": 264}],
  "events": [{"frame": 264, "t": 11.0, "action": "cut", "shot": "shot_2"}],
  "on_screen_text": [{"id": "marker_alpha", "role": "caption", "text": "HARBOR LIVE", "style": "lower_third_bar", "font_family": "Lato Heavy", "font_size": 26, "opacity": 0.9, "outline_width": 1, "shadow_x": 2, "shadow_y": 2, "observability": {"guaranteed": true, "sampling_fps": 4, "resolution": [320, 180], "sample_frame": 126, "minimum_text_height_px_at_640x360": 18}}],
  "audio": {"source_mix": {"gain_db": -16.0, "dialogue_duck_gain_db": -32.4, "dialogue_duck_padding_seconds": 0.3, "minimum_dialogue_source_snr_db": 12.0, "snr_method": "conservative_tts_rms_vs_ducked_source_peak_bound"}}
}
```

The v1.3+ edit-event vocabulary is `cut`, `speed_change`, `freeze`, `repeat`,
`insert_foreign`, `mirror`, and `tint`. The events family also carries exact
`dialogue`, `text`, and `audio` anchors. All use the same canonical frame/time
rules as synthetic actions. The legacy synthetic action vocabulary is unchanged.

The illustrative decimal audio values above need not lie on frame boundaries:
dialogue ends come from the real synthesized clip length and are exact at the
declared audio sample rate. eSpeak's variable terminal callback silence is padded
to the next 100 ms boundary; that final PCM clip is what is mixed and timed.
Generated files, rather than this abbreviated example, are normative.

`debug_labels` controls diagnostic actor names, internal object IDs, and camera
tags burned into the image. It defaults to `false` and must be `false` for tiers
2 and 3 so OCR cannot reveal tracking or shot ground truth. Internal IDs remain
available in JSON for scoring, but generated QA uses natural descriptions such
as “the purple book,” never those IDs.

Each actor and object records `name`, `visual_description`, and
`name_grounded_by`. A QA may use the stable visual description directly. It may
use a name only when `name_grounded_by` is `dialogue` or `label` and that
grounding is present in the rendered scene. `null` means the private name is not
observable and therefore cannot appear in QA. Every concrete QA entity mention
is also listed in `references`, making this rule mechanically testable.

## Difficulty tiers
1. 2–5 source segments, shorter scenes, fewer overlays and lower edit/error probabilities.
2. 4–9 source segments, broader dialogue/overlay counts and medium edit/error probabilities.
3. 6–12 source segments, densest scenes and the highest probability of fleeting text or semantic contradictions.

No structural grammar is available to a miner. Every optional edit and error may
be present or absent at every tier, and all placements vary by seed. Tier reveals
only sampling ranges and probabilities; predictions still require perceptual
evidence.

## Observation and scoring references

See the [observation API](observations.md) for evidence access and metering, and
[versioned scoring references](index.md) for matching rules.

## Source separation

Keep source identity, license, media hashes and foreign-insert dependencies in
validator-side manifests. Assign connected source groups to partitions before
creating clips. Every source in a scene must belong to the same partition.
A deterministic public generator does not establish an independent holdout or
resistance to source-aware attacks. See [Evaluation](evaluation-v2.md).
