# Witness benchmark v1.5 — frozen scoring and observability specification

**Historical specification.** Known false-positive and QA-credit weaknesses allow
rewarded predictions without adequate observations. Do not use v1.5 for
production reward allocation. Corrections require a separately versioned contract
and independent evaluation; historical rules remain unchanged.

This document is normative for the v1.5 recomposer/scorer contract. A later
corpus lock may pin scene hashes, but it must not change the rules below.

## Scored families

| Family | Weight | Identity and timing | Metric | False-positive cap |
| --- | ---: | --- | --- | ---: |
| Events | 0.25 | Exact action and resolvable actor/object; point tolerance by tier | F1 | 8 |
| Dialogue | 0.20 | Lines matched by interval only (IoU ≥0.5 or both endpoints within tier tolerance); each matched line credited 1 − WER against the normalized truth text; speaker labels are not scored | fractional F1 | 4 |
| Shots | 0.10 | Internal cut boundaries within ±2 frames; same ID when both sides provide one | F1 | 8 |
| On-screen text | 0.05 | Normalized text and exact role; interval IoU ≥0.5 or both endpoints within tier tolerance | F1 | 6 |
| Audio events | 0.05 | Exact kind; point tolerance by tier | F1 | 4 |
| Intentional errors | 0.15 | Exact type and applicable actor/object/audio/text target; point tolerance by tier | F0.5 | 3 |
| QA | 0.20 | Answer keyed by QA ID | normalized discrete credit | n/a |

The point and endpoint tolerances are ±0.50 s at tier 1, ±0.375 s at
tier 2, and ±0.25 s at tier 3. Matching is deterministic,
maximum-cardinality, one-to-one, and uses stable nearest-first/input-order tie
breaks. Empty truth plus empty prediction scores 1.0 for that family.

QA normalization applies Unicode NFKC, case folding, punctuation removal,
whitespace folding, and removal of English articles. Credit is 1 for a normalized
exact match, the complete normalized truth as a token-bounded substring, or
token F1 of at least 0.6; otherwise it is 0. Extra undeclared QA keys are ignored.

## Gate and cost model

`quality` is the weighted sum above. The default aggregate quality gate is 0.70
at tier 1, 0.65 at tier 2, and 0.60 at tier 3. Below the gate, final score is
zero. At or above it:

`score = quality × max(0, 1 − 0.30 × cost / cost_ref)`

`cost` is the validator-supplied sum of non-negative finite `visual_tokens`,
`audio_seconds`, and `transcript_chars`. The reference is a full 1 fps 640×360
read:

`cost_ref = ceil(duration_seconds) × ceil(640 / 14) × ceil(360 / 14)`

The miner cannot self-report or discount tool cost.

## Recomposition contract

The scene schema is `1.5`; frames are canonical at 24 fps and visual intervals
are half-open. The base source, segment selection/order, arbitrary-frame shot
durations, optional repeat/freeze/foreign insert, speed, mirror, tint, overlays,
dialogue, sound events, and semantic errors are deterministic functions of the
seed and manifest. Tier exposes ranges and probabilities, not a fixed structure.
Rendered H.264 samples use a one-second keyframe interval so metered random-frame
reads do not depend on long decoder seeks.

Foreign inserts select another manifest item with the same explicit per-video
`query_tag` (or equivalent `query`, `format_tag`, `format`, `tag`, `genre`, or
`category`) when one is available. With no tagged peer, they fall back to any
other source. The chosen policy and whether a same-genre match occurred are
recorded in scene truth.

Scripted dialogue contains 6–14 words using statements, questions, requests,
numbers, names, times, and places. Voice is selected from all 13 supported
OpenAI voices and speed from the declared five-value set. Source audio gain is
seeded from −18, −16, −14, or −12 dB. During each dialogue window (including
0.3 s padding), the source is ducked far enough that the recorded conservative
TTS-RMS versus source-peak lower bound is at least 12 dB. Source speech remains
background and is not dialogue truth.

## Text observability guarantee

Normal on-screen text uses one of six semantic styles: lower-third bar, corner
caption, centered title, subtitle strip, watermark, or price-tag/UI chip. The
renderer varies at least seven installed DejaVu/Liberation/FreeSans/Lato font
files, font size, color, opacity, background, outline, and shadow.

Every non-flash text truth item:

- lasts at least 0.8 s;
- uses a font size of at least 18 px at 640×360;
- declares an in-interval frame on the global 4 fps grid; and
- is independently checked in encoded media after downsampling to 320×180.

Only tier 3 may mark `observability.exception = "tier_3_flash"`. Such a flash is
explicitly outside the 4 fps guarantee and lasts 125–250 ms. It remains subject
to full-resolution media-presence validation.

## Anti-gaming rules

- All item matching is one-to-one; duplicate predictions cannot share truth.
- Malformed family items are omitted from matching, diagnosed, and counted as
  false positives. A malformed root or QA object zeros the report.
- False positives are capped only after matches are known:
  `effective_predictions = matches + min(unmatched, family_cap)`. A family with
  zero matches still scores zero. The cap prevents a single flooded family from
  asymptotically erasing otherwise correct work. The launch audit disproves the
  original claim that this prevents rewarding spam: capped false positives let
  dense public-vocabulary predictions retain reward despite many false claims.
- Repeated semantic events within two seconds are deduplicated, while rapid edit
  and overlay anchors remain distinct.
- Ambiguous partial actor/object descriptions do not resolve. Internal IDs are
  accepted for oracle/validator use but are not exposed as miner evidence.
- Shot starts and ends are not themselves extra cuts; only internal boundaries
  score. Provided shot IDs must agree when present on both sides.
- The exact `scene.json` truth, seed, source provenance, manifest hash, and media
  validation checks are validator-side authority. Miners receive only metered
  perceptual tools.

## Benchmark identity

A corpus lock records media, labels and scoring dependencies. Frozen artifacts
are not included in the public repository. Use a matching verified lock when
reproducing a frozen evaluation; local quickstart scenes use an explicitly
unlocked diagnostic round. A lock establishes identity, not corpus validity.
