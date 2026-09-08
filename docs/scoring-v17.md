# Scoring 1.7 candidate

This is a development candidate, selectable with benchmark `--score-version
1.7-candidate` or campaign `score_version: 1.7-candidate`. It does not change the
validator's default scorer or certify any dataset. Historical 1.5 and 1.6
implementations and reports remain unchanged.

Changes from 1.6:

- Every submitted event contributes to precision. Nearby repeated true actions
  can match separately; duplicate predictions cannot disappear from the count.
- Shot boundaries match by time, independently of private or miner-assigned IDs.
  Repeated boundaries, including duplicate initial shots, count as predictions.
- Visible text matches text and interval without requiring a private role label;
  intentional errors do not require an internal text ID. Both follow the public
  reconstruction contract; wrong text or error categories still receive no credit.
- Points must lie within the video and interval endpoints cannot exceed its
  duration. Boolean timestamps and negative sub-frame times are invalid.
- Dialogue uses maximum lexical-credit one-to-one matching among temporally
  eligible pairs. The old nearest-first order could assign overlapping lines to
  the wrong words. Unmatched and duplicated predictions still reduce precision.

Weights, tier thresholds and timing tolerances are unchanged. Regression checks do not establish independent model performance or resistance
to every reward shortcut.

Known remaining review areas include complete observable labels for native text,
audio and cuts. Do not promote on this scorer
alone. Final evaluation still requires a valid independent corpus, sealed code and
data identities, and fixed no-observation controls.
