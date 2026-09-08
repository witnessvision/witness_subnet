# Scoring v1.6 candidate

This development scorer is implemented in `witness/score_v16.py`. Historical
v1.5 rules remain unchanged. Selection is explicit through `--score-version
1.6-candidate` in the benchmark or validator CLI.

Two rule changes address the demonstrated public-source attack:

1. Family precision uses every submitted unmatched prediction. There is no
   false-positive cap. Matching, timing tolerances, event deduplication, weights,
   tier quality gates and metered cost formula retain the v1.5 semantics.
2. QA requires exact normalized equality (Unicode normalization, case, punctuation,
   whitespace and English articles as before). A response containing several
   alternative answers no longer receives substring or token-F1 credit.

Reports identify `benchmark_version: 1.6-candidate`. The default remains 1.5.
Use a separate evaluation history for each scoring identity.

## Limitations

These changes address known false-positive and QA-credit weaknesses. They do
not establish resistance to all attacks or validate a corpus. Incomplete native
text annotations can penalize correct visible-text predictions. A release
requires complete in-scope labels, an observable task scope and
[independent evaluation](evaluation-v2.md).
