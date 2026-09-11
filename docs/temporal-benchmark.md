# Temporal benchmark 2.0 (in development)

The previous procedural benchmark repeats an action script for each difficulty.
Its random appearance does not prevent reconstructing that script from a few
images. The new version must measure observations of changing state over time.
It is a controlled video-reconstruction task, not certification of unrestricted
natural-video understanding. Historical scorers and scene generators retain
their original meaning.

## Required behavior

- Generate fresh stories with independently randomized participants, object
  assignments, interaction order, destinations and timing. A fresh seed must
  change the events and their relationships, not merely appearance.
- Use observable entity descriptions. Questions must not disclose outcomes,
  future attributes, event timestamps, counts or their own answers.
- Ask about complete interaction histories. Matched counterfactual videos must
  share public task metadata and initial/final states while requiring different
  answers because their intervening histories differ.
- Keep event, dialogue, shot, text, sound, error and QA reconstruction fields.
  Absent families must not give free quality. Temporal reconstruction is required
  in addition to the ordinary weighted reconstruction quality.
- Retain the inclusive quality threshold `0.4`, metered observation efficiency,
  five-round mean with EMA `0.2` and `70%` burn. Use a fresh scoring identity/history.
- Duplicate comparison must operate on scored meaning: equivalent entity names,
  list order, ignored fields and timestamp perturbations that preserve a match
  must not create independent credit.
- Never send seeds, private labels or generator state in observation responses.
  The base miner remains empty and independent of a model implementation.

## Acceptance evidence

Before miner tuning, freeze the task contract, scorer and control strategies.
Use separate development, calibration and freshly generated final cases; do not
reuse exposed historical holdouts. Record all unsuccessful inference attempts.

1. Check decoded media against labels, including event visibility, temporal
   separation, spoken content, visible text and counterfactual changes.
2. The original template attack, empty and question-only outputs must receive
   zero reward on final cases. Test a strong static-image control and an
   order-blind control as separate information restrictions. A known rewarded
   shortcut is a failure, not a reason to tune the scoring threshold afterward.
3. An oracle verifies scoring arithmetic only. Actual vision inference must
   independently establish task solvability through metered observations.
4. Compare the original private miner and a frozen improved candidate on the same
   reserved cases. Target mean quality `>=0.8`, each tier `>=0.7`, mean reward `>=0.6`
   and at least `90%` positive-reward tasks, with failures in the denominator. These
   targets do not replace the shortcut and observability checks.
5. Verify signed task transport, round aggregation, feedback and code identity.
   Chain activation is separate from passing a local evaluation.

No acceptance result is claimed by this specification. Correctness tests, attack
controls, rendered evidence and model evaluation must supply that evidence.
