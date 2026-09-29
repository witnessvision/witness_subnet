# Mainnet v2 sampling, scoring and consensus

## Evaluation window

A window spans **two `SubnetEpochIndex` increments**, starting from the agreed
activation epoch. Validators use the same release/policy hash, activation block
and epoch. Finalized chain history determines boundaries and submissions.
Candidates must be registered and committed strictly before the opening block;
new submissions enter the next window. The incumbent king is frozen at opening.

Each evaluator independently samples ten distinct videos from window 12 (five before that) from
[`catalogue-v2.json`](../witness/benchmark/data/catalogue-v2.json), containing
1,000 public-source metadata entries. A private persistent secret, evaluator
hotkey and window ID determine the draw and random clip intervals. Each video
contributes two 10–30 second clips with audio. Two Luna reference passes label
each clip from sampled frames and GPU-produced speech/sound evidence.

The batch remains identical across retries within a window. The next window
samples afresh; random sampling can naturally repeat a video. All challengers
and the king use the same batch within one evaluator/window. The king runs once,
and its hardware identity must match challengers'. Evaluators can use different
batches: each comparison is paired locally before stake aggregation.

Unfinished jobs defer to a fresh batch in the next window without reserving a
new challenge. Source, labeling, judge or GPU infrastructure failures do not
consume the hotkey. Invalid miner packages and completed invalid/timeout answers
do. Generic and empty controls must average at most `0.05` before publication.

Small batches increase throughput and also sampling variance. The `0.02` margin
is a protocol rule, not a statistical confidence claim. Machine references and
control passes do not establish human-level understanding or eliminate gaming.

## Clip, video and evaluation scores

Answers contain timestamped claims in visual, speech, on-screen text and sound
modalities. Quality follows the versioned precision/recall contract in
[`reward.py`](../witness/benchmark/reward.py) and
[`scoring.py`](../witness/benchmark/scoring.py). Across two references, a claim is
supported if either supports it and contradicted only if both contradict it;
recall averages the references.

```
time_score = clamp(1 - elapsed_seconds / min(60, 10 * clip_duration_seconds), 0, 1)
clip_reward = quality * (0.8 + 0.2 * time_score)
video_quality, video_reward = arithmetic means of that video's clips
eval_quality, eval_reward = arithmetic means of the video means
```

Loading and download are excluded from answer latency. Each clip has a **60 s** hard
supervisor deadline; a hung worker is killed and the next clip can proceed.
Missing/invalid model answers score zero. Provider or hardware failures defer
instead of quietly turning into a zero.

## Early stopping and execution budget

The king always completes all ten videos in one model job. Challenger inference
runs the first three videos in one job and, only if needed, the remaining seven in a
second job; it does not reload the model for every video. Grading follows the
original random draw order, never download-completion order. A deterministic best-possible-completion bound
may stop a challenger that cannot clear the paired margin even with perfect
remaining answers.

There is also **one** statistical look, after three complete videos. If `m` is
their largest video reward, `(4 + 6*m) / 10` is a one-sided upper confidence
bound of at least 90% for the fixed ten-video batch: the chance of missing the
largest five is `choose(5,3) / choose(10,3) = 10/120`, below 0.1. The earlier
five-video windows retain `(1 + 4*m) / 5`. Stop
only if the smaller of this and the deterministic bound cannot beat the king
by `0.02`, with upward-rounded bounds in the chain representation. There is no
extra statistical look after each clip or after four videos. No normality,
clip independence or zero-variance assumption is made.
The finite-population probability uses the
[hypergeometric sampling formula](https://itl.nist.gov/div898/software/dataplot/refman2/ch8/hypcdf.pdf).

This permits up to 10% false statistical cuts per evaluator/window, conditional
on fixed scores and uniform sampling order. It is not a posterior probability,
a generalization claim, or a subnet-wide 10% error guarantee; repeated
evaluators/windows increase aggregate exposure. The known public catalogue can
also be trained on. These are explicit limitations of this release.

Early results use flag `8`: quality and reward encode **upper bounds**, not
measured full-batch scores. Their stake stays in the comparison. A partial bound
can never crown a model. If the stake-weighted upper comparison leaves a possible
winner, the evaluator resumes the remaining videos, and a full result supersedes
its provisional bound. Unresolved bounds may block an ambiguous promotion.
Only a bound that rules out victory at window close consumes that evaluator's
attempt; an unresolved partial remains retryable. Bootstrap uses full results.

From window 13 the paired attempt has a 1,800-second budget starting before
shared preparation and downloads. It may cross an epoch boundary within the
same two-epoch window; window closure always cancels it. Earlier windows retain
the 900-second budget and epoch cancellation. Completed first answers are
atomically cached by model, clip/task, runtime and policy. Interrupted GPU jobs
retain fully flushed responses; a judge retry reuses the same measured answer
and latency rather than generating another. Missing interrupted tasks remain
pending, never implicit zero scores. Local media and GPU
processes poll cancellation and terminate owned children with a five-second kill
backstop. Model loading is capped at 120 seconds per worker and warmup at one clip
deadline. Label/judge requests use at most 30 seconds and the remaining attempt
budget; in-flight network operations have their own bounded unwind time. GPU
setup belongs to installation. Real timing and cancellation remain required
activation checks; this is not a claim that every cold model can finish in 15 minutes.
Completed video grades are checkpointed against the exact clip hashes within
the window, so a deferred attempt resumes without rerunning completed videos.

## Compact commitments and replay

`wr2|` encodes a 91-byte record in 126 ASCII bytes: window, miner UID, immutable
challenge SHA-256 (bound to miner hotkey and manifest), 96-bit policy prefix,
report SHA-256, four uint16 scores
(candidate quality/reward, paired king quality/reward), and result flags.
Scores round to the nearest `1/65535`; followers use exact rational arithmetic
on those integers. A separate baseline record binds the king's scores for the
same evaluator/window. Full reports are optional display/audit data: no HTTP
request or dashboard vote participates in consensus.

Every finalized block from the activation checkpoint is replayed contiguously.
The first block of the next window closes the previous window before accepting
its records. Late results cannot be moved to another window. The first terminal
record for an evaluator/miner hotkey is permanent; later records remain audit
history and cannot provide another vote or attempt. The SQLite ledger preserves
this across restarts. A follower needs historical state access, not just the
latest `CommitmentOf` value.

## Single winner

At closing, a permitted evaluator's weight is its positive total chain stake in
the closing finalized snapshot. There is no application quorum or extra stake
floor. Only matching-policy results with matching paired king baselines count.
A definitive rejection contributes zero and retains its evaluator stake in the
denominator. For each challenger, normalize over the stake of its qualifying evaluators:

```
paired_delta = sum(stake_i * (reward_candidate_i - reward_king_i)) / sum(stake_i)
```

A candidate must have stake-averaged quality above `0.05` and paired delta above
`0.02`. Choose the largest delta; break exact ties by submission block then
hotkey. A lone qualifying evaluator can determine the outcome. Follower stake
is not counted a second time as independent evaluation evidence.

Without an incumbent, the first two eligible coldkey-fair submissions form the
bootstrap pair. An incomplete pair rotates behind the next eligible pair at the
next window, so offline miners cannot block bootstrap forever. Only evaluators
with terminal results for both join the panel;
a rejected model can complete the pair with a zero score. A partially published
bootstrap pair consumes neither hotkey: both must finalize in the same window,
or the incomplete work can retry with a fresh batch. Choose the highest reward
above the quality floor. If a completed pair yields no king, retire it and try
the next pair. Without a registered winner, use the subnet-owner burn UID.

The selected king receives the entire normalized vector `[1.0]`. Weights are
never spread across the kings named by different validators. Decision, extrinsic
submission and observed applied weights are distinct states, especially with
chain commit/reveal enabled.
