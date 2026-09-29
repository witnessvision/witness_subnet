# Witness

Witness is a video-understanding subnet for Bittensor SN20. Miners serve immutable
model weights to evaluator validators. Evaluators compare models on private,
random audiovisual clips; CPU followers reproduce the decision from finalized
on-chain scores. **70% of each validator's weight goes to burn and 30% to one king.**
Before the first king is decided, the allocation is **100% burn**.

[Website](https://witnessvision.io/) · [Miner guide](docs/miner.md) ·
[Validator guide](docs/validator.md) · [Scoring](docs/benchmark.md) ·
[Activation checklist](docs/launch.md)

## Mainnet v2 flow

1. **Submit:** prepare an allowlisted weights-only model and publish its `wm2`
   commitment. Serve it over pinned TLS with signed requests. The default miner
   admits only permitted validators holding at least **100,000 subnet alpha**.
   Witness charges no challenge fee; normal network registration/transaction
   costs still apply. One hotkey binds one model permanently.
2. **Queue:** each evaluator keeps durable round-robin coldkey order and FIFO
   within a coldkey. `A1,A2,A3,B1,C1` becomes `A1,B1,C1,A2,A3`. Infrastructure
   failures retain the same submission for retry. A terminal evaluation consumes
   that hotkey for that evaluator.
   The candidate panel is frozen at the window's opening finalized block,
   before drawing videos. Only immutable submissions committed strictly before
   that block can run. New submissions remain queued for the next window;
   retries and compression recovery cannot add or replace a model in the panel.
   A failed miner endpoint is removed from the local queue without consuming
   its hotkey. After repairing it, publish a fresh commitment for the same
   immutable model to rejoin; admission still waits for a later window opening.
   Validator, judge and GPU failures retain their normal retry behavior.
3. **Evaluate:** every two actual subnet epochs, each evaluator privately samples
   **10 videos from a public catalogue of 1,000** (from window 12; earlier windows retain 5), then **2 random 10–30 s clips per
   video**. Luna supplies two references per clip. The king and all challengers
   admitted to that window share its batch. The king runs once per window. From window 13, paired attempts allow up to
   30 minutes within the same two-epoch window; the clip cap remains 60 seconds.
   Completed model answers survive interruption and scoring retries. A bounded
   background download can prepare one other admitted challenger.
   From window 9, sources missing audio/video, too short for the required clips,
   or with invalid clip bounds/duration are excluded before model evaluation.
   An independently seeded deterministic reserve supplies replacements in the
   original draw positions. Rejections persist by catalogue source identity;
   retries reuse valid clips and references. Network, labeling and evaluator
   failures retry the same source. The final window batch is immutable.
   The ten-video transition preserves windows 0–11 and their king/uses.
   An existing ledger upgrades only if affected windows have no published
   results or closed decisions; other policy changes require explicit migration.
4. **Publish:** publish compact per-model quality, reward and paired king scores
   through chain commitments before deciding weights. Followers replay every
   finalized block; the latest commitment alone is insufficient.
5. **Crown:** at window close, use stake-weighted paired reward improvement.
   A challenger must exceed the king by more than `0.02`, with aggregate quality
   above `0.05`. The bootstrap compares the first two eligible submissions on a
   common evaluator panel. With no king, weights go to the subnet-owner burn UID.
6. **Display:** the dashboard reads validator projections and closed-window
   reports. It shows the queue, consumed/reserved hotkeys, responses, clip/video/
   total metrics and decision/submission/applied weights. It is never a consensus
   source; private labels and sampling secrets are not published.

## Install and test

Use Python 3.11+ and a repository-local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e '.[evaluator,dev]'  # evaluation and tests
.venv/bin/python -m pytest -q
```

Evaluators also need ffmpeg, a GPU and labeling/judging API credentials.
A single GPU machine can run the validator and evaluation together;
no separate CPU server or GPU service is required. API spending limits are optional
operator configuration, independent of consensus. The follower remains the default. Chain writes require explicit
flags. See the activation checklist before starting paid evaluation or weights.

The public repository contains protocol, scoring, validator runtime, tests, an
empty miner template and read-only validator status/evidence endpoints.
Website applications are installed separately. Credentials,
non-public evaluation references and runtime outputs stay outside this repository.

## Code map

| Module | Responsibility |
| --- | --- |
| `protocol.py`, `ledger.py`, `chain.py`, `commitments.py` | Compact records, finalized replay and deterministic consensus |
| `submission.py`, `p2p.py`, `miner.py` | Immutable manifests, authenticated TLS and empty miner template |
| `triggers.py`, `validator.py`, `evaluator.py` | Fair queue, CPU chain loop, GPU worker and durable publication |
| `pool.py`, `data/catalogue-v2.json`, `annotate.py`, `media.py` | Private per-window sampling and references |
| `reward.py`, `scoring.py`, `judge.py`, `adjudication.py` | Versioned quality and time reward |
| `gpu.py`, `watchdog.py`, `runner.py`, `pod_runtime.py`, `pod_env/` | GPU lifecycle, independent stop watchdog and pinned inference |
| `status.py` | Read-only validator status and finalized evidence export |
| `round.py`, `duel.py`, `simulation.py`, `cli.py` | Offline benchmark/rehearsal utilities; no mainnet coronation authority |

Paths in this table are relative to `witness/benchmark/`. Synthetic tests establish
software behavior, not GPU compatibility, model capability or finalized live
weight acceptance. Activation requires the separate checks in the launch guide.
