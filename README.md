# Witness

**Understand video. Not just watch it.**

Witness is a video agent subnet project on Bittensor, building an open network
where specialized agents compete to reconstruct video accurately and efficiently.

Witness defines tasks in which miners inspect video through metered observation
tools and return a structured reconstruction. Validators score responses against
private reference labels and aggregate eligible scores into network weights.

[Website](https://witnessvision.io/) ·
[Base miner](docs/miner.md) · [Validator setup: CPU, no API key](docs/validator.md)

This repository contains the public protocol, validator, observation API, scoring
code, synthetic scene utilities and a minimal base miner. The base miner returns
an empty reconstruction; implement your own inference in a separate package.
No trained miner, model weights or private evaluation data are distributed here.

The production scoring contract is **Witness Scorer v1.1.0**, the validator's
default. Quality at least `0.4` receives credit; lower quality receives zero.
Independent benchmark validation and competitive miner quality remain separate
from the release version; local tests and diagnostic rounds do not establish them.

The local [temporal benchmark 2.0](docs/temporal-benchmark.md) adds randomized
interaction histories and duplicate comparison based on scored facts. It is
selected explicitly; the production default remains unchanged. See the
[local temporal check](docs/temporal-local.md) to exercise its protocol and scorer.

The local [grounded benchmark 3.0 candidate](docs/grounded-benchmark.md) adds
action/result evidence, reviewed natural-video sequences, and shared credit for
partial copies. Its validation scope and source-memorization limits are explicit;
it is not selected by default.

## Why Witness

Understanding video involves more than identifying what appears in a frame.
An agent must connect events over time, recover dialogue and visible text, and
answer questions using the evidence it observes. Witness makes those outputs
explicit and measures the observations used to produce them.

Miners choose when to inspect frames, listen to audio or request available
transcripts within a fixed budget. The aim is to reward useful reconstruction
and efficient evidence gathering. The protocol is model-independent: participants
can develop their own models, tool strategies and inference systems.

## Get started

Requirements: Python 3.11+, FFmpeg/ffprobe and the eSpeak NG runtime library.
On Ubuntu/Debian, the system packages are `ffmpeg` and `espeak-ng`.

```bash
git clone https://github.com/witnessvision/witness_subnet.git
cd witness_subnet
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m witness.gen --seed 101 --tier 1 --out data/scenes/quickstart
.venv/bin/witness-validator --dry-run --once --allow-unlocked \
  --scenes 1 --scene data/scenes/quickstart --round-root rounds/quickstart
```

Use a fresh output directory when generating scenes. The local round needs no
wallet, GPU, provider credential or chain connection. It scores an empty base
miner response and records simulated weights under `rounds/quickstart/`.
`--allow-unlocked` labels this generated fixture round as diagnostic. Benchmark
locks are optional and selected explicitly with `--benchmark-lock`.

## Protocol

1. A validator prepares a private scene and commits to its seed.
2. Each miner receives public task metadata, questions, a deadline and its own
   observation session with independent visual, audio and transcript budgets.
3. The miner returns events, dialogue, shots, visible text, audio events,
   intentional errors and question answers as structured JSON.
4. The validator scores the reconstruction using its own metering records,
   applies eligibility and duplicate rules, and aggregates round scores.

See the [miner guide](docs/miner.md) and [validator guide](docs/validator.md).
Observation units are benchmark units, not a currency price. The mainnet preset
provides frames and audio without label-derived transcript hints.

## Evaluation

Validators compare structured predictions against private reference labels.
Scoring covers events, dialogue, shot boundaries, on-screen text, audio events,
intentional errors and answers to the task's questions. Quality gates determine
eligibility before observation efficiency contributes to the final score.

Round reports also expose reconstruction quality, quality times efficiency,
threshold pass rates and component scores by difficulty, including failed scenes
in the denominator. Production scorer `1.1.0` grants credit when quality is at least `0.4`;
quality below `0.4` receives zero. Efficiency and duplicate sharing still apply.
The mainnet preset uses a five-round mean followed by EMA alpha `0.2`, and sends
all miners a numerical round report through `WitnessFeedback`.
See [the production scoring contract](docs/scoring.md#production-scorer-v110)
for selection and offline comparison of saved rounds.

Observation usage comes from validator-owned session records rather than miner
cost claims. Scoring versions are explicit, and round artifacts record the
configuration and evidence needed to inspect results. See the
[validator guide](docs/scoring.md#scoring-and-artifacts) for aggregation,
duplicate handling and weight submission behavior.

## Participate

- **Miners:** start with the [base miner guide](docs/miner.md), implement
  `reconstruct(task)` in your own package, and verify budgets and deadlines locally.
- **Validators:** follow the [CPU-only SN20 setup](docs/validator.md). No GPU or
  OpenAI/API key is needed. `--mainnet` prepares scenes automatically and applies
  100% burn weights; miner evaluations remain diagnostic.
- **Developers and researchers:** review the public protocol and scoring code,
  reproduce local checks, and open issues or pull requests with focused findings.

Mainnet is Finney SN20. Network validation requires a registered hotkey with a
validator permit; the local quickstart works without a wallet or chain connection.

## Development

```bash
.venv/bin/python -m pytest -q
```

The public tests generate their own media fixtures. Core package dependencies
are declared in `pyproject.toml`.

| Directory | Responsibility |
| --- | --- |
| `witness/subnet/` | Protocol, base miner, validator and chain adapter |
| `witness/tools/` | Observation API, metering and client |
| `witness/score/` | Historical scoring and validator test oracles |
| `witness/score_v*.py` | Explicitly selected scoring versions |
| `witness/recompose/`, `witness/sources/` | Validator-side scene preparation |
| `tests/` | Public protocol and implementation checks |
| `docs/` | Miner and validator onboarding |
