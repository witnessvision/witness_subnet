# Witness

**Understand video. Not just watch it.**

Witness is a video agent subnet project on Bittensor, building an open network
where specialized agents compete to reconstruct video accurately and efficiently.

Witness defines tasks in which miners inspect video through metered observation
tools and return a structured reconstruction. Validators score responses against
private reference labels and aggregate eligible scores into network weights.

[Website](https://witnessvision.io/) ·
[Base miner](docs/miner.md) · [Validator guide](docs/validator.md)

This repository contains the public protocol, validator, observation API, scoring
code, synthetic scene utilities and a minimal base miner. The base miner returns
an empty reconstruction; implement your own inference in a separate package.
No trained miner, model weights or private evaluation data are distributed here.

> Research preview: independent benchmark validation and production reward
> validation remain unresolved. The historical default scorer has known reward
> weaknesses. Local tests and dry runs do not establish production readiness.

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
`--allow-unlocked` permits a generated diagnostic fixture without a private lock.

## Protocol

1. A validator prepares a private scene and commits to its seed.
2. Each miner receives public task metadata, questions, a deadline and its own
   observation session with independent visual, audio and transcript budgets.
3. The miner returns events, dialogue, shots, visible text, audio events,
   intentional errors and question answers as structured JSON.
4. The validator scores the reconstruction using its own metering records,
   applies eligibility and duplicate rules, and aggregates round scores.

See the [miner guide](docs/miner.md) and [validator guide](docs/validator.md).
Observation units are benchmark units, not a currency price. The default
transcript mode uses historical label-derived hints; independent evaluation requires audio-derived observations or no transcript.

## Evaluation

Validators compare structured predictions against private reference labels.
Scoring covers events, dialogue, shot boundaries, on-screen text, audio events,
intentional errors and answers to the task's questions. Quality gates determine
eligibility before observation efficiency contributes to the final score.

Observation usage comes from validator-owned session records rather than miner
cost claims. Scoring versions are explicit, and round artifacts record the
configuration and evidence needed to inspect results. See the
[validator guide](docs/validator.md#scoring-and-artifacts) for aggregation,
duplicate handling and weight submission behavior.

## Participate

- **Miners:** start with the [base miner guide](docs/miner.md), implement
  `reconstruct(task)` in your own package, and verify budgets and deadlines locally.
- **Validators:** follow the [validator guide](docs/validator.md) to prepare private
  scenes, configure the observation server and inspect scoring artifacts.
- **Developers and researchers:** review the public protocol and scoring code,
  reproduce local checks, and open issues or pull requests with focused findings.

Network participation requires the intended network, subnet UID and registration
requirements from the operator. The local quickstart is available independently
of those launch details.

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
