# Witness

**Understand video. Not just watch it.**

Witness is a video agent subnet on **Bittensor Finney SN20**. Miners receive video
clips and return structured events; validators compare their responses with
private human annotations and rank eligible miners.

[Website](https://witnessvision.io/) · [Validator setup](docs/validator.md) ·
[MP4 protocol and miner integration](docs/production-v5.2.md)

## Production MP4 5.2

- Five shared random MP4 clips per round, 60–120 seconds each, sent to every
  registered endpoint with an announced IP and port.
- Hotkey-signed requests and responses, complete-body hashes, random task IDs
  and stripped metadata. Original identifiers and annotations stay private.
- Semantic evaluation using `gpt-5.6-luna` at `low` effort. SayGM is the default
  provider; the OpenAI adapter is also available. Only the selected API key is
  required.
- Annotation F1 with latency reward:
  `score = F1 * (0.7 + 0.3 * max(0, 1 - seconds/180))`.
  Complete rounds update hotkey EMA with alpha 0.2. Highest eligible EMA wins;
  lower UID breaks a tie.
- A persistent scheduler and separate weight writer. After activation, complete
  comparisons request 70% burn / 30% winner. Incomplete comparisons or no valid
  winner request full burn. Revealed chain state must be checked independently.
- Fixed UTC daily budgets with persistent reservations: $9 for a deployment's
  validator and $1 for its companion miner, including calibration/reevaluation.

Transport 5.2 uses event schema 5.0, annotation scorer 5.0.0 and latency reward
5.1.0; these are distinct versioned contracts. The wire limits are 128 MiB per MP4,
2 MiB per response and a 180-second external deadline. See the
[production contract](docs/production-v5.2.md) for signing, cancellation, provider
identity, ranking and activation requirements.

## Participate

**Validators:** follow [the MP4 deployment guide](docs/validator.md). You need
Python 3.11+, FFmpeg, a permitted hotkey, private evaluation assets and one funded
credential: `GM_API_KEY` for SayGM or `OPENAI_API_KEY` for OpenAI. No local GPU or
inference model is required. The guide includes evaluator/writer JSON examples,
state handling and chain verification. The public package does not distribute
private catalogs or calibration evidence.

**Miners:** use the empty, model-independent serving adapter described in
[MP4 integration](docs/production-v5.2.md#wire-contract), implementing your own
inference in a separate package. The [older base miner guide](docs/miner.md)
describes the legacy metered-observation protocol.

This repository contains the public protocol, validators, scoring, source
preparation, observation API, empty base miners, synthetic tests and documentation.
Competitive miner implementations, model weights, indexes, private data and
operational evidence belong outside it. Benchmark concordance on indexed content
does not establish general video understanding.

<a id="get-started"></a>

## Legacy protocols and local quickstart

The `witness-validator` console command still selects the earlier synthetic
validator; `--mainnet` uses scorer 1.1.0 and 100% burn. It does **not** launch
MP4 5.2. See [legacy validator setup](docs/validator-legacy.md).

The following local synthetic check needs no API key, wallet or chain connection.
Install FFmpeg and eSpeak NG (`ffmpeg` and `espeak-ng` on Ubuntu/Debian), then:

```bash
git clone https://github.com/witnessvision/witness_subnet.git
cd witness_subnet
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m witness.gen --seed 101 --tier 1 --out data/scenes/quickstart
.venv/bin/witness-validator --dry-run --once --allow-unlocked \
  --scenes 1 --scene data/scenes/quickstart --round-root rounds/quickstart
```

Use fresh output directories. This diagnostic scores an empty base miner and
records simulated weights; it is not an MP4 production acceptance test.

Earlier contracts remain available:
[scorer 1.1.0](docs/scoring.md#production-scorer-v110),
[temporal 2.0](docs/temporal-benchmark.md),
[grounded 3.0](docs/grounded-benchmark.md),
[structured-events diagnostics](docs/structured-events-v5.md) and
[experimental MP4 5.1](docs/mp4-transport.md).

## Development

```bash
.venv/bin/python -m pytest -q
```

Dependencies are declared in `pyproject.toml`. Public tests use synthetic fixtures
and have no dependency on a competitive miner or private evaluation data.

| Directory | Responsibility |
| --- | --- |
| `witness/subnet/` | Chain adapters, validators, base miner and weight writers |
| `witness/mp4_v5_2.py` | Signed production transport |
| `witness/judge.py`, `witness/providers.py`, `witness/budget.py` | Evaluator, providers and daily accounting |
| `witness/score_v*.py` | Versioned scoring |
| `witness/sources/` | Validator-side source preparation |
| `witness/tools/` | Media utilities and legacy observation API |
| `tests/`, `docs/` | Public tests and onboarding |
