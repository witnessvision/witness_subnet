# Local quickstart

This guide verifies one local Witness round without a wallet, chain connection,
GPU or provider API credential. The empty base miner tests integration,
not video-understanding quality.

## Install

Requirements: Python 3.11 or later, FFmpeg/ffprobe and the eSpeak NG runtime
library. On Ubuntu/Debian, the system packages are `ffmpeg` and `espeak-ng`.

```bash
git clone https://github.com/witnessvision/witness_subnet.git
cd witness_subnet
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Package dependencies are declared in `pyproject.toml`. The base miner requires no model backend or model weights.

## Generate and validate a scene

Use a new output directory; generation can overwrite files at an existing path.

```bash
.venv/bin/python -m witness.gen \
  --seed 101 --tier 1 --out data/scenes/quickstart
.venv/bin/python -m witness.validate data/scenes/quickstart
```

The output contains `video.mp4` and validator-side reference labels in
`scene.json`. Keep labels and raw evaluation artifacts outside miner access.
Publicly reproducible scenes are development fixtures, not sealed test data.

## Run one local round

```bash
.venv/bin/witness-validator \
  --dry-run --once --allow-unlocked \
  --scenes 1 --scene data/scenes/quickstart \
  --round-root rounds/quickstart
```

The local round creates a session for the empty base miner, scores its response
and simulates weight submission. Empty predictions earn no task reward; the
configured burn fallback can still receive the simulated weight vector. Inspect `rounds/quickstart/round_*/round.json`
for task results and submission status. `--allow-unlocked` permits this generated
fixture without a private benchmark lock; it is a diagnostic option.

## Run protocol checks

```bash
.venv/bin/python -m pytest -q tests/test_subnet.py
```

Run the complete public suite with `.venv/bin/python -m pytest -q`. Tests generate
required media in temporary directories; private datasets are not required.

## Continue

- [Miner onboarding](miner.md): implement reconstruction and serve tasks.
- [Validator onboarding](validator.md): configure sessions and inspect rounds.
- [Evaluation requirements](evaluation-v2.md): validate independent claims.
- [Scene contract](design.md): understand validator-side data.
