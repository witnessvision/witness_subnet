# Running a local temporal check

The versioned generator `witness.temporal` produces scene schema `2.0`, scored by
`2.0.0`. The task requires reconstructing observed interaction histories. It keeps
the inclusive quality threshold `0.4` and observation-efficiency accounting.
The [benchmark contract](temporal-benchmark.md) defines its acceptance criteria.

Use the existing Python environment and a fresh output directory:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from witness.temporal import generate
generate(101, 1, Path("data/scenes/temporal-example"))
PY
.venv/bin/python -m witness.validate data/scenes/temporal-example
.venv/bin/python -m witness.subnet.validator \
  --dry-run --once --allow-unlocked --score-version 2.0.0 \
  --scene data/scenes/temporal-example --scenes 1 --transcript-source none \
  --no-set-weights --round-root rounds/temporal-example
```

This uses the empty base miner and an in-memory chain. It verifies local plumbing;
its empty response is not evidence of competitive miner quality. Supplying the
temporal scene explicitly avoids the older fixtures selected by bare `--dry-run`.
Temporal scoring rejects earlier scene schemas and label-derived transcripts.

For a real implementation, keep inference in a separate package and use only the
public task and metered tools. Do not pass generated reference labels to the miner.
Scene seeds in an example are for reproducibility; validator-generated temporal
tasks use fresh private 256-bit seeds. Production selection and deployment are
separate from this local diagnostic check.
