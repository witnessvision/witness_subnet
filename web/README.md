# Witness web

The public landing page, dashboard and read-only FastAPI backend are part of
`witnessvision/witness_subnet`. This directory is a Python package, not a separate
Git repository. It depends on the sibling subnet package within this checkout.

## Local installation and checks

From the subnet repository root, use Python 3.11+ and its local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[evaluator,dev]' -e 'web[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m uvicorn witness_web.app:app --host 127.0.0.1 --port 8097
```

Configure validator status endpoints with `WITNESS_VALIDATOR_SOURCES` as described
in [the validator guide](../docs/validator.md). The dashboard displays evidence;
it does not participate in consensus or imply that mainnet activation has passed.

The backend exposes only the public subnet projection and bundled public assets.
Tests use synthetic local fixtures and require no live validators or paid APIs.
Keep production host configuration and credentials outside this repository.
