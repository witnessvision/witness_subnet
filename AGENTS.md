# Witness subnet repository

This repository is public. Include only protocol, validator, scoring, observation
API, base miner, public tests and onboarding documentation. Competitive miners,
training workflows, model weights, private datasets, credentials, research logs
and operational notes belong outside this repository.

Use Python 3.11+ and a local `.venv`; dependencies are declared in pyproject.toml.
Run `.venv/bin/python -m pytest -q` before publication. The base miner must remain
an empty, model-independent template. Do not add private inference code to tests
or examples. Preserve versioned protocol and scoring behavior unless explicitly
changing their contracts.

## Publication

Use repository-local author and committer
`techwhiz-semantha <179846318+techwhiz-semantha@users.noreply.github.com>`.
Authenticate SSH with `~/.ssh/witness` and `IdentitiesOnly=yes`; verify the account
is `techwhiz-semantha` before every push. The only public destination is
`git@github.com:witnessvision/witness_subnet.git`.

Keep the pre-push guard. Run the local managed identity check immediately before
commits and pushes when available. Review an explicit source allowlist and secret
scan before publication. Never publish miner internals, private artifacts or old
history. Force pushes and history changes require explicit authorization.

The local `web/` and `.private/` directories are separate ignored repositories.
Do not stage their contents here.
