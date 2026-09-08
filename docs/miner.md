# Base miner onboarding

The public repository provides a minimal miner template. It serves the Witness
protocol and returns an empty reconstruction. It contains no trained model,
observation strategy, provider integration or competitive inference code.

## Install and verify

Follow the [local quickstart](../README.md#get-started). The local round runs the base miner
without a wallet or chain connection. Empty responses are expected; this checks
integration and does not demonstrate mining performance.

## Implement reconstruction

Subclass `WitnessMiner` from `witness.subnet.miner` and implement its asynchronous
`reconstruct(task)` method in your own package. It must return a dictionary
following the [reconstruction schema](../witness/contract.py).

The task supplies public metadata, questions, a deadline, independent observation
budgets, `tool_base_url` and `session_id`. Use that assigned session with the
[observation API client](../witness/tools/client.py). Do not create another session or access
validator reference labels. The protocol does not prescribe a model or provider.

Use nonblocking inference and bounded external requests. Propagate cancellation;
do not suppress task deadlines. The template limits admission with `concurrency`,
returns `busy` when full, and returns empty responses for errors or timeouts.
Validator-owned metering determines observation cost.

## Network configuration

Obtain the intended network, subnet UID and registration requirements from the
subnet operator. Configure a registered wallet hotkey and reachable axon address.
CLI defaults are not confirmation of a public Witness deployment.

After setting the corresponding environment variables locally:

```bash
.venv/bin/witness-miner \
  --netuid "$WITNESS_NETUID" --network "$WITNESS_NETWORK" \
  --wallet "$WITNESS_WALLET" --hotkey "$WITNESS_HOTKEY" \
  --port 8091 --concurrency 1
```

This command starts the empty base miner. To serve your own implementation,
instantiate your subclass and attach its `forward`, `blacklist` and `priority`
callbacks through the chain adapter, as shown in the base entry point.

Use `--external-ip` and `--external-port` when the advertised address differs
from the local endpoint. `--max-deadline-s` bounds the response deadline.
`WITNESS_WALLET_PATH` selects a non-default wallet root.

```bash
.venv/bin/witness-miner --help
```

Unregistered callers are rejected by default. `--allow-unregistered` is for
isolated development only. Keep wallet material and provider credentials outside
source control. Verify schema validity, budget use, deadlines and cancellation
before serving an implemented miner. Production reward validation remains open.
