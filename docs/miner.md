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

## Starter implementation

Keep your implementation in a separate package that depends on this subnet
package. The public `WitnessMiner` owns request admission, deadlines, error
responses and axon callbacks; your subclass supplies reconstruction only.

```python
from typing import Any

from witness.subnet.miner import WitnessMiner
from witness.subnet.protocol import WitnessTask


class MyMiner(WitnessMiner):
    async def reconstruct(self, task: WitnessTask) -> dict[str, Any]:
        # Start from the protocol's empty response, with no invented answers.
        response = await super().reconstruct(task)

        # 1. Read task.task_spec for duration, fps and QA IDs/questions.
        # 2. Plan observations within task.budget and task.deadline_s.
        # 3. Query tools using task.tool_base_url and task.session_id.
        # 4. Run your own inference and fill the response from observed evidence.
        # 5. Return the reconstruction before the deadline.
        return response
```

This is a working mock: it makes no tool or model calls and returns empty
predictions. Replace the comments with your own strategy. Do not override
`forward()` just to add inference; keeping the inherited method preserves the
base miner's concurrency and timeout handling.

The observation client exposes `get_frame`, `get_frames`, `get_audio`,
`get_transcript` and `search_transcript`. Pass the assigned `session_id` when
constructing a client; miners cannot create sessions. The reference
`WitnessClient` is synchronous: use an asynchronous HTTP client for cancellable
requests, or isolate blocking calls with bounded I/O timeouts. Cancelling an
async wrapper does not stop work already running in a thread.

Populate only fields supported by your observations. The response contains
`events`, `dialogue`, `shots`, `on_screen_text`, `audio_events`,
`intentional_errors` and a `qa` object keyed by the task's question IDs.
Leave unsupported predictions empty. The validator computes quality and cost;
do not return a self-assigned score or substitute miner-side cost estimates.

For network serving, use the same setup and shutdown flow as the
[base entry point](../witness/subnet/miner.py), replacing
`WitnessMiner(chain=chain, ...)` with `MyMiner(chain=chain, ...)`. Attach the
instance's inherited `forward`, `blacklist` and `priority` callbacks to the axon.
The `witness-miner` CLI always starts the empty public template; installing a
subclass does not automatically select it.

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
