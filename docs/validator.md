# Run a Witness validator

Witness validates on **Bittensor Finney, subnet 20**. The validator needs **CPU
only**: no GPU, CUDA, OpenAI account, API key or model download. It generates
video/audio locally with FFmpeg and eSpeak NG, serves metered observations to
miners, and scores their JSON responses with **Witness Scorer v1.0.0**.
Miners provide their own inference. The validator does not run a miner.

You need a Linux server with Python 3.11+, a registered SN20 hotkey with a validator
permit, and a public TCP port for observations. The operator deployment uses
6 vCPU / 12 GB RAM; load depends on the number of responding miners. Keep the coldkey
private key off the validator server; use your established wallet access method.

## 1. Install

On Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv ffmpeg espeak-ng
git clone https://github.com/witnessvision/witness_subnet.git
cd witness_subnet
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Already installed? Stop your validator, run `git pull --ff-only`, then reinstall
with `.venv/bin/python -m pip install .`. Preserve existing round history. If an
upgrade changes the scoring identity, use a fresh `--round-root`; do not edit
an old EMA identity to make the check pass.

Aggregation 1.1.0 changes the history format. When upgrading from the original
single-round EMA, reconcile the old `weight-submission.json`, keep the old directory,
and use a fresh `--round-root rounds/mainnet-v1.1`. Do not relabel the old EMA file.

## 2. Check one round without changing weights

Replace `MY_WALLET`, `MY_HOTKEY` and `PUBLIC_IP`. Open inbound TCP 8765 in your
provider/server firewall. The URL must be reachable from miners; it is not a
Bittensor axon. Private labels and round files are not served through this API.

```bash
.venv/bin/witness-validator --mainnet \
  --wallet MY_WALLET --hotkey MY_HOTKEY \
  --tool-public-url http://PUBLIC_IP:8765 \
  --no-set-weights --once
```

An existing wallet is read from `~/.bittensor/wallets`. Add `--wallet-path PATH`
if needed, and `--network RPC_URL` to use your own Finney RPC. No `.env` file is
required. Environment-based deployments can use `WITNESS_WALLET`,
`WITNESS_HOTKEY`, `WITNESS_WALLET_PATH`, `WITNESS_NETWORK` and
`WITNESS_TOOL_PUBLIC_URL`; export them through your existing service manager.

Inspect `rounds/mainnet-v1/round_*/round.json`: the scorer should be `1.0.0`,
`weight_policy.name` should be `winner-takes-all`, and submission status must
be `disabled`. Miner responses, reward, winner and calculated weights are recorded
there. Offline or incompatible miners can return zero; only eligible responses
can win. This command makes real miner requests but submits no weights.

## 3. Start validation

Stop any previous validator or burn-only process for this hotkey, including its
service/timer. Remove `WITNESS_BURN_ONLY` and any `WITNESS_SET_WEIGHTS=0` setting.
Run the same command without the two test flags:

```bash
.venv/bin/witness-validator --mainnet \
  --wallet MY_WALLET --hotkey MY_HOTKEY \
  --tool-public-url http://PUBLIC_IP:8765
```

**This command submits weights.** Keep it running through your existing systemd,
PM2 or container supervisor. Use one writer per hotkey. The default persistent
state is `rounds/mainnet-v1`; keep the working directory stable across restarts.
Use `--round-root PATH` for another private writable state directory. The process
records an epoch heartbeat in `epoch-state.json`. If an epoch round fails, it waits
for the next epoch instead of repeatedly querying miners.

## What the mainnet preset does

- Generates five fresh private synthetic scenes per finalized subnet epoch.
- Evaluates every advertised, serving non-owner miner, up to four simultaneously.
  Each miner receives one task at a time, with a 180-second response deadline.
- Provides frames and audio. No transcript is supplied; miners may transcribe the
  audio themselves. The preset never exposes label-derived transcript hints.
- Uses scorer 1.0.0 and aggregation 1.1.0: mean reward over the last five rounds,
  followed by EMA alpha 0.1, with a 70% burn / 30% winner-takes-all allocation.
- Selects the highest EMA among miners with positive reward in the current round.
  Exact ties use lowest UID. The registered owner burn target is discovered from
  the chain and cannot compete. If every miner has zero current-round reward,
  **100% goes to burn and no winner is selected**, even if a miner has positive
  historical EMA. Weights are never assigned randomly. The same full-burn fallback
  applies whenever no miner qualifies.
- Verifies the Finney chain, validator permit, owner burn destination and weight
  constraints. It checks miner hotkeys again before submitting.

The round window and EMA alpha are configurable with `--score-window 5 --ema-alpha 0.1`
(or `WITNESS_SCORE_WINDOW` and `WITNESS_EMA_ALPHA`). Operators should agree on the
same values. Changing either requires a new round root, preserving previous evidence.

For each miner, let `m` be the mean of its last N round rewards. During startup,
average only the rounds observed so far; zero and missing-response rounds count.
EMA stays zero until the first positive `m`, which initializes EMA directly as a
quick start. Afterward, `EMA = 0.1 * m + 0.9 * previous_EMA`. For example, round
rewards `0, 0, 0.9` initialize EMA at `0.3`, rather than `0.03` or zero. This does
not discard the first positive reward. The N-round window bounds raw samples;
EMA still retains influence from older rounds. Smoothing reduces reactions to
individual rounds but increases response lag and cannot eliminate winner changes.

History survives restarts. A temporarily unadvertised miner receives zero samples;
returning does not erase its history. A replacement hotkey cannot inherit another
miner's history. `round.json` records the current reward, window mean, EMA,
observed window length and aggregation identity. Current-round eligibility and
the all-zero full-burn rule still apply after smoothing.

`--mainnet` fixes the remaining settings. Use the advanced CLI without `--mainnet`
for other configurations. Scorer version and
synthetic-corpus coverage are distinct: this preset runs the current generated
workload; it is not proof of performance on independent real-world videos.

## Check that weights became active

The round receipt distinguishes `disabled`, `rate_limited`, `rejected`,
`unknown`, and submission/inclusion/finalization evidence. With commit/reveal,
a finalized commitment is not yet an active weight vector. Verify revealed
weights in finalized chain storage and then the following epoch's incentives.
Yuma combines votes from all validators; one validator's 70/30 vote does not
promise a subnet-wide 70/30 payout. New validators can start voting before the
others update.

`weight-submission.json` preserves submission intent. An unresolved submission
blocks further submitting rounds; reconcile its transaction/chain state before
restarting, and preserve the original receipt. Do not delete the guard to retry
an action whose outcome is unknown. `--no-set-weights` remains available for
read-only diagnosis. A `rate_limited` round definitely sent no weight transaction;
normal epoch scheduling tries again with the next round.

For a deliberate temporary return to full burn, stop the mainnet writer, reconcile
pending commitments, then use the same wallet/network settings with
`--burn-only --netuid 20 --round-root rounds/burn` instead of `--mainnet`.
Verify the revealed burn vector; do not run both writers together.

[Scoring formulas, artifacts and advanced modes](scoring.md) ·
[Base miner setup](miner.md)
