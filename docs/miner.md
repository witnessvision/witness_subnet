# Miner: one immutable model per hotkey

A miner runs a small CPU model server. Evaluators download its weights directly
over authenticated, pinned TLS and execute them with validator-owned loaders.
Hugging Face may be a private development source; it is not the v2 submission
transport. No miner inference service or GPU is needed to serve the package.

## Model package

| Architecture | `config.json` model type | Maximum package bytes |
| --- | --- | --- |
| `salmonn2-pro` | `qwen3_vl` | 24,000,000,000 |
| `qwen2.5-omni` | `qwen2_5_omni` | 24,000,000,000 |
| `qwen3-omni` | `qwen3_omni_moe` | 44,000,000,000 |

These are implemented loader formats, **not a claim that each has passed the
mainnet GPU qualification**. Consult the activation checklist for the release.
Submit full weights, config and tokenizer assets, not an adapter alone. Only
safetensors, JSON, Jinja and allowlisted tokenizer files are transported. Remote
code, symlinks, traversal and unlisted shards are rejected. The canonical
manifest binds file names, sizes and SHA-256 hashes. Byte-identical weight mirrors
are detected independently of repository, filename and certificate.

```bash
.venv/bin/witness-benchmark-miner prepare \
  --model-dir /srv/witness/model --arch qwen2.5-omni \
  --state /var/lib/witness-miner
```

This hashes the package and creates a private TLS key and certificate. It does
not open a wallet or publish anything. Preserve that state directory and model
bytes: the certificate fingerprint and manifest digest are immutable.

## Serve, then submit

The following operations require your registered hotkey and explicit authority
to publish. Keep the model server available while evaluations are pending.

```bash
.venv/bin/witness-benchmark-miner serve \
  --model-dir /srv/witness/model --arch qwen2.5-omni \
  --state /var/lib/witness-miner --wallet-name NAME --wallet-hotkey HOTKEY \
  --port 8091 --external-ip PUBLIC_IP --min-stake-alpha 100000 --publish
```

`--publish` on `serve` advertises the endpoint on chain. Without it, the server
starts using an already advertised endpoint. The IP must be publicly routable.
From a second terminal, inspect the submission, then publish it:

```bash
.venv/bin/witness-benchmark-miner commit \
  --model-dir /srv/witness/model --arch qwen2.5-omni \
  --state /var/lib/witness-miner
# Add --wallet-name NAME --wallet-hotkey HOTKEY --publish only to submit on chain.
```

The wire record is `wm2|<base64url(manifest SHA-256 || certificate SHA-256)>`.
The first valid record after protocol activation permanently reserves that
hotkey. Later replacements are ignored, even after a coldkey change or restart.
The same hotkey can be independently evaluated by several validators, once each.
The evaluated model ID binds the manifest digest to the submitting hotkey.
Copying another miner's public commitment cannot reserve its identity or reuse
its authenticated download cache. Duplicate weights are rejected after verified
acquisition, before inference; a bare public digest is not proof of possession.

**There is no Witness challenge fee.** Registering hotkeys and writing chain
transactions remain subject to the network's own costs. One coldkey can register
several hotkeys, but its submissions alternate with other coldkeys in the queue.
Creating more hotkeys does not move that coldkey to the front.

## Access protection

The default `--min-stake-alpha 100000` requires **100,000 SN20 alpha** and a
validator permit. This is a download gate, not a consensus stake threshold.
Every request uses `btauth/1`, binds the receiving miner and URL, expires quickly,
and rejects replay. Chain state older than 120 seconds fails closed. Downloads
are bounded and resumable; the client checks the committed certificate before
sending authentication and verifies every complete file.

A hotkey signature prevents someone from claiming another validator's identity.
An admitted validator receives plaintext weights and can retain or redistribute
them; the stake gate cannot prevent that. Keep any unrelated private files and
credentials outside the served model directory. Never place wallet secrets in
an environment file or model package.

The public `respond` command remains an empty model-independent template:

```bash
.venv/bin/witness-benchmark-miner respond --mp4 example.mp4
```
