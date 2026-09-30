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

The model server supports HTTP/1.1 keep-alive and up to two concurrent requests
from each validator, with a 4 MiB range limit per request. Keep the endpoint
available across evaluation windows so interrupted downloads can resume. Older
model servers using HTTP/1.0 remain compatible and close each response instead
of reusing connections between ranges.

Evaluators allow at most 15 minutes of active acquisition per model across
retries, in turns of at most 15 minutes. Waiting in the queue is excluded. Serve
the full package from a stable, sufficiently fast endpoint: reaching the total
limit removes the submission from that evaluator's queue, without producing a
model score or consuming the hotkey. Submit the same bound commitment again after
removal to request another acquisition quota in the next eligible window. A slow drip of HTTP headers does not extend
the per-request 30-second absolute deadline.

Validators require lossless Zstandard compression for every requested file range,
including tiny and incompressible ranges (`Accept-Encoding: zstd, identity;q=0`). The original model files, manifest,
hashes and commitment stay unchanged; do not upload a ZIP or replace the weights
with an archive. Compression happens per range in memory, so no extra full-model
copy is needed. Old validators still receive raw bytes. Upgrade the package
dependencies and restart the model server. Servers that return raw bytes, an
unsupported encoding or invalid Zstandard frames are excluded from the operational
queue without a score, result commitment or consumed hotkey. A bounded check
retries after 25 finalized blocks (about five minutes); a compliant server returns
to the queue with its original model and hotkey binding. No new commitment is needed.

### Integrating compression into another miner server

The built-in `witness-benchmark-miner serve` command already negotiates compression.
Custom servers can reuse the same helper, without importing a wallet or running
model inference:

```python
from witness.benchmark.compression import MAX_CHUNK_BYTES, encode_model_chunk

# Authenticate btauth/1 and validate the committed file/range first.
# Read exactly the requested original range, at most MAX_CHUNK_BYTES (4 MiB).
wire, encoding = encode_model_chunk(raw_range, accept_encoding=accept_encoding_header)
headers = {
    "Content-Type": "application/octet-stream",
    "Content-Length": str(len(wire)),
    "Vary": "Accept-Encoding",
}
if encoding:
    headers["Content-Encoding"] = encoding  # "zstd"
# Send HTTP 200 with these headers and wire as the response body.
```

Install the project's declared dependencies (`zstandard==0.25.0` is included).
The helper respects `zstd;q=0` and sends identity to older clients. With
`identity;q=0`, it always returns a Zstandard frame, even if that frame is slightly
larger than the original range. For optional negotiation it skips unhelpful compression. Do not change range offsets,
manifest sizes, model files, hashes, TLS pinning or request authentication.
Custom clients can use `decode_model_chunk(wire, expected_size)` from the same
module for a `Content-Encoding: zstd` response; `expected_size` is the requested
original range length, not the compressed Content-Length. The validator still
verifies each complete file against its committed SHA-256 after decoding.

A hotkey signature prevents someone from claiming another validator's identity.
An admitted validator receives plaintext weights and can retain or redistribute
them; the stake gate cannot prevent that. Keep any unrelated private files and
credentials outside the served model directory. Never place wallet secrets in
an environment file or model package.

The public `respond` command remains an empty model-independent template:

```bash
.venv/bin/witness-benchmark-miner respond --mp4 example.mp4
```
