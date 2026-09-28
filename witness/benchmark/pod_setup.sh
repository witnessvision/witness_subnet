#!/usr/bin/env bash
# Idempotent GPU setup for pod_runtime.py; everything lives under $WITNESS_GPU_WORKSPACE
# (default /workspace, the RunPod volume). Needs python3 with CUDA torch preinstalled
# (image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404), git and ffmpeg.
# Two exact environments from pod_env/*.txt (copied next to this script's workspace):
# "salmonn" (with the authors' repo at a fixed commit) and "omni" (Qwen2.5/3-Omni).
# An environment is rebuilt only when its lockfile changes.
set -euo pipefail
SALMONN_COMMIT=2d28e841c8378885d2f30982be801d0c44f7a5bb
WORKSPACE=${WITNESS_GPU_WORKSPACE:-/workspace}
mkdir -p "$WORKSPACE"/envs "$WORKSPACE"/models "$WORKSPACE"/jobs
cd "$WORKSPACE"
if ! command -v ffmpeg >/dev/null; then
  [ "$(id -u)" = 0 ] || { echo "ffmpeg is required" >&2; exit 1; }
  apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
fi
for env in salmonn omni; do
  # The salmonn environment also pins the authors' repository commit.
  ready=$(sha256sum "pod_env/$env.txt" | cut -d' ' -f1)
  [ "$env" = salmonn ] && ready="$ready:$SALMONN_COMMIT"
  if [ "$(cat "envs/$env/.ready" 2>/dev/null)" != "$ready" ]; then
    if [ "$env" = salmonn ]; then
      rm -rf video-SALMONN-2
      git clone -q https://github.com/bytedance/video-SALMONN-2 video-SALMONN-2
      git -C video-SALMONN-2 checkout -q "$SALMONN_COMMIT"
    fi
    rm -rf "envs/$env"
    python3 -m venv --system-site-packages "envs/$env"
    "envs/$env/bin/pip" install -q --no-deps -r "pod_env/$env.txt"
    echo "$ready" > "envs/$env/.ready"
  fi
done
echo ready
