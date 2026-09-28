"""One structured Codex CLI call billed to the subscription session, not an API key."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from witness.providers import RATES


class QuotaExhausted(RuntimeError):
    """The active Codex account hit its usage limit; switch account and retry."""


def run_json(prompt: str, schema: dict, *, model: str, effort: str, images: list[str] = (),
             timeout_s: float = 900) -> dict:
    """Return the schema-shaped answer, token usage and its API-equivalent price.

    Shell and sub-agents are disabled; the model sees only the prompt and images.
    """
    environment = dict(os.environ)
    for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "CODEX_API_KEY"):
        environment.pop(name, None)  # an API key would silently switch billing
    with tempfile.TemporaryDirectory(prefix="witness-codex-") as work:
        schema_path, output = Path(work) / "schema.json", Path(work) / "output.json"
        schema_path.write_text(json.dumps(schema))
        command = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                   "--model", model, "-c", f'model_reasoning_effort="{effort}"',
                   "-c", "features.multi_agent=false", "-c", "features.shell_tool=false",
                   "--cd", work, "--color", "never", "--json", "--output-schema", str(schema_path),
                   "--output-last-message", str(output), *[arg for image in images for arg in ("--image", image)], "-"]
        started = time.monotonic()
        result = subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=timeout_s,
                                env=environment)
        if any(marker in (result.stdout + result.stderr).casefold()
               for marker in ("usage_limit_reached", "usage limit", "rate_limit_exceeded")):
            raise QuotaExhausted(model)
        if result.returncode or not output.exists():
            raise RuntimeError("codex_call_failed: " + result.stderr[-300:])
        usage = {}
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "turn.completed":
                usage = event.get("usage") or {}
        answer = json.loads(output.read_text())
    rate_in, rate_cached, rate_out = RATES[model]
    cached = usage.get("cached_input_tokens", 0)
    equivalent = ((usage.get("input_tokens", 0) - cached) * rate_in + cached * rate_cached
                  + usage.get("output_tokens", 0) * rate_out) / 1e6
    return {"output": answer, "usage": usage, "elapsed_s": time.monotonic() - started,
            "cost_usd": 0., "api_equivalent_usd": equivalent, "cost_basis": "codex_subscription_quota"}
