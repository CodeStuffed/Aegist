"""
One LLM call that should come back as a JSON object: parse it, and if it
doesn't parse, ask once more with an explicit nudge before giving up.
Shared by the router and every persona Agent.
"""

from __future__ import annotations

import json
import re

RETRY_NUDGE = (
    "\n\nYour previous reply was not valid JSON. Reply with valid JSON only: a single "
    "JSON object in exactly the format your instructions describe, with no prose "
    "before or after it and no code fences."
)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict | None:
    """Best-effort: the whole reply, a fenced block, or the first {...} object in it."""
    text = (text or "").strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def call_json(backend, system: str, prompt: str, *, role: str = "panel",
              temperature: float | None = None) -> tuple[dict | None, dict]:
    """Return (parsed object or None, {"backend", "model", "attempts", "raw"})."""
    out = backend.generate(system, prompt, role=role, temperature=temperature)
    parsed, attempts = extract_json(out["text"]), 1
    if parsed is None:
        out = backend.generate(system, prompt + RETRY_NUDGE, role=role, temperature=temperature)
        parsed, attempts = extract_json(out["text"]), 2
    return parsed, {"backend": out["backend"], "model": out["model"], "attempts": attempts,
                    "raw": out["text"]}
