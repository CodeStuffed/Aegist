"""
Local inference backend. Calls Ollama's HTTP API (http://localhost:11434)
using the model tag configured for the current hardware tier.

Build brief: docs/build-brief.md, step 2.
"""

from __future__ import annotations


def generate(system: str, prompt: str) -> dict:
    """Return {"text": str, "backend": "local", "model": str}."""
    raise NotImplementedError
