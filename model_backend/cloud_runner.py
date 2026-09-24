"""
Cloud inference backend. Calls the Anthropic Messages API with the model
configured in config/settings.yaml (claude-opus-5-5 for the Judge,
claude-sonnet-5 for the panel, by default).

Build brief: docs/build-brief.md, step 2.
"""

from __future__ import annotations


def generate(system: str, prompt: str) -> dict:
    """Return {"text": str, "backend": "cloud", "model": str}."""
    raise NotImplementedError
