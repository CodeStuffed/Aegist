"""
Agent wrapper: loads a persona's system prompt from config/personas/,
calls a backend (local or cloud), and parses the persona's structured
JSON reply.

Build brief: docs/build-brief.md, step 5.
"""

from __future__ import annotations


class Agent:
    def __init__(self, persona_name: str, backend):
        self.persona_name = persona_name
        self.backend = backend
        raise NotImplementedError

    def respond(self, claim: str, context: str = "") -> dict:
        """Return the persona's parsed JSON reply."""
        raise NotImplementedError
