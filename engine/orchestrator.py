"""
Runs the council: Believer and Skeptic always, Investor when routed, then
Judge. This is the public entry point - both the CLI and other agents in
the business-agent system should call evaluate() directly.

Build brief: docs/build-brief.md, step 6.
"""

from __future__ import annotations


def evaluate(claim: str) -> dict:
    """Run one full council session; return the Judge's verdict plus each
    persona's response."""
    raise NotImplementedError
