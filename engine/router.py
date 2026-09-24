"""
Cheap classifier: decides whether the Investor persona should run, by
checking whether the input claim contains a monetizable idea.

Build brief: docs/build-brief.md, step 4.
"""

from __future__ import annotations


def route(claim: str) -> dict:
    """Return {"is_money_idea": bool, "reason": str}."""
    raise NotImplementedError
