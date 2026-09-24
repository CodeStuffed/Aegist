"""
Runs the council twice and compares verdicts; flags disagreement as low
confidence instead of silently picking one result.

Build brief: docs/build-brief.md, step 7.
"""

from __future__ import annotations


def evaluate_with_confidence_check(claim: str) -> dict:
    raise NotImplementedError
