"""
Runs the council twice and compares verdicts; flags disagreement as low
confidence instead of silently picking one result.

The re-run keeps dropout switched on while scoring (Monte Carlo dropout):
each pass samples a slightly different network. If the verdict survives
that, it's a real signal; if it flips, it was noise.

Build brief: docs/build-brief.md, step 7.
"""

from __future__ import annotations

LOW_NOTE = "Low confidence: the council didn't agree with itself on a repeat run"


def compare_verdicts(first: dict, second: dict) -> list[str]:
    """Material disagreements between two Judge verdicts (empty if none)."""
    reasons = []
    if first.get("stance") != second.get("stance"):
        reasons.append(f"verdict flipped: {first.get('stance')} -> {second.get('stance')}")
    if first.get("confidence") != second.get("confidence"):
        reasons.append(f"confidence changed: {first.get('confidence')} -> {second.get('confidence')}")
    return reasons


def reconcile(result: dict, rerun_judge: dict) -> dict:
    """Fold a repeat run's Judge into `result`. On disagreement, overall
    confidence drops to Low and both verdicts are kept side by side."""
    reasons = compare_verdicts(result["judge"], rerun_judge)
    result = dict(result, consistency={"agreed": not reasons, "reasons": reasons,
                                       "rerun_judge": rerun_judge})
    if reasons:
        result["overall_confidence"] = "Low"
        result["confidence_note"] = LOW_NOTE
    return result


def evaluate_with_confidence_check(claim: str, **kwargs) -> dict:
    from engine.orchestrator import evaluate
    return evaluate(claim, recheck=True, **kwargs)
