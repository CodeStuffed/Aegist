"""
Cheap classifier: decides whether the Investor persona should run, by
checking whether the input claim contains a monetizable idea.

Build brief: docs/build-brief.md, step 4.
"""

from __future__ import annotations

from engine.llm_json import call_json

ROUTER_SYSTEM = """\
You triage claims for an evaluation council. Decide whether the claim is a
monetizable idea: a product, business, pricing change, revenue strategy, or
anything whose success ultimately comes down to whether people will pay for
it. Factual claims, opinions, and technical or personal arguments are not
money ideas unless they hinge on making money.
Respond as JSON: {"is_money_idea": true|false, "reason": <one line>,
"topic": <the claim's subject in 2-5 lowercase words>}"""


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def route(claim: str, backend=None) -> dict:
    """Return {"is_money_idea": bool, "reason": str, "topic": str}.

    `topic` is extra: a short subject label the research loop counts to find
    recurring subjects across sessions.
    """
    if backend is None:
        from model_backend import select_backend
        backend = select_backend()
    parsed, _ = call_json(backend, ROUTER_SYSTEM, f"CLAIM:\n{claim.strip()}", role="router")
    if parsed is None:
        return {"is_money_idea": False,
                "reason": "Router reply wasn't valid JSON; skipped the Investor.",
                "topic": ""}
    return {
        "is_money_idea": _as_bool(parsed.get("is_money_idea")),
        "reason": str(parsed.get("reason", "")).strip(),
        "topic": str(parsed.get("topic", "")).strip().lower(),
    }
