"""
Runs the council: Believer and Skeptic always, Investor when routed, then
Judge. This is the public entry point - both the CLI and other agents in
the business-agent system should call evaluate() directly.

Build brief: docs/build-brief.md, step 6.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from config import load_settings
from engine.agents import CONFIDENCE_LEVELS, Agent
from engine.router import route

log = logging.getLogger("council")

PANEL_FIELDS = ("position", "key_points", "confidence", "what_would_change_my_mind")

# Appended to the Judge's per-call prompt (its system prompt stays exactly as
# written in config/personas/judge.md). "stance" lets the uncertainty check
# detect a flipped verdict; "relied_on" lets code enforce the Judge's
# confidence ceiling instead of hoping the model applies it.
JUDGE_EXTRAS = """\
In addition to the fields in your instructions, include these two:
"stance": "yes" | "no" | "mixed" - does the claim hold up overall?
"relied_on": [the personas whose cases your verdict rests on most, from: {names}]"""


def format_case(result: dict) -> str:
    title = f"{result['persona'].upper()}'S CASE"
    if "error" in result:
        return f"{title} (reply wasn't valid JSON; raw text follows):\n{result.get('raw', '')}"
    body = {k: result.get(k) for k in PANEL_FIELDS}
    return f"{title}:\n{json.dumps(body, indent=2, ensure_ascii=False)}"


def _rank(confidence: str | None) -> int:
    # Missing or unparseable confidence counts as the weakest level.
    return CONFIDENCE_LEVELS.index(confidence) if confidence in CONFIDENCE_LEVELS else 0


def apply_confidence_cap(judge: dict, panel: dict) -> dict:
    """Enforce: the Judge's confidence never exceeds the lowest confidence
    among the inputs it relied on most. If the Judge didn't say which inputs
    those were, every panel input counts."""
    relied = judge.get("relied_on")
    names = [str(n).strip().lower() for n in relied] if isinstance(relied, list) else []
    basis = [n for n in names if n in panel] or list(panel)
    ceiling = min(_rank(panel[n].get("confidence")) for n in basis)
    judge = dict(judge, relied_on=basis)
    stated = judge.get("confidence")
    capped = CONFIDENCE_LEVELS[min(_rank(stated), ceiling)]
    if stated is not None and capped != stated:
        judge["confidence_before_cap"] = stated
    judge["confidence"] = capped
    return judge


def run_council(claim: str, *, backend, routing: dict, context: str = "",
                temperature: float | None = None) -> dict:
    """One pass: the panel (in parallel), then the Judge. Returns {"panel", "judge"}."""
    names = ["believer", "skeptic"] + (["investor"] if routing["is_money_idea"] else [])
    log.info("Panel deliberating: %s", ", ".join(n.capitalize() for n in names))
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {n: pool.submit(Agent(n, backend).respond, claim, context, temperature=temperature)
                   for n in names}
        panel = {n: f.result() for n, f in futures.items()}

    log.info("Judge deliberating")
    judge_context = "\n\n".join(
        [context] + [format_case(panel[n]) for n in names] + [JUDGE_EXTRAS.format(names=", ".join(names))]
    )
    judge = Agent("judge", backend).respond(claim, judge_context, temperature=temperature)
    return {"panel": panel, "judge": apply_confidence_cap(judge, panel)}


def evaluate(claim: str, *, backend=None) -> dict:
    """Run one full council session; return the Judge's verdict plus each
    persona's response.

    `backend` may be a runner module, "auto" / "cloud" / "local", or None
    for settings.yaml's choice.
    """
    settings = load_settings()
    if backend is None or isinstance(backend, str):
        from model_backend import select_backend
        backend = select_backend(backend, settings)

    log.info("Routing claim")
    routing = route(claim, backend)
    council = run_council(claim, backend=backend, routing=routing)
    return {
        "claim": claim,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend.NAME,
        "models": {"judge": backend.model_for("judge"), "panel": backend.model_for("panel")},
        "routing": routing,
        **council,
        "overall_confidence": council["judge"]["confidence"],
    }
