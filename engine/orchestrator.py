"""
Runs the council: Believer and Skeptic always, Investor when routed, then
Judge. This is the public entry point - both the CLI and other agents in
the business-agent system should call evaluate() directly.

Build brief: docs/build-brief.md, step 6.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from config import load_settings
from engine import uncertainty
from engine.agents import Agent, build_prompt, confidence_from_signal, min_confidence
from engine.memory import knowledge_store, session_log
from engine.router import route
from model_backend.brain import load_brain

log = logging.getLogger("council")

VERDICTS = {
    "yes": "Yes - on what this model has read, the claim holds up.",
    "no": "No - on what this model has read, the claim doesn't hold up.",
    "mixed": "Undecided - the model finds the claim about as compatible with 'true' as with 'false'.",
}


def assess_familiarity(brain, claim: str, settings: dict) -> dict:
    """How far the claim is from anything the model has read. Unfamiliar
    claims can't get a confident verdict, however strong the signals look."""
    cfg = settings["council"]
    typical = brain.stats.get("val_loss") or float("nan")
    claim_loss = brain.text_nll(claim)
    ratio = claim_loss / typical if typical == typical else float("inf")
    cap = "High"
    if ratio > cfg["familiarity"]["low_above"]:
        cap = "Low"
    elif ratio > cfg["familiarity"]["medium_above"]:
        cap = "Medium"
    notes = []
    if cap != "High":
        notes.append(f"the claim is unlike most of what the model has read "
                     f"(loss {claim_loss:.2f} vs. a typical {typical:.2f})")
    if brain.stats.get("tokens_seen", 0) < cfg["min_tokens_trained"]:
        cap = "Low"
        notes.append(f"the model has only trained on {brain.stats.get('tokens_seen', 0) / 1e6:.1f}M "
                     f"tokens (under {cfg['min_tokens_trained'] / 1e6:.0f}M)")
    return {"claim_loss": round(claim_loss, 3), "typical_loss": round(typical, 3),
            "ratio": round(ratio, 2), "cap": cap, "notes": notes}


def retrieve(claim: str, settings: dict) -> list[dict]:
    """Top knowledge-base passages for the claim (step 10)."""
    mem = settings["memory"]
    return knowledge_store.search(claim, k=mem["top_k"], settings=settings,
                                  min_relevance=mem["min_relevance"])


def run_judge(claim: str, panel: dict, passages: list[dict], familiarity: dict, brain,
              settings: dict, rng=None, gen_rng=None) -> dict:
    cfg = settings["council"]
    believer, skeptic, investor = panel["believer"], panel["skeptic"], panel.get("investor")
    truth_margin = believer["signal"] - skeptic["signal"]
    margin = truth_margin if investor is None else (truth_margin + investor["signal"]) / 2

    if margin > cfg["mixed_margin"]:
        stance, relied_on = "yes", ["believer"]
    elif margin < -cfg["mixed_margin"]:
        stance, relied_on = "no", ["skeptic"]
    else:
        stance, relied_on = "mixed", list(panel)
    if investor is not None and stance != "mixed" and (investor["signal"] > 0) == (stance == "yes"):
        relied_on.append("investor")

    own = confidence_from_signal(abs(margin), cfg["confidence_thresholds"])
    # The ceiling: never more confident than the weakest input relied on,
    # nor than the model's familiarity with the claim allows.
    confidence = min_confidence(own, familiarity["cap"], *(panel[n]["confidence"] for n in relied_on))

    parts = [f"Believer signal {believer['signal']:+.2f} vs. Skeptic {skeptic['signal']:+.2f} "
             f"nats/token (margin {truth_margin:+.2f})."]
    if investor is not None:
        parts.append(f"Investor {investor['signal']:+.2f}; combined margin {margin:+.2f}.")
    unresolved = []
    if stance == "mixed":
        unresolved.append("neither side's signal clearly beats the other")
    unresolved += familiarity["notes"]

    result = {
        "persona": "judge",
        "verdict": VERDICTS[stance],
        "stance": stance,
        "margin": round(margin, 3),
        "reasoning": " ".join(parts),
        "in_its_own_words": Agent("judge", brain, settings).speak(build_prompt(claim, passages), gen_rng),
        "relied_on": relied_on,
        "confidence": confidence,
        "unresolved": "; ".join(unresolved) or None,
    }
    if confidence != own:
        result["confidence_before_cap"] = own
    return result


def run_council(claim: str, *, brain, routing: dict, passages: list[dict], familiarity: dict,
                settings: dict, stochastic: bool = False, seed: int | None = None) -> dict:
    """One pass: panel then Judge. `stochastic` turns dropout on while
    scoring (Monte Carlo dropout), used by the uncertainty re-run."""
    seeds = np.random.SeedSequence(seed)
    score_rng = np.random.default_rng(seeds.spawn(1)[0]) if stochastic else None
    gen_rng = np.random.default_rng(seeds.spawn(1)[0])
    names = ["believer", "skeptic"] + (["investor"] if routing["is_money_idea"] else [])
    log.info("Panel deliberating: %s", ", ".join(n.capitalize() for n in names))
    panel = {n: Agent(n, brain, settings).respond(claim, passages, familiarity_cap=familiarity["cap"],
                                                  rng=score_rng, gen_rng=gen_rng)
             for n in names}
    log.info("Judge deliberating")
    judge = run_judge(claim, panel, passages, familiarity, brain, settings, score_rng, gen_rng)
    return {"panel": panel, "judge": judge}


def evaluate(claim: str, *, recheck: bool | None = None, record: bool = True,
             settings: dict | None = None) -> dict:
    """Run one full council session; return the Judge's verdict plus each
    persona's response.

    recheck: re-run with dropout on and compare (defaults to
    uncertainty.enabled in settings.yaml). record: log the session so the
    research loop can learn what you ask about.
    """
    settings = settings if settings is not None else load_settings()
    brain = load_brain(settings)
    routing = route(claim, settings)
    passages = retrieve(claim, settings)
    log.info("Found %d relevant passage(s) in the knowledge base", len(passages))
    familiarity = assess_familiarity(brain, claim, settings)
    shared = dict(brain=brain, routing=routing, passages=passages, familiarity=familiarity,
                  settings=settings)

    council = run_council(claim, **shared)
    result = {
        "claim": claim,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"description": brain.describe(), "steps": brain.stats["steps"],
                  "tokens_seen": brain.stats["tokens_seen"], "val_loss": brain.stats["val_loss"]},
        "routing": routing,
        "memory": [{"text": p["text"], "relevance": p["relevance"], **p.get("metadata", {})}
                   for p in passages],
        "familiarity": familiarity,
        **council,
        "overall_confidence": council["judge"]["confidence"],
    }

    if recheck is None:
        recheck = settings["uncertainty"]["enabled"]
    if recheck:
        log.info("Re-running the council with dropout on, to check it agrees with itself")
        rerun = run_council(claim, stochastic=True, **shared)
        result = uncertainty.reconcile(result, rerun["judge"])

    if record:
        session_log.record(result, settings)
    return result
