"""Personas, the Judge's rules, orchestration, and the uncertainty check."""

from __future__ import annotations

import numpy as np
import pytest

from engine import uncertainty
from engine.agents import Agent, build_prompt, min_confidence
from engine.memory import knowledge_store, session_log
from engine.orchestrator import evaluate, run_judge
from model_backend.brain import load_brain

FAMILIAR = {"cap": "High", "notes": []}


def panel(believer, skeptic, investor=None, conf="High"):
    p = {"believer": {"signal": believer, "confidence": conf},
         "skeptic": {"signal": skeptic, "confidence": conf}}
    if investor is not None:
        p["investor"] = {"signal": investor, "confidence": conf}
    return p


@pytest.fixture
def judge(settings, trained):
    brain = load_brain(settings)
    return lambda p, fam=FAMILIAR: run_judge("A claim.", p, [], fam, brain, settings,
                                             gen_rng=np.random.default_rng(0))


def test_min_confidence():
    assert min_confidence("High", "Medium") == "Medium"
    assert min_confidence("High", None) == "Low"


def test_build_prompt_puts_best_passage_next_to_claim():
    prompt = build_prompt("Claim", [{"text": "best"}, {"text": "second"}])
    assert prompt == "second\n\nbest\n\nClaim."


def test_judge_can_say_no(judge):
    out = judge(panel(0.1, 0.9))
    assert out["stance"] == "no" and out["verdict"].startswith("No") and out["relied_on"] == ["skeptic"]
    assert out["confidence"] == "High"


def test_judge_says_undecided_on_close_signals(judge):
    out = judge(panel(0.30, 0.25))
    assert out["stance"] == "mixed" and out["confidence"] == "Low" and out["unresolved"]


def test_judge_capped_by_weakest_input_relied_on(judge):
    p = panel(0.9, 0.0)
    p["believer"]["confidence"] = "Medium"
    out = judge(p)
    assert out["stance"] == "yes" and out["confidence"] == "Medium"
    assert out["confidence_before_cap"] == "High"


def test_judge_capped_by_unfamiliar_claim(judge):
    out = judge(panel(0.9, 0.0), {"cap": "Low", "notes": ["the claim is unlike most of what it read"]})
    assert out["confidence"] == "Low" and "unlike" in out["unresolved"]


def test_investor_moves_the_margin(judge):
    assert judge(panel(0.5, 0.3, investor=-1.0))["stance"] == "no"
    out = judge(panel(0.5, 0.3, investor=0.8))
    assert out["stance"] == "yes" and out["relied_on"] == ["believer", "investor"]


def test_agent_scores_writes_and_cites_evidence(settings, trained):
    brain = load_brain(settings)
    passages = [{"text": "This is true. Light bends in glass.", "metadata": {"title": "Optics"}}]
    out = Agent("believer", brain, settings).respond("Light is refracted by glass", passages,
                                                     gen_rng=np.random.default_rng(0))
    assert out["position"].startswith("This is true because")
    assert isinstance(out["signal"], float) and out["confidence"] in ("Low", "Medium", "High")
    assert all(k["delta"] > 0 and k["source"] == "Optics" for k in out["key_points"])


def test_agent_familiarity_cap(settings, trained):
    out = Agent("skeptic", load_brain(settings), settings).respond("x", familiarity_cap="Low")
    assert out["confidence"] == "Low"


def test_unknown_persona(settings, trained):
    with pytest.raises(ValueError, match="available"):
        Agent("oracle", load_brain(settings), settings)


def test_evaluate_end_to_end(settings, trained):
    knowledge_store.add("Customers pay for products that are profitable to sell.",
                        {"title": "Markets"}, settings)
    result = evaluate("Customers will pay for a profitable subscription", settings=settings)
    assert list(result["panel"]) == ["believer", "skeptic", "investor"]
    assert result["memory"] and result["memory"][0]["title"] == "Markets"
    assert result["judge"]["stance"] in ("yes", "no", "mixed")
    assert "consistency" in result and result["overall_confidence"] in ("Low", "Medium", "High")
    assert session_log.sessions(settings)[-1]["claim"].startswith("Customers")


def test_evaluate_plain_claim_skips_investor(settings, trained):
    result = evaluate("Light bends in glass", settings=settings, recheck=False, record=False)
    assert list(result["panel"]) == ["believer", "skeptic"] and "consistency" not in result
    assert session_log.sessions(settings) == []


def test_uncertainty_flags_disagreement():
    first = {"judge": {"stance": "yes", "confidence": "High"}, "overall_confidence": "High"}
    agreed = uncertainty.reconcile(first, {"stance": "yes", "confidence": "High"})
    assert agreed["consistency"]["agreed"] and agreed["overall_confidence"] == "High"
    flipped = uncertainty.reconcile(first, {"stance": "no", "confidence": "High"})
    assert flipped["overall_confidence"] == "Low" and flipped["confidence_note"] == uncertainty.LOW_NOTE
    assert flipped["consistency"]["rerun_judge"]["stance"] == "no"
    shifted = uncertainty.reconcile(first, {"stance": "yes", "confidence": "Medium"})
    assert shifted["consistency"]["reasons"] == ["confidence changed: High -> Medium"]
