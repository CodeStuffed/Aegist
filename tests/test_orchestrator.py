"""Step 6: orchestration, Investor routing, and the Judge's confidence ceiling."""

from __future__ import annotations

from engine.orchestrator import apply_confidence_cap, evaluate
from model_backend import local_runner


def test_evaluate_money_idea_runs_investor(fake_ollama):
    result = evaluate("We should switch our SaaS to usage-based pricing", backend=local_runner)
    assert list(result["panel"]) == ["believer", "skeptic", "investor"]
    assert result["routing"]["is_money_idea"] is True
    judge_prompt = fake_ollama.requests[-1]["messages"][1]["content"]
    assert "INVESTOR'S CASE" in judge_prompt and '"stance"' in judge_prompt


def test_evaluate_plain_claim_skips_investor(fake_ollama):
    result = evaluate("The Great Wall is visible from orbit", backend=local_runner)
    assert list(result["panel"]) == ["believer", "skeptic"]


def test_cap_uses_relied_on_inputs():
    panel = {"believer": {"confidence": "Medium"}, "skeptic": {"confidence": "High"},
             "investor": {"confidence": "Low"}}
    judge = apply_confidence_cap({"confidence": "High", "relied_on": ["Skeptic"]}, panel)
    assert judge["confidence"] == "High" and "confidence_before_cap" not in judge
    judge = apply_confidence_cap({"confidence": "High", "relied_on": ["skeptic", "believer"]}, panel)
    assert judge["confidence"] == "Medium" and judge["confidence_before_cap"] == "High"


def test_cap_without_relied_on_counts_every_input():
    panel = {"believer": {"confidence": "High"}, "skeptic": {"confidence": "Low"}}
    assert apply_confidence_cap({"confidence": "High"}, panel)["confidence"] == "Low"


def test_unparseable_judge_is_low():
    panel = {"believer": {"confidence": "High"}, "skeptic": {"confidence": "High"}}
    assert apply_confidence_cap({"confidence": None, "error": "x"}, panel)["confidence"] == "Low"
