"""Step 5: Agent loads a persona, calls a backend, parses JSON."""

from __future__ import annotations

import pytest

from engine.agents import Agent, normalize_confidence
from model_backend import local_runner
from tests.fakes import ScriptedBackend

GOOD = ('{"position": "p", "key_points": ["a"], "confidence": "high", '
        '"what_would_change_my_mind": "w"}')


def test_loads_persona_prompt_and_role():
    believer = Agent("believer", ScriptedBackend())
    judge = Agent("judge", ScriptedBackend())
    assert believer.system_prompt.startswith("You are the Believer")
    assert believer.role == "panel" and judge.role == "judge"


def test_unknown_persona():
    with pytest.raises(ValueError, match="available"):
        Agent("oracle", ScriptedBackend())


def test_respond_parses_and_normalizes():
    backend = ScriptedBackend(GOOD)
    out = Agent("skeptic", backend).respond("Claim here", context="EXTRA CONTEXT", temperature=0.5)
    assert out["confidence"] == "High" and out["position"] == "p" and out["persona"] == "skeptic"
    call = backend.calls[0]
    assert call["system"].startswith("You are the Skeptic")
    assert call["prompt"] == "CLAIM:\nClaim here\n\nEXTRA CONTEXT"
    assert call["temperature"] == 0.5


def test_respond_retries_then_succeeds():
    out = Agent("believer", ScriptedBackend("Great idea!!", GOOD)).respond("c")
    assert out["attempts"] == 2 and "error" not in out


def test_respond_surfaces_raw_text_after_two_failures():
    out = Agent("believer", ScriptedBackend("Great idea!!", "Still great!!")).respond("c")
    assert out["raw"] == "Still great!!" and "error" in out and out["confidence"] is None


@pytest.mark.parametrize("raw,want", [("High", "High"), (" medium ", "Medium"), ("LOW", "Low"),
                                      ("very high", None), (None, None), (3, None)])
def test_normalize_confidence(raw, want):
    assert normalize_confidence(raw) == want


def test_agent_over_real_local_runner(fake_ollama):
    out = Agent("investor", local_runner).respond("We should sell a $9/month plan")
    assert out["backend"] == "local" and out["confidence"] == "Low" and out["key_points"]
