"""Step 4: router, and the shared JSON-call helper it uses."""

from __future__ import annotations

from engine.llm_json import RETRY_NUDGE, call_json, extract_json
from engine.router import route
from model_backend import local_runner
from tests.fakes import ScriptedBackend


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    assert extract_json("[1, 2]") is None
    assert extract_json("no json here") is None


def test_call_json_retries_once_with_nudge():
    backend = ScriptedBackend("I think yes.", '{"ok": true}')
    parsed, meta = call_json(backend, "sys", "prompt")
    assert parsed == {"ok": True} and meta["attempts"] == 2
    assert backend.calls[1]["prompt"] == "prompt" + RETRY_NUDGE


def test_call_json_surfaces_raw_after_second_failure():
    backend = ScriptedBackend("nope", "still nope")
    parsed, meta = call_json(backend, "sys", "prompt")
    assert parsed is None and meta["raw"] == "still nope" and len(backend.calls) == 2


def test_route_money_idea():
    backend = ScriptedBackend('{"is_money_idea": "true", "reason": "Pricing change.", "topic": "Usage Pricing"}')
    assert route("We should switch to usage-based pricing", backend) == {
        "is_money_idea": True, "reason": "Pricing change.", "topic": "usage pricing"}
    assert backend.calls[0]["role"] == "router"


def test_route_not_money():
    backend = ScriptedBackend('{"is_money_idea": false, "reason": "Factual claim.", "topic": "sleep"}')
    assert route("Humans need 8 hours of sleep", backend)["is_money_idea"] is False


def test_route_unparseable_skips_investor():
    out = route("x", ScriptedBackend("garbage", "garbage"))
    assert out["is_money_idea"] is False and "valid JSON" in out["reason"]


def test_route_over_real_local_runner(fake_ollama):
    assert route("We should launch a $9/month subscription tier", local_runner)["is_money_idea"] is True
    assert route("The Great Wall is visible from orbit", local_runner)["is_money_idea"] is False
