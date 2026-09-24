"""Steps 1-2: hardware tiers and the two runners."""

from __future__ import annotations

import pytest

import model_backend
from config import load_settings
from model_backend import cloud_runner, local_runner
from model_backend.hardware_detect import detect_hardware, parse_nvidia_smi, pick_tier
from tests.fakes import message_body

TIERS = load_settings()["local"]["tiers"]


@pytest.mark.parametrize("gb,tier", [(7.6, "tiny"), (8, "tiny"), (15.5, "small"),
                                     (31.2, "medium"), (64, "large"), (512, "large")])
def test_pick_tier_uses_settings_thresholds(gb, tier):
    assert pick_tier(gb, TIERS) == tier


def test_pick_tier_follows_edited_thresholds():
    custom = {"a": {"max_ram_gb": 4}, "b": {"max_ram_gb": 100}}
    assert pick_tier(3, custom) == "a"
    assert pick_tier(50, custom) == "b"
    assert pick_tier(500, custom) == "b"


def test_parse_nvidia_smi_sums_gpus():
    assert parse_nvidia_smi("24576 MiB\n8192 MiB\n") == 32.0


def test_detect_hardware_shape():
    hw = detect_hardware()
    assert hw["tier"] in TIERS
    assert hw["ram_gb"] > 0 and hw["cpu_cores"] >= 1 and hw["vram_gb"] >= 0


def test_local_generate_sends_tier_model_and_json_format(fake_ollama):
    out = local_runner.generate("You are the Skeptic on this council.", "CLAIM:\nx", temperature=0.7)
    req = fake_ollama.requests[-1]
    expected_model = TIERS[detect_hardware()["tier"]]["ollama_model"]
    assert out["backend"] == "local" and out["model"] == expected_model == req["model"]
    assert req["format"] == "json" and req["stream"] is False and req["think"] is False
    assert req["options"]["temperature"] == 0.7 and req["options"]["num_ctx"] == 8192
    assert req["messages"][0] == {"role": "system", "content": "You are the Skeptic on this council."}
    assert '"confidence": "High"' in out["text"]


def test_local_strips_think_blocks(fake_ollama):
    fake_ollama.reply_override = lambda s, p: '<think>hmm</think>\n{"ok": true}'
    assert local_runner.generate("sys", "p")["text"] == '{"ok": true}'


def test_local_model_not_pulled_is_a_clear_error(fake_ollama):
    fake_ollama.pulled.clear()
    with pytest.raises(local_runner.LocalBackendError, match="ollama pull"):
        local_runner.generate("sys", "p")


def test_local_unreachable_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")  # nothing listens on the discard port
    with pytest.raises(local_runner.LocalBackendError, match="Can't reach Ollama"):
        local_runner.generate("sys", "p")


def test_cloud_generate_picks_model_and_effort_per_role(fake_anthropic):
    judge = cloud_runner.generate("You are the Judge on this council.", "CLAIM:\nx", role="judge",
                                  temperature=0.9)
    panel = cloud_runner.generate("You are the Believer on this council.", "CLAIM:\nx")
    judge_req, panel_req = fake_anthropic.requests
    assert judge["model"] == judge_req["model"] == "claude-opus-5-5"
    assert panel["model"] == panel_req["model"] == "claude-sonnet-5"
    assert judge_req["output_config"] == {"effort": "high"}
    assert panel_req["output_config"] == {"effort": "medium"}
    assert "temperature" not in judge_req  # these models reject sampling params
    # thinking block is dropped, text kept
    assert judge["backend"] == "cloud" and judge["text"].startswith('{"verdict"')


def test_cloud_refusal_raises(fake_anthropic):
    fake_anthropic.response_override = lambda body: {
        **message_body(body["model"], [], "refusal"),
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": "x"},
    }
    with pytest.raises(cloud_runner.CloudBackendError, match="declined"):
        cloud_runner.generate("sys", "p")


def test_select_backend_auto_prefers_reachable_cloud(fake_anthropic):
    assert model_backend.select_backend("auto") is cloud_runner


def test_select_backend_auto_falls_back_to_local_without_key():
    assert model_backend.select_backend("auto") is local_runner


def test_select_backend_cloud_without_key_errors():
    with pytest.raises(model_backend.BackendUnavailable, match="ANTHROPIC_API_KEY"):
        model_backend.select_backend("cloud")


def test_select_backend_respects_env(monkeypatch, fake_anthropic):
    monkeypatch.setenv("COUNCIL_BACKEND", "local")
    assert model_backend.select_backend() is local_runner
