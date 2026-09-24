"""BPE tokenizer and hardware tiers."""

from __future__ import annotations

import pytest

from config import load_settings
from model_backend.hardware_detect import detect_hardware, pick_tier
from model_backend.tokenizer import Tokenizer

TEXT = "The rays of light are refracted by the prism. The prism separates the colours. " * 30


def test_roundtrip_any_unicode():
    tok = Tokenizer.train(TEXT, 300)
    for s in [TEXT, "Héllo wörld — 日本語 ✓ $9.99/month", "", "\n\n  tabs\there"]:
        assert tok.decode(tok.encode(s)) == s


def test_training_compresses_and_respects_vocab_size():
    tok = Tokenizer.train(TEXT, 300)
    assert tok.vocab_size == 300
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 3


def test_merges_survive_serialization():
    tok = Tokenizer.train(TEXT, 290)
    again = Tokenizer([list(m) for m in tok.merges])  # as stored in brain.json
    assert again.encode(TEXT) == tok.encode(TEXT)


def test_stops_when_nothing_repeats():
    assert Tokenizer.train("abc", 1000).vocab_size < 1000


TIERS = load_settings()["model"]["tiers"]


@pytest.mark.parametrize("gb,tier", [(7.6, "tiny"), (8, "tiny"), (15.5, "small"),
                                     (31.2, "medium"), (64, "large")])
def test_pick_tier_uses_settings_thresholds(gb, tier):
    assert pick_tier(gb, TIERS) == tier


def test_detect_hardware_reports_model_size():
    hw = detect_hardware()
    assert hw["tier"] in TIERS and hw["ram_gb"] > 0 and hw["cpu_cores"] >= 1
    assert hw["model_size"]["n_layer"] == TIERS[hw["tier"]]["n_layer"]
