"""Corpus handling, training, checkpoints, and the inference-time Brain."""

from __future__ import annotations

import numpy as np
import pytest

from model_backend import trainer
from model_backend.brain import Brain, NoBrainError, load_brain


def test_normalize_unwraps_hard_wrapped_lines():
    assert trainer.normalize_text("one\ntwo  three\n\nfour\r\nfive") == "one two three\n\nfour five"


def test_import_texts_copies_only_text_files(settings, tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.md").write_text("beta")
    (src / "c.pdf").write_bytes(b"%PDF")
    assert trainer.import_texts(src, settings) == 2
    assert {p.name for p in trainer.corpus_files(settings)} == {"a.txt", "b.md"}


def test_empty_corpus_is_a_clear_error(settings):
    with pytest.raises(trainer.TrainingError, match="corpus is empty"):
        trainer.train(steps=1, settings=settings, log=lambda *_: None)


def test_tiny_corpus_is_a_clear_error(settings, tmp_path):
    (tmp_path / "t.txt").write_text("too short")
    trainer.import_texts(tmp_path / "t.txt", settings)
    with pytest.raises(trainer.TrainingError, match="KB of text so far"):
        trainer.train(steps=1, settings=settings, log=lambda *_: None)
    settings["training"]["min_new_model_chars"] = 1
    with pytest.raises(trainer.TrainingError, match="only"):
        trainer.train(steps=1, settings=settings, log=lambda *_: None)


def test_training_learns_and_resumes(settings, corpus):
    first = trainer.train(steps=40, settings=settings, log=lambda *_: None, seed=0)
    vocab = Brain.load(settings).tokenizer.vocab_size
    assert first["val_loss"] < np.log(vocab) - 1.0  # well below random guessing
    second = trainer.train(steps=10, settings=settings, log=lambda *_: None, seed=1)
    assert second["steps"] == 50 and second["steps_this_session"] == 10
    assert (trainer.brain_dir(settings) / "optim.npz").exists()


def test_token_cache_reused_until_file_changes(settings, trained):
    tok = Brain.load(settings).tokenizer
    cache = trainer.brain_dir(settings) / "token_cache"
    before = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.npy")}
    trainer.corpus_tokens(tok, settings)
    assert {p.name: p.stat().st_mtime_ns for p in cache.glob("*.npy")} == before


def test_no_model_yet(settings):
    with pytest.raises(NoBrainError, match="train"):
        load_brain(settings)


def test_brain_scores_generates_and_senses_familiarity(settings, trained):
    brain = load_brain(settings)
    assert load_brain(settings) is brain  # cached until the checkpoint changes
    seen = brain.continuation_logprob("Light is refracted when it passes from air into", " glass.")
    unseen = brain.continuation_logprob("Light is refracted when it passes from air into", " zebra.")
    assert seen > unseen
    assert brain.text_nll("The rays of light bend toward the perpendicular.") < \
        brain.text_nll("Zxq vbnm qwpl kjhg tyvr.") - 0.5
    text = brain.generate("White light is", max_new_tokens=8, rng=np.random.default_rng(0))
    assert isinstance(text, str) and "\n\n" not in text


def test_mc_dropout_perturbs_scores(settings, trained):
    brain = load_brain(settings)
    plain = brain.continuation_logprob("The market", " grows.")
    assert brain.continuation_logprob("The market", " grows.") == plain
    noisy = {round(brain.continuation_logprob("The market", " grows.", np.random.default_rng(i)), 6)
             for i in range(4)}
    assert len(noisy) > 1
