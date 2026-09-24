from __future__ import annotations

import pytest

from config import load_settings
from engine.memory import knowledge_store
from model_backend import brain as brain_module
from model_backend import trainer

CORPUS = """\
Light is refracted when it passes from air into glass. The rays of light bend toward the
perpendicular. This is true. White light is made of many colours, and a prism separates them.

The market for a product depends on customers who will pay for it. A business that charges
more than its costs is profitable. It will make money when customers pay for it.

Heavy objects do not fall faster than light objects in a vacuum. This is false. That is not
right, because gravity accelerates every body at the same rate.
"""


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own empty data dir; nothing touches data/."""
    monkeypatch.setenv("COUNCIL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("COUNCIL_SETTINGS", raising=False)
    knowledge_store._stores.clear()
    brain_module._loaded.clear()
    yield
    knowledge_store._stores.clear()
    brain_module._loaded.clear()


@pytest.fixture
def settings():
    """Real settings.yaml, shrunk so a model trains in about a second."""
    s = load_settings()
    s["model"]["tiers"] = {"test": {"max_ram_gb": 9999, "n_layer": 1, "n_head": 2, "d_model": 32,
                                    "block_size": 48, "vocab_size": 320}}
    s["training"].update(batch_size=8, min_corpus_tokens=100, warmup_steps=5, learning_rate=3e-3,
                         eval_every_s=3600,
                         log_every_s=3600, checkpoint_every_s=3600)
    s["council"]["generate"]["max_new_tokens"] = 12
    s["council"]["min_tokens_trained"] = 0
    return s


@pytest.fixture
def corpus(settings, tmp_path):
    src = tmp_path / "texts"
    src.mkdir()
    (src / "notes.txt").write_text(CORPUS * 20)
    trainer.import_texts(src, settings)
    return src


@pytest.fixture
def trained(settings, corpus):
    """A small model trained on CORPUS; returns the train() stats."""
    return trainer.train(steps=40, settings=settings, log=lambda *_: None, seed=0)
