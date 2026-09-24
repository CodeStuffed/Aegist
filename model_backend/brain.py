"""
The trained model at inference time: what the council actually talks to.

Three things it can do:
  generate(prompt)                 sample a continuation, token by token
  continuation_logprob(p, c)       how likely the model finds text c after p
  text_nll(text)                   how surprising text is overall (familiarity)

Passing an `rng` to the scoring methods switches dropout on (Monte Carlo
dropout): the same question gets a slightly different network each time,
which is how the uncertainty check tells a firm answer from a coin flip.
"""

from __future__ import annotations

import json

import numpy as np

from config import load_settings
from model_backend import autograd as ag
from model_backend.trainer import brain_dir, load_checkpoint


class NoBrainError(RuntimeError):
    pass


class Brain:
    def __init__(self, model, tokenizer, meta: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta
        self.block_size = model.config.block_size
        self._lead = tokenizer.encode("\n")  # context for text that starts from nothing

    @classmethod
    def load(cls, settings: dict | None = None) -> "Brain":
        loaded = load_checkpoint(settings)
        if loaded is None:
            raise NoBrainError(
                f"No trained model in {brain_dir(settings)}. Train one first: "
                "`python cli.py train --data <folder of .txt/.md files> --hours 1`, "
                "or `python cli.py research --hours 1` to collect text and train on it.")
        return cls(*loaded)

    @property
    def stats(self) -> dict:
        return self.meta["stats"]

    def describe(self) -> str:
        c = self.model.config
        return f"{c.n_layer}x{c.d_model} transformer, {self.model.num_params / 1e6:.2f}M params"

    def _logprobs(self, ids: list[int], rng=None) -> np.ndarray:
        """Log-probabilities of the next token at every position of `ids`."""
        with ag.no_grad():
            logits, _ = self.model.forward(np.array([ids], dtype=np.int64), rng=rng)
        return ag.log_softmax(logits.data[0])

    def continuation_logprob(self, prompt: str, continuation: str, rng=None) -> float:
        """Mean log-probability per token of `continuation` following `prompt`.
        Long prompts are cut from the left: the end of the prompt matters most."""
        cont = self.tokenizer.encode(continuation)[: self.block_size - 1]
        ids = (self._lead + self.tokenizer.encode(prompt) + cont)[-(self.block_size + 1):]
        logp = self._logprobs(ids[:-1], rng)
        targets = ids[-len(cont):]
        rows = np.arange(len(ids) - 1 - len(cont), len(ids) - 1)
        return float(logp[rows, targets].mean())

    def text_nll(self, text: str) -> float:
        """Mean negative log-likelihood per token of `text` (lower = more familiar)."""
        return -self.continuation_logprob("", text)

    def generate(self, prompt: str, *, max_new_tokens: int = 48, temperature: float = 0.8,
                 top_k: int = 40, rng: np.random.Generator | None = None) -> str:
        """Sample a continuation. Stops at a paragraph break or after two sentences."""
        rng = rng if rng is not None else np.random.default_rng()
        ids = self._lead + self.tokenizer.encode(prompt)
        start = len(ids)
        for _ in range(max_new_tokens):
            logp = self._logprobs(ids[-self.block_size:])[-1]
            if temperature <= 0:
                nxt = int(np.argmax(logp))
            else:
                scaled = logp / temperature
                if 0 < top_k < scaled.size:
                    scaled[scaled < np.partition(scaled, -top_k)[-top_k]] = -np.inf
                probs = np.exp(scaled - scaled.max())
                nxt = int(rng.choice(scaled.size, p=probs / probs.sum()))
            ids.append(nxt)
            text = self.tokenizer.decode(ids[start:])
            if "\n\n" in text.strip() or sum(text.count(p) for p in ".!?") >= 2:
                break
        text = self.tokenizer.decode(ids[start:]).strip().split("\n\n")[0]
        return " ".join(text.split())


_loaded: dict[tuple[str, str], Brain] = {}


def load_brain(settings: dict | None = None) -> Brain:
    """The current checkpoint, loaded once and reloaded only when it changes."""
    meta_path = brain_dir(settings) / "brain.json"
    if not meta_path.exists():
        return Brain.load(settings)  # raises NoBrainError with instructions
    key = (str(meta_path), json.loads(meta_path.read_text())["stats"]["updated_at"])
    if key not in _loaded:
        _loaded.clear()
        _loaded[key] = Brain.load(settings)
    return _loaded[key]
