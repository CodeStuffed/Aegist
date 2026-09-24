"""
A GPT-style decoder-only transformer, written from scratch on the autograd
engine in autograd.py. No pretrained weights: it starts as random numbers
and learns only from the text it's trained on.

Layout per block (pre-LayerNorm):
    x = x + Dropout(Attention(LayerNorm(x)))
    x = x + Dropout(MLP(LayerNorm(x)))       MLP = Linear -> GELU -> Linear
The output layer shares its weights with the token embedding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from model_backend import autograd as ag
from model_backend.autograd import Tensor


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128   # longest context, in tokens
    n_layer: int = 4        # transformer blocks ("hidden layers")
    n_head: int = 4
    d_model: int = 192
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerLM:
    def __init__(self, config: ModelConfig, seed: int = 0, dtype=np.float32):
        if config.d_model % config.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.config = config
        rng = np.random.default_rng(seed)
        C, L = config.d_model, config.n_layer

        def w(*shape, std=0.02):
            return Tensor(rng.normal(0.0, std, shape).astype(dtype), requires_grad=True)

        def const(value, *shape):
            return Tensor(np.full(shape, value, dtype=dtype), requires_grad=True)

        residual_std = 0.02 / np.sqrt(2 * L)  # keeps the residual stream from growing with depth
        p = {"wte": w(config.vocab_size, C), "wpe": w(config.block_size, C)}
        for i in range(L):
            p.update({
                f"h{i}.ln1.g": const(1.0, C), f"h{i}.ln1.b": const(0.0, C),
                f"h{i}.attn.wq": w(C, C), f"h{i}.attn.bq": const(0.0, C),
                f"h{i}.attn.wk": w(C, C), f"h{i}.attn.bk": const(0.0, C),
                f"h{i}.attn.wv": w(C, C), f"h{i}.attn.bv": const(0.0, C),
                f"h{i}.attn.wo": w(C, C, std=residual_std), f"h{i}.attn.bo": const(0.0, C),
                f"h{i}.ln2.g": const(1.0, C), f"h{i}.ln2.b": const(0.0, C),
                f"h{i}.mlp.wfc": w(C, 4 * C), f"h{i}.mlp.bfc": const(0.0, 4 * C),
                f"h{i}.mlp.wproj": w(4 * C, C, std=residual_std), f"h{i}.mlp.bproj": const(0.0, C),
            })
        p["lnf.g"], p["lnf.b"] = const(1.0, C), const(0.0, C)
        self.params: dict[str, Tensor] = p

    @property
    def num_params(self) -> int:
        return sum(t.data.size for t in self.params.values())

    def _split_heads(self, x: Tensor, B: int, T: int) -> Tensor:
        H = self.config.n_head
        return ag.transpose(ag.reshape(x, (B, T, H, self.config.d_model // H)), (0, 2, 1, 3))

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None,
                rng: np.random.Generator | None = None) -> tuple[Tensor, Tensor | None]:
        """idx: (B, T) token ids. Pass `rng` to switch dropout on (training,
        or Monte Carlo dropout at inference). Returns (logits, loss or None)."""
        cfg, p = self.config, self.params
        B, T = idx.shape
        if T > cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {cfg.block_size}")
        drop = cfg.dropout

        x = ag.embedding(p["wte"], idx) + ag.embedding(p["wpe"], np.arange(T))
        x = ag.dropout(x, drop, rng)
        for i in range(cfg.n_layer):
            h = ag.layernorm(x, p[f"h{i}.ln1.g"], p[f"h{i}.ln1.b"])
            q = self._split_heads(h @ p[f"h{i}.attn.wq"] + p[f"h{i}.attn.bq"], B, T)
            k = self._split_heads(h @ p[f"h{i}.attn.wk"] + p[f"h{i}.attn.bk"], B, T)
            v = self._split_heads(h @ p[f"h{i}.attn.wv"] + p[f"h{i}.attn.bv"], B, T)
            a = ag.reshape(ag.transpose(ag.causal_attention(q, k, v), (0, 2, 1, 3)), (B, T, cfg.d_model))
            x = x + ag.dropout(a @ p[f"h{i}.attn.wo"] + p[f"h{i}.attn.bo"], drop, rng)

            h = ag.layernorm(x, p[f"h{i}.ln2.g"], p[f"h{i}.ln2.b"])
            h = ag.gelu(h @ p[f"h{i}.mlp.wfc"] + p[f"h{i}.mlp.bfc"])
            x = x + ag.dropout(h @ p[f"h{i}.mlp.wproj"] + p[f"h{i}.mlp.bproj"], drop, rng)

        x = ag.layernorm(x, p["lnf.g"], p["lnf.b"])
        logits = x @ ag.transpose(p["wte"], (1, 0))
        loss = ag.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: t.data for name, t in self.params.items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        missing = set(self.params) - set(state)
        if missing:
            raise ValueError(f"checkpoint is missing {sorted(missing)[:3]}...")
        for name, t in self.params.items():
            if state[name].shape != t.data.shape:
                raise ValueError(f"{name}: checkpoint shape {state[name].shape} != {t.data.shape}")
            t.data = state[name].astype(t.data.dtype)
