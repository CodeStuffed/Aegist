"""Every op's hand-written gradient matches finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from model_backend import autograd as ag
from model_backend.transformer import ModelConfig, TransformerLM

rng = np.random.default_rng(0)


def numeric_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        plus = f()
        x[i] = old - eps
        minus = f()
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def check(build, *shapes):
    tensors = [ag.Tensor(rng.standard_normal(s), requires_grad=True) for s in shapes]
    weights = rng.standard_normal(build(*tensors).shape)

    def scalar():
        return float((build(*tensors).data * weights).sum())

    out = build(*tensors)
    out._backward(weights)  # push an arbitrary upstream gradient through the op
    for t in tensors:
        np.testing.assert_allclose(t.grad, numeric_grad(scalar, t.data), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name,build,shapes", [
    ("add broadcast", lambda a, b: ag.add(a, b), [(2, 3, 4), (4,)]),
    ("matmul shared weight", lambda a, b: ag.matmul(a, b), [(2, 3, 4), (4, 5)]),
    ("matmul batched", lambda a, b: ag.matmul(a, b), [(2, 3, 4), (2, 4, 5)]),
    ("transpose", lambda a: ag.transpose(a, (0, 2, 1, 3)), [(2, 3, 4, 5)]),
    ("reshape", lambda a: ag.reshape(a, (6, 4)), [(2, 3, 4)]),
    ("layernorm", lambda x, g, b: ag.layernorm(x, g, b), [(2, 3, 6), (6,), (6,)]),
    ("gelu", lambda x: ag.gelu(x), [(3, 4)]),
    ("causal attention", lambda q, k, v: ag.causal_attention(q, k, v), [(2, 2, 5, 3)] * 3),
    ("cross entropy", lambda l: ag.cross_entropy(l, np.array([[1, 0, 3], [2, 2, 1]])), [(2, 3, 4)]),
])
def test_op_gradients(name, build, shapes):
    check(build, *shapes)


def test_embedding_gradient_accumulates_repeated_ids():
    w = ag.Tensor(rng.standard_normal((6, 3)), requires_grad=True)
    idx = np.array([[1, 1, 4], [0, 5, 1]])
    out = ag.embedding(w, idx)
    out._backward(np.ones(out.shape))
    assert np.allclose(w.grad[1], 3.0) and np.allclose(w.grad[2], 0.0)


def test_attention_is_causal():
    q, k, v = (ag.Tensor(rng.standard_normal((1, 1, 4, 2))) for _ in range(3))
    before = ag.causal_attention(q, k, v).data[..., :2, :].copy()
    v.data[..., 3, :] += 100.0  # changing the last position can't affect earlier outputs
    assert np.allclose(ag.causal_attention(q, k, v).data[..., :2, :], before)


def test_full_model_gradient():
    model = TransformerLM(ModelConfig(vocab_size=11, block_size=6, n_layer=2, n_head=2, d_model=8,
                                      dropout=0.0), seed=1, dtype=np.float64)
    x, y = rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))
    model.forward(x, y)[1].backward()
    for name in ["wte", "wpe", "h0.attn.wq", "h1.mlp.wproj", "h1.ln2.g", "lnf.b"]:
        p = model.params[name]
        np.testing.assert_allclose(p.grad, numeric_grad(lambda: float(model.forward(x, y)[1].data), p.data),
                                   rtol=1e-4, atol=1e-8)


def test_no_grad_records_nothing():
    a = ag.Tensor(np.ones(3), requires_grad=True)
    with ag.no_grad():
        out = a + a
    assert out._parents == () and not out.requires_grad


def test_dropout_identity_without_rng():
    x = ag.Tensor(np.ones((4, 4)))
    assert ag.dropout(x, 0.5, None) is x
    dropped = ag.dropout(x, 0.5, np.random.default_rng(0)).data
    assert set(np.unique(dropped)) <= {0.0, 2.0}
