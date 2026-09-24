"""
A tiny reverse-mode autograd engine on NumPy - the only "framework" the
council's neural net uses. Every op records how to push gradients back to
its inputs; Tensor.backward() walks the graph in reverse.

Ops are deliberately coarse (layernorm, causal attention and cross-entropy
are single fused ops with hand-derived gradients) so the Python overhead
stays small next to the matrix multiplies.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

_grad_enabled = True


@contextmanager
def no_grad():
    """Inference mode: ops skip recording the graph."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")

    def __init__(self, data, requires_grad: bool = False):
        self.data = np.asarray(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents: tuple = ()
        self._backward = None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def _accumulate(self, grad) -> None:
        if not self.requires_grad:
            return
        self.grad = grad if self.grad is None else self.grad + grad

    def backward(self) -> None:
        order, seen = [], set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)
                if node._parents:  # free intermediate grads as we go
                    node.grad = None

    # Operator sugar for the common cases.
    def __add__(self, other):
        return add(self, other)

    def __matmul__(self, other):
        return matmul(self, other)


def _result(data, parents, backward) -> Tensor:
    out = Tensor(data)
    if _grad_enabled and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._parents = parents
        out._backward = backward
    return out


def _unbroadcast(grad, shape):
    """Sum out the axes NumPy broadcast over, so grad matches `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def add(a: Tensor, b: Tensor) -> Tensor:
    def backward(g):
        a._accumulate(_unbroadcast(g, a.shape))
        b._accumulate(_unbroadcast(g, b.shape))
    return _result(a.data + b.data, (a, b), backward)


def mul_const(a: Tensor, c: np.ndarray) -> Tensor:
    """Multiply by a constant array (e.g. a dropout mask); no grad for `c`."""
    def backward(g):
        a._accumulate(_unbroadcast(g * c, a.shape))
    return _result(a.data * c, (a,), backward)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """a (..., m, k) @ b (k, n) or batched b (..., k, n)."""
    def backward(g):
        if a.requires_grad:
            a._accumulate(g @ np.swapaxes(b.data, -1, -2))
        if b.requires_grad:
            if b.data.ndim == 2:  # shared weight: fold every batch dim together
                k = a.data.shape[-1]
                b._accumulate(a.data.reshape(-1, k).T @ g.reshape(-1, g.shape[-1]))
            else:
                b._accumulate(_unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.shape))
    return _result(a.data @ b.data, (a, b), backward)


def transpose(a: Tensor, axes) -> Tensor:
    inverse = np.argsort(axes)
    def backward(g):
        a._accumulate(g.transpose(inverse))
    return _result(a.data.transpose(axes), (a,), backward)


def reshape(a: Tensor, shape) -> Tensor:
    def backward(g):
        a._accumulate(g.reshape(a.shape))
    return _result(a.data.reshape(shape), (a,), backward)


def embedding(weight: Tensor, idx: np.ndarray) -> Tensor:
    def backward(g):
        grad = np.zeros_like(weight.data)
        np.add.at(grad, idx, g)
        weight._accumulate(grad)
    return _result(weight.data[idx], (weight,), backward)


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    mean = x.data.mean(axis=-1, keepdims=True)
    centered = x.data - mean
    inv_std = 1.0 / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + eps)
    norm = centered * inv_std

    def backward(g):
        gamma._accumulate(_unbroadcast(g * norm, gamma.shape))
        beta._accumulate(_unbroadcast(g, beta.shape))
        if x.requires_grad:
            gn = g * gamma.data
            x._accumulate(inv_std * (gn - gn.mean(axis=-1, keepdims=True)
                                     - norm * (gn * norm).mean(axis=-1, keepdims=True)))
    return _result(norm * gamma.data + beta.data, (x, gamma, beta), backward)


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x: Tensor) -> Tensor:
    """tanh approximation, as in GPT-2."""
    x2 = x.data * x.data  # plain multiplies: np.power is several times slower
    t = np.tanh(_GELU_C * x.data * (1.0 + 0.044715 * x2))

    def backward(g):
        du = _GELU_C * (1.0 + 3 * 0.044715 * x2)
        x._accumulate(g * (0.5 * (1.0 + t) + 0.5 * x.data * (1.0 - t * t) * du))
    return _result(0.5 * x.data * (1.0 + t), (x,), backward)


def dropout(x: Tensor, p: float, rng: np.random.Generator | None) -> Tensor:
    """Inverted dropout. With rng=None (or p=0) it's the identity."""
    if rng is None or p <= 0:
        return x
    mask = (rng.random(x.shape) >= p).astype(x.data.dtype) / (1.0 - p)
    return mul_const(x, mask)


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """softmax(q k^T / sqrt(d), causally masked) v for (B, H, T, D) inputs."""
    T, d = q.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = (q.data @ np.swapaxes(k.data, -1, -2)) * scale
    scores = np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)

    def backward(g):
        dprobs = g @ np.swapaxes(v.data, -1, -2)
        dscores = probs * (dprobs - (dprobs * probs).sum(axis=-1, keepdims=True)) * scale
        q._accumulate(dscores @ k.data)
        k._accumulate(np.swapaxes(dscores, -1, -2) @ q.data)
        v._accumulate(np.swapaxes(probs, -1, -2) @ g)
    return _result(probs @ v.data, (q, k, v), backward)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean negative log-likelihood of `targets` under softmax(logits)."""
    flat = logits.data.reshape(-1, logits.shape[-1])
    tflat = targets.reshape(-1)
    logp = log_softmax(flat)
    n = tflat.shape[0]
    loss = -logp[np.arange(n), tflat].mean()

    def backward(g):
        grad = np.exp(logp)
        grad[np.arange(n), tflat] -= 1.0
        logits._accumulate((grad * (g / n)).reshape(logits.shape))
    return _result(np.asarray(loss, dtype=logits.data.dtype), (logits,), backward)
