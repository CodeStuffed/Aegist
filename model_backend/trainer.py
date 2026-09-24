"""
Training: turns the text corpus into a model, and keeps improving it.

The corpus is every .txt / .md file under data/corpus/ - text you import
with `cli.py train --data DIR`, plus articles the research loop collects.
The first run trains a BPE vocabulary and creates a model sized for this
machine; every later run continues from the saved checkpoint.

Checkpoint layout (data/brain/):
    brain.json   config, tokenizer merges, training stats
    model.npz    weights
    optim.npz    AdamW moments, so training resumes smoothly
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import data_dir, load_settings
from model_backend import autograd as ag
from model_backend.hardware_detect import detect_hardware
from model_backend.tokenizer import Tokenizer
from model_backend.transformer import ModelConfig, TransformerLM

TEXT_SUFFIXES = {".txt", ".md"}
EVAL_BATCHES = 8


class TrainingError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def corpus_dir(settings: dict | None = None) -> Path:
    path = data_dir(settings) / "corpus"
    path.mkdir(parents=True, exist_ok=True)
    return path


def brain_dir(settings: dict | None = None) -> Path:
    path = data_dir(settings) / "brain"
    path.mkdir(parents=True, exist_ok=True)
    return path


def corpus_files(settings: dict | None = None) -> list[Path]:
    root = corpus_dir(settings)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)


def import_texts(src: str | Path, settings: dict | None = None) -> int:
    """Copy .txt/.md files from `src` (a file or folder) into the corpus."""
    src = Path(src).expanduser()
    if not src.exists():
        raise TrainingError(f"{src} doesn't exist")
    dest_root = corpus_dir(settings) / "imported"
    if src.is_file():
        pairs = [(src, dest_root / src.name)]
    else:
        pairs = [(f, dest_root / src.name / f.relative_to(src)) for f in sorted(src.rglob("*"))]
    count = 0
    for source, dest in pairs:
        if source.is_file() and source.suffix.lower() in TEXT_SUFFIXES:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            count += 1
    return count


_WRAPPED_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_SPACES = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Unwrap hard-wrapped lines (a single newline is just a space) so the
    model learns sentences, not line lengths. Paragraph breaks survive."""
    text = _WRAPPED_NEWLINE.sub(" ", text.replace("\r\n", "\n"))
    return _SPACES.sub(" ", text).strip()


def _read(path: Path) -> str:
    return normalize_text(path.read_text(encoding="utf-8", errors="replace"))


def read_corpus(settings: dict | None = None) -> str:
    return "\n\n".join(_read(p) for p in corpus_files(settings))


def corpus_tokens(tokenizer: Tokenizer, settings: dict | None = None) -> np.ndarray:
    """Token ids for the whole corpus. Each file is encoded once and cached
    until it changes, so a growing research corpus stays cheap to reload."""
    root = corpus_dir(settings)
    cache_dir = brain_dir(settings) / "token_cache"
    cache_dir.mkdir(exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    merges_hash = hashlib.sha1(json.dumps(tokenizer.merges).encode()).hexdigest()
    if manifest.get("merges_hash") != merges_hash:
        manifest = {"merges_hash": merges_hash, "files": {}}

    separator = tokenizer.encode("\n\n")
    pieces, seen = [], set()
    for path in corpus_files(settings):
        rel = str(path.relative_to(root))
        seen.add(rel)
        stat = path.stat()
        key = [stat.st_size, stat.st_mtime_ns]
        cached = cache_dir / (hashlib.sha1(rel.encode()).hexdigest() + ".npy")
        if manifest["files"].get(rel) == key and cached.exists():
            ids = np.load(cached)
        else:
            ids = np.array(tokenizer.encode(_read(path)), dtype=np.uint16)
            np.save(cached, ids)
            manifest["files"][rel] = key
        pieces += [ids, np.array(separator, dtype=np.uint16)]
    manifest["files"] = {k: v for k, v in manifest["files"].items() if k in seen}
    manifest_path.write_text(json.dumps(manifest))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint16)


class AdamW:
    """Adam with decoupled weight decay (not applied to biases or LayerNorm gains)."""

    def __init__(self, params: dict, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1):
        self.params = params
        self.betas, self.eps, self.weight_decay = betas, eps, weight_decay
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.t = 0

    def step(self, lr: float) -> None:
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1**self.t, 1 - b2**self.t
        for k, p in self.params.items():
            if p.grad is None:
                continue
            m, v = self.m[k], self.v[k]
            m *= b1
            m += (1 - b1) * p.grad
            v *= b2
            v += (1 - b2) * p.grad * p.grad
            if p.data.ndim >= 2:
                p.data -= lr * self.weight_decay * p.data
            p.data -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.grad = None

    def state_arrays(self) -> dict[str, np.ndarray]:
        out = {"t": np.array(self.t)}
        out.update({f"m.{k}": a for k, a in self.m.items()})
        out.update({f"v.{k}": a for k, a in self.v.items()})
        return out

    def load_state_arrays(self, state) -> None:
        for k in self.params:
            if f"m.{k}" in state and state[f"m.{k}"].shape == self.m[k].shape:
                self.m[k], self.v[k] = state[f"m.{k}"], state[f"v.{k}"]
        self.t = int(state["t"])


def clip_grad_norm(params: dict, max_norm: float) -> float:
    grads = [p.grad for p in params.values() if p.grad is not None]
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm > max_norm:
        for g in grads:
            g *= max_norm / (norm + 1e-6)
    return norm


def _atomic_savez(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_checkpoint(model: TransformerLM, tokenizer: Tokenizer, meta: dict,
                    optimizer: AdamW | None = None, settings: dict | None = None) -> None:
    d = brain_dir(settings)
    _atomic_savez(d / "model.npz", model.state_dict())
    if optimizer is not None:
        _atomic_savez(d / "optim.npz", optimizer.state_arrays())
    meta["stats"]["updated_at"] = _now()
    tmp = d / "brain.tmp.json"
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, d / "brain.json")


def load_checkpoint(settings: dict | None = None) -> tuple[TransformerLM, Tokenizer, dict] | None:
    d = brain_dir(settings)
    if not (d / "brain.json").exists():
        return None
    meta = json.loads((d / "brain.json").read_text())
    model = TransformerLM(ModelConfig(**meta["config"]))
    with np.load(d / "model.npz") as z:
        model.load_state_dict({k: z[k] for k in z.files})
    return model, Tokenizer(meta["merges"]), meta


def _new_model(settings: dict, log) -> tuple[TransformerLM, Tokenizer, dict]:
    hw = detect_hardware(settings)
    size = hw["model_size"]
    text = read_corpus(settings)[: settings["training"]["tokenizer_train_chars"]]
    log(f"Creating a new model for the '{hw['tier']}' tier ({hw['ram_gb']} GB RAM).")
    log(f"Learning a {size['vocab_size']}-token vocabulary from the corpus...")
    tokenizer = Tokenizer.train(text, size["vocab_size"])
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=size["block_size"],
                         n_layer=size["n_layer"], n_head=size["n_head"], d_model=size["d_model"],
                         dropout=settings["model"]["dropout"])
    model = TransformerLM(config, seed=int(time.time()) % 2**31)
    now = _now()
    meta = {"config": config.to_dict(), "merges": tokenizer.merges,
            "stats": {"steps": 0, "tokens_seen": 0, "train_seconds": 0.0, "train_loss": None,
                      "val_loss": None, "created_at": now, "updated_at": now, "history": []}}
    log(f"Model: {config.n_layer} layers x {config.d_model} wide, {config.n_head} heads, "
        f"{config.block_size}-token context, {model.num_params / 1e6:.2f}M parameters.")
    return model, tokenizer, meta


def train(*, minutes: float | None = None, steps: int | None = None,
          settings: dict | None = None, log=print, seed: int | None = None) -> dict:
    """Train (or keep training) the model for `minutes` of wall time or
    `steps` optimizer steps, whichever is given. Saves checkpoints as it
    goes. On Ctrl-C it saves and then re-raises KeyboardInterrupt."""
    if minutes is None and steps is None:
        raise ValueError("give minutes or steps")
    settings = settings if settings is not None else load_settings()
    tcfg = settings["training"]
    if not corpus_files(settings):
        raise TrainingError("The corpus is empty. Add text with `python cli.py train --data DIR`, "
                            "or let `python cli.py research` collect some.")

    loaded = load_checkpoint(settings)
    model, tokenizer, meta = loaded if loaded else _new_model(settings, log)
    stats, cfg = meta["stats"], model.config
    optimizer = AdamW(model.params, weight_decay=tcfg["weight_decay"])
    optim_path = brain_dir(settings) / "optim.npz"
    if loaded and optim_path.exists():
        with np.load(optim_path) as z:
            optimizer.load_state_arrays({k: z[k] for k in z.files})

    tokens = corpus_tokens(tokenizer, settings)
    T, B = cfg.block_size, tcfg["batch_size"]
    if len(tokens) < max(tcfg["min_corpus_tokens"], 4 * (T + 2)):
        raise TrainingError(f"The corpus is only {len(tokens):,} tokens; add more text first "
                            f"(need at least {tcfg['min_corpus_tokens']:,}).")
    n_val = max(int(len(tokens) * tcfg["val_fraction"]), T + 2)
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]
    log(f"Corpus: {len(tokens):,} tokens ({len(train_tokens):,} train / {len(val_tokens):,} held out).")

    rng = np.random.default_rng(seed)

    def batch(source: np.ndarray, gen: np.random.Generator):
        starts = gen.integers(0, len(source) - T - 1, B)
        x = np.stack([source[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([source[s + 1:s + T + 1] for s in starts]).astype(np.int64)
        return x, y

    def evaluate() -> float:
        gen = np.random.default_rng(1234)  # same held-out batches every time
        with ag.no_grad():
            return float(np.mean([model.forward(*batch(val_tokens, gen))[1].data
                                  for _ in range(EVAL_BATCHES)]))

    def lr_at(step_in_session: int, elapsed: float) -> float:
        warm = min(1.0, (stats["steps"] + 1) / tcfg["warmup_steps"])
        frac = min(elapsed / (minutes * 60), 1.0) if minutes else step_in_session / steps
        return tcfg["learning_rate"] * warm * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))

    def checkpoint() -> None:
        stats["val_loss"] = evaluate()
        stats["history"] = (stats["history"] + [{"step": stats["steps"], "val_loss": stats["val_loss"],
                                                  "at": _now()}])[-500:]
        save_checkpoint(model, tokenizer, meta, optimizer, settings)

    start = last_log = last_eval = last_save = time.time()
    step_i, loss_ema, interrupted = 0, stats.get("train_loss"), False
    try:
        while True:
            elapsed = time.time() - start
            if (minutes is not None and elapsed >= minutes * 60) or (steps is not None and step_i >= steps):
                break
            _, loss = model.forward(*batch(train_tokens, rng), rng=rng)
            loss.backward()
            clip_grad_norm(model.params, tcfg["grad_clip"])
            lr = lr_at(step_i, elapsed)
            optimizer.step(lr)
            optimizer.zero_grad()

            step_i += 1
            stats["steps"] += 1
            stats["tokens_seen"] += B * T
            loss_ema = float(loss.data) if loss_ema is None else 0.95 * loss_ema + 0.05 * float(loss.data)
            stats["train_loss"] = loss_ema

            now = time.time()
            if now - last_eval >= tcfg["eval_every_s"]:
                stats["val_loss"], last_eval = evaluate(), now
            if now - last_log >= tcfg["log_every_s"]:
                rate = step_i * B * T / (now - start)
                val = f"{stats['val_loss']:.3f}" if stats["val_loss"] is not None else "-"
                log(f"step {stats['steps']:>6} | loss {loss_ema:.3f} | held-out {val} | "
                    f"{rate:,.0f} tok/s | lr {lr:.1e}")
                last_log = now
            if now - last_save >= tcfg["checkpoint_every_s"]:
                stats["train_seconds"] += now - last_save
                checkpoint()
                last_save = now
    except KeyboardInterrupt:
        interrupted = True

    stats["train_seconds"] += time.time() - last_save
    checkpoint()
    log(f"Saved: {stats['steps']:,} steps total, {stats['tokens_seen'] / 1e6:.1f}M tokens seen, "
        f"held-out loss {stats['val_loss']:.3f}.")
    if interrupted:
        raise KeyboardInterrupt
    return {"steps_this_session": step_i, "tokens_this_session": step_i * B * T, **stats}
