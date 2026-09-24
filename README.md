# council-engine

A claim goes in and gets judged by four personas: Believer, Skeptic,
Investor and Judge. They all run on **a transformer written and trained from
scratch in this repo, in Rust**. There's no Claude, no Ollama, no pretrained
weights and no outside AI service. The model starts as random numbers and
learns only from text you give it, or text its research loop collects.

- **Guide:** it's a terminal program (VS Code's terminal works). See
  [HOW_TO_RUN.md](HOW_TO_RUN.md) for setup on Windows, macOS and Linux.
- **Command reference:** [COMMANDS.md](COMMANDS.md) covers every option,
  annotated output, and the settings behind each command.

```bash
council doctor                               # hardware, model size, status
council train --data ~/my_texts --hours 2    # a folder of .txt/.md files
council ask "We should switch to usage-based pricing"
council research --hours 4                   # read Wikipedia, keep training
```

## Read this first: what to expect

- **It's a small language model you train on your own computer.** A few
  hours on a few MB of text gives a model that has picked up your corpus's
  vocabulary and phrasing. It doesn't reason like a chatbot. Its positions
  "in its own words" will often be rough, and that's honest output for its size.
- **The verdicts come from measurable signals, not from the prose.** Each
  persona is scored on how much the claim raises the model's probability of
  phrases like " This is true." or " This is false." (see *How it decides*).
  Those numbers mean something even when the generated text doesn't. How
  much they mean depends entirely on what the model has read.
- **It says "I don't know" readily, by design.** Confidence is capped by how
  familiar the claim is to the model and by how much it has trained, and a
  repeat run with dropout on must agree before anything scores above Low.
- **It keeps improving the longer it trains.** `train` and `research` update
  the model's actual weights. That is real self-training.

## Speed

Everything is hand-written for speed:

- **Multithreaded, SIMD math:** matrix multiplies use AVX2 / AVX-512 / NEON,
  whichever the CPU has, through the `gemm` crate (plain math, no AI).
- **Attention as matrix multiplies:** attention runs as strided matrix
  multiplies, with no copying.
- **Fused, parallel operations:** normalization, activations and the loss
  are each a single parallel pass over memory.
- **No allocation during training:** buffers are set up once, before the
  first step.
- **Caching when answering:** `ask` processes each prompt once and scores
  every probe phrase against the cached result in one batch, and text
  generation reuses cached attention state.

Training throughput on a 4-core CPU with AVX-512 (`cargo run --release --example bench`):

| Tier | Model | Tokens / second |
|---|---|---|
| tiny | 4 × 192, 2.2M params | ~7,900 |
| small | 6 × 256, 5.9M params | ~3,700 |
| medium | 8 × 384, 17M params | ~1,500 |
| large | 12 × 512, 42M params | ~680 |

At the same model size (2.2M parameters) that is about 2× the old NumPy version.

## How it's built

**The model** (`src/`), all from scratch:

- `model.rs`: a decoder-only transformer in the style of current small
  language models: RMSNorm, rotary position embeddings, a SwiGLU MLP, tied
  embeddings, and dropout. The forward and backward passes are written by
  hand, and every gradient is checked against finite differences in the tests.
- `kernels.rs`: the math, forward and backward, parallelized.
- `tokenizer.rs`: byte-level BPE trained on your corpus. Any text in any
  language can be encoded.
- `trainer.rs`, `optim.rs`: AdamW, warmup plus cosine learning rate,
  gradient clipping, held-out evaluation (loss and bits per byte), and
  checkpoints; Ctrl-C saves.
- `brain.rs`: inference: batched probe scoring on a shared cached prompt,
  and text generation with cached attention state.
- `hardware.rs`: picks the size of a *new* model from your RAM:

  | Tier | RAM | Layers × width | Context | Params |
  |---|---|---|---|---|
  | tiny | ≤ 8 GB | 4 × 192 | 128 | ~2.2M |
  | small | ≤ 16 GB | 6 × 256 | 256 | ~5.9M |
  | medium | ≤ 32 GB | 8 × 384 | 256 | ~17M |
  | large | more | 12 × 512 | 512 | ~42M |

**The council:**

- `router.rs`: matches money words to decide whether the Investor runs.
- `config/personas/*.yaml`: a small model can't follow written instructions,
  so each persona is a *lead-in* the model continues ("This is true
  because…") plus *probe phrases* it's scored on.
- `knowledge.rs`: a BM25 search index, written from scratch, over passages in
  `data/knowledge_base/`. The top hits go in front of the claim, so they
  change what the model scores and writes. Each persona lists the passages
  that moved the model toward its side.
- `research.rs`: uses the plain Wikipedia search API. Topics are
  `research.seed_topics` plus words that keep coming up in your past
  questions. Fetches are capped per hour, and training fills the time in between.
- `council.rs`: the personas, the Judge's rules, familiarity, and the repeat-run check.

## How it decides

1. **Signal.** For each persona, the signal is the average over its probe
   phrases of `log P(probe | claim) − log P(probe | neutral lead-in)`. This
   is pointwise mutual information, in nats per token. Believer probes
   are "true / correct / right"; Skeptic probes are their negations; the
   Investor compares "will make money" against "will lose money".
2. **Stance.** `margin = Believer − Skeptic`, averaged with the Investor's
   signal when it runs. Above `+mixed_margin` means *yes*, below
   `−mixed_margin` means *no*, and anything between is *undecided*. A flat
   "no" is as available as "yes".
3. **Confidence** starts from the size of the margin, then can only go down:
   - it's capped at the weakest confidence among the personas the verdict
     relied on;
   - it's capped by **familiarity**, the model's loss on the claim compared
     with its typical held-out loss. A claim unlike anything it has read
     can't be rated confident;
   - it's capped at Low until the model has trained on
     `council.min_tokens_trained` tokens.
4. **Repeat run.** The council runs a second time with dropout switched on
   (Monte Carlo dropout), which samples a slightly different network. If
   the stance flips or the confidence tier changes, the result becomes
   "Low confidence: the council didn't agree with itself on a repeat run",
   and both verdicts are shown.

All the thresholds live in `config/settings.yaml` under `council:`.

## Layout

```
Cargo.toml, src/            the program (council) and library
config/settings.yaml        every tunable number
config/personas/*.yaml      lead-ins and probe phrases
examples/bench.rs           training-speed benchmark
data/                       corpus, checkpoints, knowledge base (gitignored)
.github/workflows/          CI (tests + builds on Linux/macOS/Windows) and releases
docs/build-brief.md         the original plan (Claude/Ollama), kept for history
```

`cargo test` runs the whole suite (gradient checks, the tokenizer, training,
inference, the council rules, and the research loop against a fake
Wikipedia) in a few seconds, without internet.
