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
- **Have an NVIDIA GPU?** [GPU_TRAINING.md](GPU_TRAINING.md) is the plan for
  training on it (an RTX 5080 for 4 days on all of Wikipedia), with honest
  expectations.

```bash
council doctor                               # hardware, model size, status
council train --data ~/my_texts --hours 2    # a folder of .txt/.md files
council ask "We should switch to usage-based pricing"
council research --hours 4                   # read Wikipedia, keep training
council import-wikipedia enwiki-latest-pages-articles-multistream.xml.bz2
council gpu-check                            # test an NVIDIA GPU for training
council eval                                 # how often it tells true from false
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
- **The model's size follows your training time.** A new model gets the
  size that will be smartest when the time is up: the biggest that can still
  read ~20 tokens of text per parameter. Bigger isn't smarter if it can't
  finish learning.
- **It looks things up.** The knowledge base can hold all of Wikipedia
  on disk. Each question gets the most relevant passages plus the passages
  linked to them (like following links in Obsidian), as much as fits in
  the model's context.

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

Measured on a 4-core CPU with AVX-512 (16 GB RAM):

| Same 11 MB of text, 10 minutes each | Speed | Held-out bits per byte (lower is better) |
|---|---|---|
| Old Python/NumPy version, 2.2M params | 3,990 tokens/s | 1.900 |
| This version, same 2.2M size | **9,227 tokens/s (2.3×)** | **1.527 (20% better)** |
| This version, default 5.9M size | 3,600 tokens/s | 1.652 |

A full `council ask` (panel, Judge, repeat run, 3 knowledge-base passages)
on the 5.9M model takes about **0.3 seconds** with int8 weights.

Training speed per size, from `cargo run --release --example bench`:

| Tier | Model | Tokens / second |
|---|---|---|
| tiny | 4 × 192, 2.2M params | ~8,400 |
| small | 6 × 256, 5.9M params | ~3,600 |
| medium | 8 × 384, 17M params | ~1,600 |
| large | 12 × 512, 42M params | ~680 |
| xl | 16 × 768, 126M params | ~260 |

## Bigger models, and quantization

Two switches:

- `council train --tier NAME` picks the model size yourself. Otherwise
  `train --hours N` (or `--plan-hours N` for several sessions) picks the
  size that ends up smartest in that time on your CPU or GPU. The sizes
  run from `tiny` (2.2M params) through `xl` (~126M), the GPU sizes `110m`,
  `235m`, `xxl` (~337M) and `730m`, up to `1b` (~1.28B). `council doctor`
  (CPU) and `council gpu-check` (GPU) list every size with its memory and
  how long it takes to train well, and `train` refuses a size that doesn't
  fit.
- `council ask --precision f32|int8|int4` (default `int8`, set in
  `inference.precision`) picks the weights used to answer:
  - `int8` is 4× smaller than f32 with practically identical answers;
  - `int4` is about 6× smaller, at a small accuracy cost.

  Both have hand-written integer kernels.

**What quantization does, measured** on a trained 5.9M model
(`cargo run --release --example quant_eval` runs this on yours):

| Precision | Weights | Held-out bits per byte | Generation speed |
|---|---|---|---|
| f32 | 23.5 MB | 1.3573 | ~800 tokens/s |
| int8 | 6.0 MB | 1.3574 | ~1,650 tokens/s |
| int4 | 3.7 MB | 1.3576 | ~1,250 tokens/s |

**Why there's no "trillion" setting.** Training takes about
6 × parameters × tokens operations, and a model needs about 20 tokens of text
per parameter to become good. These are `council doctor`'s estimates for the
same 4-core PC:

| Size | Memory to train | Time to train well |
|---|---|---|
| tiny, 2.2M | 0.4 GB | ~1 hour |
| small, 5.9M | 0.8 GB | ~8 hours |
| medium, 17M | 1.6 GB | ~3 days |
| large, 42M | 3.4 GB | ~18 days |
| xl, 126M | 7.4 GB | ~5 months |
| xxl, 337M | 19 GB | ~3 years |
| 1b, 1.28B | 40 GB | ~44 years |
| 1 trillion | ~16,000 GB | ~27 million years |

Quantization shrinks a *trained* model (the 1b tier is 800 MB at int4); it
doesn't speed up training, and an untrained giant model is just random
numbers. Billion-parameter models are trained on thousands of GPUs. On a
PC, the best results come from the size `train` picks plus lots of text.
For short sessions a *smaller* size often wins, because it reads more text
in the same time.

**On an NVIDIA GPU** training should run on the order of 100 times faster
than on the 4-core CPU above. That's an estimate from the hardware's speed;
`council gpu-check` measures yours. It moves the sweet spot to a few hundred
million parameters for a few days of training. See
[GPU_TRAINING.md](GPU_TRAINING.md).

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
- `gpu/`: the same training step on an NVIDIA GPU. It uses hand-written
  CUDA kernels (`kernels.cu`, compiled at run time with NVRTC) and cuBLAS
  for matrix multiplies, both loaded only if present. It keeps only each
  layer's input during the forward pass and recomputes one layer at a time
  during the backward pass. A self-check against the CPU runs before any
  training. `cargo test --features gpu-emulator` runs the kernels on a CPU
  emulator and checks them against the CPU model.
- `trainer.rs` also sizes a *new* model for its training time (above).
  Without a time budget (`--steps`), `hardware.rs` picks from your RAM:

  | Tier | RAM | Layers × width | Context | Params |
  |---|---|---|---|---|
  | tiny | ≤ 8 GB | 4 × 192 | 128 | ~2.2M |
  | small | ≤ 16 GB | 6 × 256 | 256 | ~5.9M |
  | medium | ≤ 32 GB | 8 × 384 | 256 | ~17M |
  | large | more | 12 × 512 | 512 | ~42M |
- `corpus.rs`: all the training text's tokens in one file on disk that
  training memory-maps, so billions of tokens cost disk space, not RAM.

**The council:**

- `router.rs`: matches money words to decide whether the Investor runs.
- `config/personas/*.yaml`: a small model can't follow written instructions,
  so each persona is a *lead-in* the model continues ("This is true
  because…") plus *probe phrases* it's scored on.
- `knowledge.rs`: a BM25 search index, written from scratch, over passages in
  `data/knowledge_base/`. The index lives on disk in segments that are
  memory-mapped, so millions of passages open instantly (1M passages: 0.1 ms
  to open, ~20 ms per search on 4 cores). Passages are linked like Obsidian notes (to their
  neighbors in the same article, and to passages sharing their rarest
  words). The best matches plus their strongest links go in front of the
  claim, filling the model's context window with only what's relevant, so
  they change what the model scores and writes. Each persona lists the
  passages that moved the model toward its side.
- `wikipedia.rs`: imports a whole Wikipedia dump: unpacks the multistream
  `.bz2` on every core, turns wiki markup into plain paragraphs, and fills
  both the corpus and the knowledge base.
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
2. **Negation test.** The claim's verb is flipped ("prices will rise" →
   "prices will not rise"). With the knowledge-base evidence in front, the
   model scores how likely the rest of the claim is after each version. The
   cost of the word "not" itself is left out, so this measures which version
   fits what the model has read. It's the strongest signal a model trained
   on real text like Wikipedia has, and it's skipped for claims without a
   verb it can flip ("is", "will", "can", "has"...).
3. **Stance.** `margin = Believer − Skeptic`, averaged with the Investor's
   signal when it runs, then blended with the negation test
   (`council.negation_weight`, default half). Above `+mixed_margin` means
   *yes*, below `−mixed_margin` means *no*, and anything between is
   *undecided*. A flat "no" is as available as "yes".
4. **Confidence** starts from the size of the margin, then can only go down:
   - it's capped at the weakest confidence among the personas (and the
     negation test) the verdict relied on;
   - it's capped by **familiarity**, the model's loss on the claim compared
     with its typical held-out loss. A claim unlike anything it has read
     can't be rated confident;
   - it's capped at Low until the model has trained on
     `council.min_tokens_trained` tokens.
5. **Repeat run.** The council runs a second time with dropout switched on
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
src/gpu/                    GPU training: CUDA kernels, cuBLAS/NVRTC bindings, CPU emulator
examples/bench.rs           training speed per size on this machine
examples/quant_eval.rs      what f32 / int8 / int4 cost and buy on your trained model
examples/kb_bench.rs        knowledge-base speed at a million passages
examples/tok_speed.rs       tokenizer training and encoding speed
GPU_TRAINING.md             training on an NVIDIA GPU, and what to expect
data/                       corpus, checkpoints, knowledge base (gitignored)
.github/workflows/          CI (tests + builds on Linux/macOS/Windows) and releases
docs/build-brief.md         the original plan (Claude/Ollama), kept for history
```

`cargo test` runs the whole suite in a few seconds, without internet:
- gradient checks;
- the tokenizer, checked against textbook BPE;
- training;
- inference;
- the council rules;
- the knowledge base (on disk and in memory);
- the Wikipedia importer, on a small multistream dump;
- the research loop, against a fake Wikipedia.

`cargo test --features gpu-emulator` adds the GPU tests (a few minutes).
