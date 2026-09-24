# council-engine

A claim goes in and gets judged by four personas: Believer, Skeptic,
Investor and Judge. They all run on **a neural network written and trained
from scratch in this repo**. There's no Claude, no Ollama, no pretrained
weights and no outside AI service. The model starts as random numbers
and learns only from text you give it, or text its research loop collects.

## Read this first: what to expect

- **It's a small language model you train on a normal computer.** A few
  hours of training on a few MB of text gives a model that has picked up
  your corpus's vocabulary and phrasing. It doesn't reason. Its positions
  "in its own words" will often be word salad, and that's honest output
  for its size.
- **The verdicts come from measurable signals, not from the prose.** Each
  persona is scored on how much the claim raises the model's probability of
  phrases like " This is true." or " This is false." (see *How it decides*).
  Those numbers mean something even when the generated text doesn't. How
  much they mean depends entirely on what the model has read.
- **It says "I don't know" readily, by design.** Confidence is capped by how
  familiar the claim is to the model and by how much it has trained, and a
  repeat run with dropout on must agree before anything scores above Low.
  An undertrained model answers Low to everything, and that's the correct answer.
- **It keeps improving the longer it trains.** `train` and `research` update
  the model's actual weights. That is real self-training, and it's the
  point of building from scratch.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py doctor                               # hardware, model size, status
python cli.py train --data ~/my_texts --hours 2    # a folder of .txt/.md files
python cli.py ask "We should switch to usage-based pricing"
python cli.py research --hours 4                   # read Wikipedia, keep training
```

Plan on **several MB of text** at least. Books, articles, your own notes,
anything in plain text. A new model won't start from less than 200 KB,
because its vocabulary is learned once, from whatever text is there at the start. With a small corpus the model memorizes it:
training loss keeps falling while the held-out loss stalls, and `train`
prints both so you can see it happen. `research` grows the corpus from
Wikipedia on its own if you don't have text handy.

## Commands

| Command | What it does |
|---|---|
| `ask "<claim>"` | Full council run: each persona's signal, confidence and position, the Judge's verdict, the repeat-run check, and overall confidence. `--json` for the raw dict, `--no-recheck` to skip the repeat run. |
| `train --hours N` / `--steps N` | Train, or keep training, on everything in `data/corpus/`. `--data DIR` copies a folder of text in first. Ctrl-C saves and stops. |
| `research --hours N` | Loops until time's up: pick a topic, fetch Wikipedia articles, store them, then train on the grown corpus until the next fetch. |
| `doctor` | RAM/CPU and model tier, corpus size, training progress, knowledge-base size. |

From other code, call `evaluate(claim) -> dict` in `engine/orchestrator.py`.
The CLI is a thin wrapper around it.

## How it's built

**The model** (`model_backend/`), all from scratch on NumPy:

- `autograd.py`: reverse-mode automatic differentiation. Every op has a
  hand-derived gradient, and each is checked against finite differences in
  `tests/test_autograd.py`.
- `tokenizer.py`: byte-level BPE, trained on your corpus. Any text can be
  encoded.
- `transformer.py`: a GPT-style decoder with causal self-attention, a GELU
  MLP, pre-LayerNorm and dropout. These are the "hidden layers".
- `trainer.py`: hand-written AdamW, warmup plus cosine learning rate,
  gradient clipping, held-out evaluation, and checkpoints in `data/brain/`.
- `hardware_detect.py`: picks the size of a *new* model from your RAM:

  | Tier | RAM | Layers × width | Context | Params | Speed (4-core laptop) |
  |---|---|---|---|---|---|
  | tiny | ≤ 8 GB | 2 × 128 | 128 | ~0.5M | ~20M tokens/hour |
  | small | ≤ 16 GB | 4 × 192 | 128 | ~2.2M | ~12M tokens/hour |
  | medium | ≤ 32 GB | 6 × 256 | 192 | ~5.8M | ~5M tokens/hour |
  | large | more | 8 × 384 | 256 | ~16M | ~2.5M tokens/hour |

  Edit `model.tiers` in `config/settings.yaml` to change these. A trained
  model keeps its size; delete `data/brain/` to start over at a new size.

**The council** (`engine/`):

- **Router**: matches keywords from `router.money_keywords` to decide
  whether the Investor runs.
- **Personas** (`config/personas/*.yaml`): a small model can't follow
  written instructions, so each persona is a *lead-in* the model continues
  ("This is true because…") plus *probe phrases* it's scored on.
- **Memory**: a BM25 search index, written from scratch, over passages in
  `data/knowledge_base/`. The top hits go in front of the claim, so they
  change what the model scores and writes. Each persona also lists the
  passages that moved the model toward its side.
- **Research loop**: uses the plain Wikipedia search API. Topics are
  `research.seed_topics` plus words that keep coming up in your past
  `ask` sessions. Fetches are capped at `max_topics_per_hour`, and training
  fills the time in between.

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

All the thresholds live in `config/settings.yaml` under `council:`. They're
starting points; tune them once your model has trained for a while.

## Layout

```
config/settings.yaml         every tunable number
config/personas/*.yaml       lead-ins and probe phrases
model_backend/               the from-scratch model (see above)
engine/orchestrator.py       evaluate(): the public entry point
engine/agents.py             persona scoring and writing
engine/router.py             Investor or not
engine/uncertainty.py        the repeat-run comparison
engine/memory/               knowledge store, session log, research loop
cli.py                       ask / train / research / doctor
data/                        corpus, checkpoints, knowledge base (gitignored)
tests/                       pytest suite; runs in about 30 s, no network
```

`docs/build-brief.md` is the original plan, which used Claude and Ollama.
It's kept for history. This README describes what the code actually does.
