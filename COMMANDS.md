# Command reference

The full guide to every command: what it does, every option, what the output
means, and the settings that change it. For setup (downloading or building
`council`, using VS Code's terminal), see [HOW_TO_RUN.md](HOW_TO_RUN.md).

```bash
council <command> [options]
```

| Command | In one line | Needs a trained model? | Needs internet? |
|---|---|---|---|
| [`doctor`](#doctor) | Status check | No | No |
| [`train`](#train) | Teach the model from text | No (creates one) | No |
| [`research`](#research) | Read Wikipedia and train itself | No (creates one) | Yes |
| [`ask`](#ask) | Put a claim in front of the council | **Yes** | No |

- `council --help` lists the commands; `council <command> --help` lists one
  command's options.
- `council --version` prints the version.
- The global option `--threads N` sets how many CPU threads to use. The
  default is every core for `train`/`research`, and up to 2 for `ask`:
  generating one token at a time is fastest on 1–2 threads, and it leaves
  cores free for a training run. You can also set it
  with the `COUNCIL_THREADS` environment variable.

---

## `doctor`

**Shows the state of everything. It never changes anything, so run it
whenever you're curious.**

```bash
council doctor
```

### Example output (a real run on a 4-core laptop-class CPU)

```
Files
  Settings   /home/you/Aegist/config/settings.yaml
  Data       /home/you/Aegist/data
Hardware
  RAM        15.7 GB
  CPU        4 cores, 4 threads, AVX-512
  Tier       small -> a new model would be 6 layers x 256 wide, 256-token context (~5.9M params)
Sizes (council train --tier NAME; * = picked automatically; time assumes 140 GFLOP/s)
   tiny       2.2M params |    0.4 GB to train | ~  1.1 hours to train well | int4 file    1 MB
  *small      5.9M params |    0.8 GB to train | ~  8.2 hours to train well | int4 file    4 MB
   medium    17.3M params |    1.6 GB to train | ~     3 days to train well | int4 file   11 MB
   large     42.2M params |    3.4 GB to train | ~    18 days to train well | int4 file   26 MB
   xl       125.9M params |    7.4 GB to train | ~   158 days to train well | int4 file   79 MB
   xxl      337.2M params |   18.9 GB to train | ~    3 years to train well | int4 file  211 MB  (too big for this machine)
   1b        1.28B params |   40.3 GB to train | ~   44 years to train well | int4 file  798 MB  (too big for this machine)
Corpus
  152 file(s), 10.5 MB in /home/you/Aegist/data/corpus
Model
  6x256 transformer, 5.87M params, vocabulary 4096
  Answers with int8 weights (inference.precision): f32 23 MB | int8 ~6 MB | int4 ~4 MB
  544 steps, 2.2M tokens seen, 0.2 h trained, held-out loss 3.563 (1.652 bits per byte)
  Trained on 2% of the ~117M tokens (20 per parameter) a model this size should see
Memory
  228 passage(s) in the knowledge base
  9 past council session(s)
```

### Reading it

| Line | Meaning |
|---|---|
| **Settings / Data** | Where your settings file and everything it has learned are stored. |
| **Sizes** | Every model size in settings.yaml: parameters, memory needed to train it, and how long it takes on this machine to read ~20 tokens per parameter (a rough rule for "trained well"). `*` marks the size picked automatically for a new model; others need `council train --tier NAME`. |
| **CPU** | Cores, threads, and the fastest vector instructions detected (AVX-512 > AVX2 + FMA > SSE; NEON on Apple Silicon). All are used automatically. |
| **Tier** | The model size your RAM supports. It only applies when a **new** model is created; an existing model keeps its size. |
| **Corpus** | All the text the model trains on: your imported files plus Wikipedia articles from `research`. |
| **params** | Number of learned numbers ("weights") in the model. |
| **Answers with** | The precision `ask` uses by default, and the model's size at each precision. |
| **Trained on N%** | How far along it is toward ~20 tokens per parameter. |
| **steps / tokens seen** | How many learning updates it has made, and how much text it has read in total, counting re-reads. A token is roughly ¾ of a word. |
| **held-out loss** | How well the model predicts text it has *never trained on*, in its own units. Lower is better. |
| **bits per byte** | **The number to watch.** The same measure per byte of text, so it compares fairly across tokenizers and model sizes. Plain compression (like zip) manages about 2–3; lower means the model has learned the language better. |
| **passages** | Paragraphs it can look up while answering. |
| **past council sessions** | Claims you've asked about. `research` reads these to choose topics. |

If it says `Model: none yet`, you haven't trained yet. Run `train` or `research`.

---

## `train`

**Teaches the model from text. The first run creates the model; every later
run keeps improving the same one.**

```bash
council train [--data PATH] [--hours N | --steps N]
```

| Option | Default | What it does |
|---|---|---|
| `--data PATH` | none | A folder (searched recursively) or a single file. Every `.txt` and `.md` file is **copied** into `data/corpus/imported/`, then training starts. You only need to import a folder once; re-importing just refreshes the copies. Other file types (PDF, Word…) are skipped, so save them as `.txt` first. |
| `--hours N` | `1` | Train for N hours. Decimals work: `0.25` is 15 minutes. |
| `--steps N` | none | Train for exactly N learning steps instead of a set time. Can't be combined with `--hours`. |
| `--tier NAME` | picked from RAM | Size for a **new** model: `tiny`, `small`, `medium`, `large`, or the opt-in `xl`, `xxl`, `1b`. `council doctor` lists them with memory and time. It refuses a size that won't fit in memory; to resize an existing model, delete `data/brain/` first. |

### Examples

```bash
council train --data ~/Documents/books --hours 2    # import a folder, train 2 hours
council train --data notes.txt --hours 0.5          # import one file
council train --hours 8                             # keep training on what it already has
council train --steps 200                           # quick top-up
```

On Windows, write paths like `--data "C:\Users\you\Documents\books"`, with
quotes if there are spaces.

### What happens, in order

1. **Import:** with `--data`, files are copied into the corpus.
2. **First run only:** it learns a vocabulary (word pieces) from your text,
   then creates a model sized for your RAM. This needs at least **200 KB** of
   text; with less it stops and tells you how much it has.
3. **Tokenizing:** text is turned into numbers, using every core. This
   happens once per file and is cached; only new or changed files are redone.
4. **Training:** it prints progress every 10 seconds, measures held-out loss
   every minute, and saves every 5 minutes.
5. **Done** (or **Ctrl+C**): it saves and prints a summary.

### Example output (real: the start of one run and the end of a 10-minute one, same settings and text)

```
Imported 152 file(s) into /home/you/Aegist/data/corpus.
Creating a new model for the 'small' tier (15.7 GB RAM).
Learning a 4096-token vocabulary from the corpus...
Model: 6 layers x 256 wide, 4 heads, 256-token context, 5.87M parameters.
Corpus: 3,294,382 tokens (3,129,663 train / 164,719 held out).
step      9 | loss 8.302 | held-out - |   3,500 tok/s | lr 7.7e-5
At 3,500 tokens/s, this 5.9M-parameter model needs about 9.3 hours more training to read ~117M tokens (20 per parameter, roughly "trained well").
step     19 | loss 8.114 | held-out - |   3,786 tok/s | lr 7.8e-5
...
step    542 | loss 3.376 | held-out 3.592 |   3,710 tok/s | lr 1.0e-4
Saved: 544 steps total, 2.2M tokens seen, held-out loss 3.563 (1.652 bits per byte).
```

| Column | Meaning |
|---|---|
| `step` | Learning updates so far, across all sessions. |
| `loss` | How wrong it is on the text it's training on. Lower is better. |
| `held-out` | How wrong it is on the 5% of text it never trains on. **This is its real skill.** It shows `-` until the first measurement, a minute in. |
| `tok/s` | Speed: tokens learned from per second. See the table below. |
| `lr` | Learning rate. It ramps up for the first 100 steps, then eases down to 10% by the end of the session. That's automatic; you don't need to touch it. |

### Speed

Measured on a 4-core CPU with AVX-512 (more cores is faster):

| Tier | Model | Tokens / second | To read 20 tokens per parameter |
|---|---|---|---|
| tiny (≤ 8 GB RAM) | 4 × 192, 2.2M params | ~8,400 | ~1 hour |
| small (≤ 16 GB) | 6 × 256, 5.9M params | ~3,600 | ~9 hours |
| medium (≤ 32 GB) | 8 × 384, 17M params | ~1,600 | ~2.5 days |
| large (more) | 12 × 512, 42M params | ~680 | ~2.5 weeks |
| xl (opt-in) | 16 × 768, 126M params | ~260 | ~4 months |

`cargo run --release --example bench` measures your own machine, and
`council doctor` estimates every size.

**Short on time? Pick a smaller size.** In a 10-minute test on the same
text, the tiny size scored better than the small one, because it read 2.5×
more text in that time. Bigger sizes only pull ahead after hours of training.

The first progress line is followed by one estimate, based on the speed it
just measured: how much longer this model needs to have read about 20 tokens
per parameter (a rough rule for "trained well").

### Is it working?

- **Both numbers falling:** it's learning. Keep going.
- **`loss` falling but `held-out` stuck:** it's memorizing your text instead
  of learning general patterns. It needs **more text**; more time won't help.
- **Both stuck:** it has learned what it can at this size. Add more text, or
  delete `data/brain/` and let it start over at a bigger tier (edit
  `model.tiers` in `config/settings.yaml`).

### How much text and time?

- **Text:** at least 200 KB to start a model, several MB to do well, and
  more is better. If held-out stops improving while training loss keeps
  falling, it needs more text, not more time.
- **Time:** see the table above. A few hours on `tiny` or `small` gives
  real progress; `research` overnight keeps it going on its own.

### Stopping and resuming

Press **Ctrl+C** at any time. It finishes the current step, saves, and prints
`Stopped; the checkpoint was saved.` Press Ctrl+C a second time to quit
immediately without saving. The next `train` picks up exactly where it left off.

### Settings that change it (`config/settings.yaml` → `training:`)

| Setting | Default | Effect |
|---|---|---|
| `tokens_per_step` | 4096 | Text per learning step (batch size × context length). |
| `learning_rate` | 0.001 | Peak learning speed. Too high and the loss jumps around; too low and it's slow. |
| `val_fraction` | 0.05 | Share of text held out to measure `held-out`. |
| `checkpoint_every_s` | 300 | How often it saves, in seconds. |
| `min_new_model_chars` | 200000 | Minimum text needed to create a new model. |

---

## `research`

**Runs on its own: reads Wikipedia, stores what it finds, and trains itself on it.**

```bash
council research [--hours N]
```

| Option | Default | What it does |
|---|---|---|
| `--hours N` | `1` | How long to run. Decimals work. Leave it running overnight with `--hours 8`. |

### What it does, on repeat

1. **Picks a topic:**
   - first the `seed_topics` in `config/settings.yaml`,
   - then words that keep coming up in your `ask` questions, most frequent first,
   - skipping anything it already read in the last week (`refresh_hours`).
2. **Reads Wikipedia:** it searches for the topic and downloads the top 2
   articles as plain text, using Wikipedia's ordinary search, not an AI.
3. **Stores them:**
   - each paragraph goes into the knowledge base (skipping reference lists
     and headings), so `ask` can look it up;
   - the full text goes into `data/corpus/research/` for training.
4. **Trains** for 5 minutes on everything it has.
5. Repeats. It reads at most **6 topics per hour**, to be polite to
   Wikipedia; the rest of the time it just keeps training.

### Example output (illustrative)

```
Researching and self-training for 4 hour(s). Ctrl-C stops it; progress is saved as it goes.
Researched 'pricing strategy': 41 new passages (Pricing strategies, Price discrimination).
Not training yet: Only 96 KB of text so far; a new model needs at least 200 KB to learn its vocabulary from.
Researched 'unit economics': 18 new passages (Unit economics, Contribution margin).
Researched 'business model': 37 new passages (Business model, Business model canvas).
Creating a new model for the 'small' tier (15.7 GB RAM).
...
step    300 | loss 5.412 | held-out 5.630 |   3,480 tok/s | lr 8.1e-4
...
Done: 24 topic(s), 812 new passages, 214 min of training.
```

On a fresh install it reads 2–3 topics before it has enough text to create
the model. That's normal.

### If something goes wrong

- **No internet:** it prints `Couldn't fetch '...' from Wikipedia`, keeps
  training on what it has, and retries the topic later.
- **Behind a proxy:** it uses `HTTPS_PROXY` and respects `NO_PROXY`, and it
  trusts your operating system's certificates.
- **Ctrl+C:** stops cleanly, and everything read and learned so far is saved.
  Running it again continues, remembering which topics it has read and how
  many it fetched this hour.

### Steering what it reads

Edit `seed_topics` in `config/settings.yaml`. They're read first, in order:

```yaml
research:
  seed_topics: [pricing strategy, customer retention, SaaS, venture capital]
```

The more you `ask` about a subject, the more it researches it.

### Settings (`config/settings.yaml` → `research:`)

| Setting | Default | Effect |
|---|---|---|
| `seed_topics` | 8 business/reasoning topics | Topics to read first. |
| `articles_per_topic` | 2 | Wikipedia articles per topic. |
| `max_topics_per_hour` | 6 | Politeness cap on Wikipedia requests. |
| `train_minutes_per_topic` | 5 | Training time between topics. |
| `refresh_hours` | 168 | Don't re-read a topic within this many hours (a week). |
| `wikipedia_api` | English Wikipedia | Change `en` to e.g. `de` or `es` for another language's Wikipedia. |

---

## `ask`

**Puts a claim in front of the council and prints the verdict.**

```bash
council ask "<your claim>" [--no-recheck] [--json]
```

| Option | What it does |
|---|---|
| `"<claim>"` | The statement to judge. **Use quotes.** A full sentence works best. |
| `--no-recheck` | Skip the repeat run. About twice as fast, but it can't catch the council disagreeing with itself. |
| `--json` | Print the complete result as JSON instead of the readable report, for other programs or for inspecting every number. |
| `--precision P` | Weights used to answer: `f32` (exact), `int8` (default: 4× smaller, practically identical, fastest) or `int4` (~6× smaller, practically identical). The default is set by `inference.precision` in settings. Training always uses full precision. |

It takes a second or two, and works fine while `train` or `research` runs in
another terminal. Progress lines (`... Council deliberating`) go to stderr,
so `--json` output stays clean.

### Example output, annotated (illustrative numbers)

```
CLAIM: Light is not refracted when it passes into glass
Model: 6x256 transformer, 5.87M params, 18.4M tokens trained, held-out loss 3.41   <- (1)
Router: no Investor - No money-related words, so no Investor.                      <- (2)
Familiarity: claim loss 4.71 vs. typical 3.41 -> confidence ceiling Medium        <- (3)
Knowledge base: 3 relevant passage(s)                                              <- (4)

BELIEVER  [Low]  signal -0.57                                                      <- (5)
  "This is true because in the two Prisms ... refracted in the first Prism ..."    <- (6)
  - +0.07 Much after the same manner, if ACBD ... (Opticks)                        <- (7)

SKEPTIC  [Medium]  signal +0.23
  "This is false because the Sun's Light is also of these Sides ..."

JUDGE  [Medium, capped from High]  relied on: skeptic                              <- (8)
  Verdict: No - on what this model has read, the claim doesn't hold up.            <- (9)
  Why: Believer signal -0.57 vs. Skeptic +0.23 nats/token (margin -0.80).          <- (10)
  In its own words: "In conclusion, which comes through the Sun ..."
  Unresolved: the claim is unlike most of what the model has read ...              <- (11)

Repeat run (dropout on) agreed with the first.                                     <- (12)
OVERALL CONFIDENCE: Medium                                                         <- (13)
```

1. **Model:** which model answered and how trained it is.
2. **Router:** whether the **Investor** joins. It does when the claim
   contains a money word (price, customers, revenue, `$`…; the list is
   `router.money_keywords` in settings).
3. **Familiarity:** how surprising the claim's wording is to the model,
   compared with its typical held-out loss. The further above typical, the
   lower the **confidence ceiling**:
   - more than 1.25× typical: at most Medium;
   - more than 1.5× typical: Low.

   Also, until the model has read 2M tokens, everything is capped at Low.
4. **Knowledge base:** how many stored passages matched the claim well
   enough to be used. They're placed in front of the claim when the model
   reads it.
5. **Signal:** how much the claim makes the model expect "This is true."
   (Believer) or "This is false." (Skeptic), measured in nats per token.
   Positive means the claim pushes toward that side; **around 0.5 or more is
   strong**. The Investor compares "will make money" against "will lose money".
6. **The quote:** the persona's position, written by the model itself.
   Expect rough text from a small model; the signal is what counts.
7. **Evidence:** stored passages that pushed the model toward this persona's
   side, with how much each one moved the signal.
8. **Judge confidence:** "capped from High" means its own estimate was
   higher, but a rule lowered it. **Relied on** is whose case the verdict
   rests on; the Judge can never be more confident than those personas.
9. **Verdict:** **Yes**, **No**, or **Undecided**. "No" is a normal answer,
   not an error.
10. **Why:** the numbers behind the verdict. **Margin** = Believer − Skeptic
    (averaged with the Investor when present):
    - above +0.1 is Yes;
    - below −0.1 is No;
    - in between is Undecided.
11. **Unresolved:** what's holding confidence down.
12. **Repeat run:** the council answers a second time with a bit of
    randomness in the network (dropout on). A real signal survives that;
    noise doesn't. If the answers disagree, both are shown, and the result
    reads `Low confidence: the council didn't agree with itself on a repeat run`.
13. **Overall confidence:** the final word. **High** is only possible when
    the signals are strong, the claim is familiar, the model is well
    trained, and both runs agree.

### A real run

The 5.9M model from the examples above (10 minutes of training on 11 MB of
technical text), with 228 paragraphs of Newton's *Opticks* in its knowledge
base. It answers "Undecided" with Low confidence, which is correct for a
model that has barely trained and has read little like this claim. Took
0.3 seconds:

```
CLAIM: White light is a mixture of rays of different colours
Model: 6x256 transformer, 5.87M params, int8 weights (6.0 MB), 2.2M tokens trained, held-out loss 3.56
Router: no Investor - No money-related words, so no Investor.
Familiarity: claim loss 6.38 vs. typical 3.56 -> confidence ceiling Low
Knowledge base: 3 relevant passage(s)

BELIEVER  [Low]  signal -1.14
  "This is true because of the Lens, that they are so that the Rays of the Rays, and
  that these Prism, and this distance of that those were used, that be, and the
  Refractions at _FC_ I, a FE, and that"

SKEPTIC  [Low]  signal -1.16
  "This is false because the Prism in the Refraction was the violet, as the blue, the
  Glass is so in the Paper, and to that the Angle of the same as in the Air, and the
  violet, so in the Eye, a red, and the"

JUDGE  [Low]  relied on: believer, skeptic
  Verdict: Undecided - the model finds the claim about as compatible with 'true' as with
  'false'.
  Why: Believer signal -1.14 vs. Skeptic -1.16 nats/token (margin +0.02).
  In its own words: "In conclusion, and therefore the Glass being made by the red and
  the Paper which part must be in those of the Object-line of the Sines, that the Rings
  made about the other Colours of those Colours are the Light, or very many as by the"
  Unresolved: neither side's signal clearly beats the other; the claim is unlike most of
  what the model has read (loss 6.38 vs. a typical 3.56)

Repeat run (dropout on) agreed with the first.
OVERALL CONFIDENCE: Low
```

### Getting better answers

- **Train on text like your claims.** A model trained on business articles
  handles business claims; one trained on physics won't.
- **Phrase claims as plain statements:** "Raising prices will reduce sales",
  not "should I maybe raise prices??".
- **Use `research` on your subject.** It adds passages the council can cite.
- **Adjust the personas' probe phrases** (`config/personas/*.yaml`) to
  wording your text actually uses. See `config/personas/README.md`.

### Settings (`config/settings.yaml` → `council:`, `memory:`, `uncertainty:`)

| Setting | Default | Effect |
|---|---|---|
| `council.confidence_thresholds` | high 0.5, medium 0.15 | Signal strength needed for each level. |
| `council.mixed_margin` | 0.1 | How close the sides must be to count as Undecided. |
| `council.familiarity` | 1.25 / 1.5 | How unfamiliar a claim can be before confidence is capped. |
| `council.min_tokens_trained` | 2,000,000 | Below this, everything is Low. |
| `council.generate.temperature` | 0.8 | Randomness of the written positions. Lower is more repetitive, higher is wilder. |
| `memory.top_k` | 3 | Maximum passages used per question. |
| `memory.min_relevance` | 0.35 | How well a passage must match to be used. |
| `uncertainty.enabled` | true | Whether the repeat run happens by default. |

---

## Common workflows

**First day:**

```bash
council doctor
council research --hours 2
council ask "Customers prefer simple pricing"
```

**With your own documents:**

```bash
council train --data ~/my_documents --hours 3
council ask "..."
```

**Let it grow overnight:**

```bash
council research --hours 8
```

**Train and ask at the same time:** run `research` in one terminal, and `ask`
in a second one (VS Code: the **+** button in the terminal panel).

**Start over:** delete `data/brain/` (model only) or all of `data/`
(everything), then train again.

---

## Using it from other programs

Any language can run `council ask --json "..."` and read the result:

```json
{
  "claim": "...",
  "judge": {"verdict": "No - ...", "stance": "no", "margin": -0.8, "confidence": "Medium", "relied_on": ["skeptic"], ...},
  "panel": [{"persona": "believer", "signal": -0.57, "confidence": "Low", "position": "...", "key_points": [...]}, ...],
  "overall_confidence": "Medium",
  "consistency": {"agreed": true, "reasons": [], "rerun_judge": {...}},
  "familiarity": {...}, "memory": [...], "routing": {...}, "model": {...}
}
```

In Rust, depend on this crate and call `council::council::evaluate(...)`:
the same function `council ask` uses.
