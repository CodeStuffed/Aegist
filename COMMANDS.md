# Command reference

The full guide to every command: what it does, every option, what the output
means, and the settings that change it. For first-time setup (installing
Python, VS Code, the virtual environment), see [HOW_TO_RUN.md](HOW_TO_RUN.md).

Every command is typed in a terminal, inside the project folder, with the
environment activated:

```bash
python cli.py <command> [options]
```

| Command | In one line | Needs a trained model? | Needs internet? |
|---|---|---|---|
| [`doctor`](#doctor) | Status check | No | No |
| [`train`](#train) | Teach the model from text | No (creates one) | No |
| [`research`](#research) | Read Wikipedia and train itself | No (creates one) | Yes |
| [`ask`](#ask) | Put a claim in front of the council | **Yes** | No |

`python cli.py --help` lists the commands; `python cli.py <command> --help`
lists one command's options.

---

## `doctor`

**Shows the state of everything. It never changes anything, so run it
whenever you're curious.**

```bash
python cli.py doctor
```

No options.

### Example output

```
Hardware
  RAM        15.7 GB
  CPU cores  4
  Tier       small -> a new model would be 4 layers x 192 wide, 128-token context
Corpus
  1 file(s), 0.6 MB in /home/you/Aegist/data/corpus
Model
  4x192 transformer, 2.20M params, vocabulary 2048
  2,289 steps, 4.7M tokens seen, 0.5 h trained, held-out loss 4.544
Memory
  228 passage(s) in the knowledge base
  3 past council session(s)
```

### Reading it

| Line | Meaning |
|---|---|
| **Tier** | The model size your RAM supports: tiny, small, medium or large. It only applies when a **new** model is created; an existing model keeps its size. |
| **Corpus** | All the text the model trains on: your imported files plus Wikipedia articles from `research`. |
| **params** | Number of learned numbers ("weights") in the model. More is smarter but slower to train. |
| **steps** | How many learning updates it has made so far. |
| **tokens seen** | How much text it has read in total, counting re-reads. A token is roughly ¾ of a word. |
| **held-out loss** | **The number to watch.** It measures how well the model predicts text it has *never trained on*. Lower is better. It starts around 7.6 (random guessing); 4–5 is early, 3–3.5 is solid for this model size. |
| **passages in the knowledge base** | Paragraphs it can look up while answering. |
| **past council sessions** | Claims you've asked about. `research` reads these to choose topics. |

If it says `Model: none yet`, you haven't trained yet. Run `train` or `research`.

---

## `train`

**Teaches the model from text. The first run creates the model; every later
run keeps improving the same one.**

```bash
python cli.py train [--data PATH] [--hours N | --steps N]
```

| Option | Default | What it does |
|---|---|---|
| `--data PATH` | none | A folder (searched recursively) or a single file. Every `.txt` and `.md` file is **copied** into `data/corpus/imported/`, then training starts. You only need to import a folder once; re-importing just refreshes the copies. Other file types (PDF, Word…) are skipped, so save them as `.txt` first. |
| `--hours N` | `1` | Train for N hours. Decimals work: `0.25` is 15 minutes. |
| `--steps N` | none | Train for exactly N learning steps instead of a set time. Can't be combined with `--hours`. |

### Examples

```bash
python cli.py train --data ~/Documents/books --hours 2    # import a folder, train 2 hours
python cli.py train --data notes.txt --hours 0.5          # import one file
python cli.py train --hours 8                             # keep training on what it already has
python cli.py train --steps 200                           # quick top-up
```

On Windows, write paths like `--data "C:\Users\you\Documents\books"`, with
quotes if there are spaces.

### What happens, in order

1. **Import:** with `--data`, files are copied into the corpus.
2. **First run only:** it learns a vocabulary (word pieces) from your text,
   then creates a model sized for your RAM. This needs at least **200 KB** of
   text; with less it stops and tells you how much it has.
3. **Tokenizing:** text is converted to numbers. This is slow the first time
   and cached after that, so only new or changed files are redone.
4. **Training:** it prints progress every 10 seconds, measures held-out loss
   every minute, and saves every 5 minutes.
5. **Done** (or **Ctrl+C**): it saves and prints a summary.

### Example output

```
Imported 3 file(s) into /home/you/Aegist/data/corpus.
Creating a new model for the 'small' tier (15.7 GB RAM).
Learning a 2048-token vocabulary from the corpus...
Model: 4 layers x 192 wide, 6 heads, 128-token context, 2.20M parameters.
Corpus: 143,639 tokens (136,458 train / 7,181 held out).
step     53 | loss 6.556 | held-out 6.206 | 3,606 tok/s | lr 4.6e-04
step    420 | loss 4.005 | held-out 4.782 | 3,500 tok/s | lr 9.3e-04
...
Saved: 2,289 steps total, 4.7M tokens seen, held-out loss 4.544.
```

| Column | Meaning |
|---|---|
| `step` | Learning updates so far, across all sessions. |
| `loss` | How wrong it is on the text it's training on. Lower is better. |
| `held-out` | How wrong it is on the 5% of text it never trains on. **This is its real skill.** It shows `-` until the first measurement, a minute in. |
| `tok/s` | Speed. On a 4-core laptop: roughly 5,000 (tiny), 3,500 (small), 1,400 (medium), 700 (large). |
| `lr` | Learning rate. It ramps up for the first 100 steps, then eases down to 10% by the end of the session. That's automatic; you don't need to touch it. |

### Is it working?

- **Both numbers falling:** it's learning. Keep going.
- **`loss` falling but `held-out` stuck:** it's memorizing your text instead
  of learning general patterns. It needs **more text**; more time won't help.
- **Both stuck:** it has learned what it can at this size. Add more text, or
  delete `data/brain/` and let it start over at a bigger tier (edit
  `model.tiers` in `config/settings.yaml`).

### How much text and time?

| Text | Good training time (small tier) |
|---|---|
| 200 KB (minimum) | ~20 minutes, then it starts memorizing |
| 5 MB (a few dozen books) | 2–4 hours |
| 50 MB+ | overnight or longer; it keeps improving |

### Stopping and resuming

Press **Ctrl+C** at any time. It saves before exiting and prints
`Stopped; the checkpoint was saved.` The next `train` picks up exactly where
it left off.

### Settings that change it (`config/settings.yaml` → `training:`)

| Setting | Default | Effect |
|---|---|---|
| `batch_size` | 16 | Text snippets per learning step. Higher is smoother but slower per step. |
| `learning_rate` | 0.001 | Peak learning speed. Too high and the loss jumps around; too low and it's slow. |
| `val_fraction` | 0.05 | Share of text held out to measure `held-out`. |
| `checkpoint_every_s` | 300 | How often it saves, in seconds. |
| `min_new_model_chars` | 200000 | Minimum text needed to create a new model. |

---

## `research`

**Runs on its own: reads Wikipedia, stores what it finds, and trains itself on it.**

```bash
python cli.py research [--hours N]
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
5. Repeats. It reads at most **6 topics per hour**; the rest of the time it
   just keeps training.

### Example output (illustrative)

```
Researching and self-training for 4 hour(s). Ctrl-C stops it; progress is saved as it goes.
Researched 'pricing strategy': 41 new passages (Pricing strategies, Price discrimination).
Not training yet: Only 96 KB of text so far; a new model needs at least 200 KB to learn its vocabulary from.
Researched 'unit economics': 18 new passages (Unit economics, Contribution margin).
Researched 'business model': 37 new passages (Business model, Business model canvas).
Creating a new model for the 'small' tier (15.7 GB RAM).
...
step    300 | loss 5.412 | held-out 5.630 | 3,480 tok/s | lr 8.1e-04
...
Done: 24 topic(s), 812 new passages, 214 min of training.
```

On a fresh install it reads 2–3 topics before it has enough text to create
the model. That's normal.

### If something goes wrong

- **No internet:** it prints `Couldn't fetch '...' from Wikipedia`, keeps
  training on what it has, and retries the topic later.
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
python cli.py ask "<your claim>" [--no-recheck] [--json]
```

| Option | What it does |
|---|---|
| `"<claim>"` | The statement to judge. **Use quotes.** A full sentence works best. |
| `--no-recheck` | Skip the repeat run. About twice as fast, but it can't catch the council disagreeing with itself. |
| `--json` | Print the complete result as JSON instead of the readable report, for other programs or for inspecting every number. |

It takes a few seconds, and works fine while `train` or `research` runs in
another terminal.

### Example output, annotated

```
CLAIM: Light is not refracted when it passes into glass
Model: 4x192 transformer, 2.20M params, 4.7M tokens trained, held-out loss 4.54   <- (1)
Router: no Investor - No money-related words, so no Investor.                      <- (2)
Familiarity: claim loss 7.11 vs. typical 4.54 -> confidence ceiling Low           <- (3)
Knowledge base: 3 relevant passage(s)                                              <- (4)

BELIEVER  [Low]  signal -0.57                                                      <- (5)
  "This is true because in the two Prisms ... refracted in the first Prism ..."    <- (6)
  - +0.07 Much after the same manner, if ACBD ... (Opticks (Newton, 1704))         <- (7)

SKEPTIC  [Low]  signal -0.23
  "This is false because the Sun's Light is also of these Sides ..."

JUDGE  [Low, capped from Medium]  relied on: skeptic                               <- (8)
  Verdict: No - on what this model has read, the claim doesn't hold up.            <- (9)
  Why: Believer signal -0.57 vs. Skeptic -0.23 nats/token (margin -0.34).          <- (10)
  In its own words: "In conclusion, which comes through the Sun ..."
  Unresolved: the claim is unlike most of what the model has read ...              <- (11)

JUDGE, REPEAT RUN  [Low, capped from Medium]  relied on: believer                  <- (12)
  Verdict: Yes - on what this model has read, the claim holds up.
  ...
Low confidence: the council didn't agree with itself on a repeat run (verdict flipped: no -> yes).
OVERALL CONFIDENCE: Low                                                            <- (13)
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
4. **Knowledge base:** how many stored passages matched the claim well enough
   to be used. They're placed in front of the claim when the model reads it.
5. **Signal:** how much the claim makes the model expect "This is true."
   (Believer) or "This is false." (Skeptic), measured in nats per token.
   Positive means the claim pushes toward that side; **around 0.5 or more is
   strong**. The Investor compares "will make money" against "will lose money".
6. **The quote:** the persona's position, written by the model itself.
   Expect rough text from a small model; the signal is what counts.
7. **Evidence:** stored passages that pushed the model toward this persona's
   side, with how much each one moved the signal.
8. **Judge confidence:** "capped from Medium" means its own estimate was
   Medium, but a rule lowered it. **Relied on** is whose case the verdict
   rests on; the Judge can never be more confident than those personas.
9. **Verdict:** **Yes**, **No**, or **Undecided**. "No" is a normal answer,
   not an error.
10. **Why:** the numbers behind the verdict. **Margin** = Believer − Skeptic
    (averaged with the Investor when present):
    - above +0.1 is Yes;
    - below −0.1 is No;
    - in between is Undecided.
11. **Unresolved:** what's holding confidence down.
12. **Repeat run:** shown only when the second pass disagreed. The council
    re-answers with a bit of randomness in the network. A real signal
    survives that; noise doesn't.
13. **Overall confidence:** the final word. **High** is only possible when
    the signals are strong, the claim is familiar, the model is well
    trained, and both runs agree.

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
python cli.py doctor
python cli.py research --hours 2
python cli.py ask "Customers prefer simple pricing"
```

**With your own documents:**

```bash
python cli.py train --data ~/my_documents --hours 3
python cli.py ask "..."
```

**Let it grow overnight:**

```bash
python cli.py research --hours 8
```

**Train and ask at the same time:** run `research` in one terminal, and
`ask` in a second one (VS Code: the **+** button in the terminal panel).

**Start over:** delete `data/brain/` (model only) or all of `data/`
(everything), then train again.

---

## Using it from Python instead of the terminal

`ask` is a thin wrapper around one function:

```python
from engine.orchestrator import evaluate

result = evaluate("We should switch to usage-based pricing")

result["judge"]["verdict"]        # "No - on what this model has read, ..."
result["judge"]["stance"]         # "yes" | "no" | "mixed"
result["overall_confidence"]      # "Low" | "Medium" | "High"
result["panel"]["skeptic"]        # each persona's signal, confidence, position, evidence
result["consistency"]["agreed"]   # did the repeat run agree?
```

Optional arguments: `recheck=False` skips the repeat run, and `record=False`
leaves the question out of the session log. `python cli.py ask --json "..."`
shows the full structure.
