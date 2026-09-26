# Aegist

**A coding AI grown from scratch.** Aegist is a transformer written and
trained from scratch in this repository, in Rust, with its own tokenizer
and its own training loop. There's no Claude, no GPT, no Ollama, no
pretrained weights and no outside AI service. It starts as random numbers
and learns only from the code you give it.

Around the model is a workbench for coding in the terminal: a live,
highlighted session where Aegist writes, completes and fixes code, runs
programs and tests, keeps servers and games running, serves websites, and
looks at web pages in a headless browser, all without leaving the terminal.

Everything the model writes is **checked before you get it**, and when the
checks fail it says **no** instead of handing you something that looks
right and isn't.

```bash
aegist learn --pack python          # code to learn from (or: aegist learn ~/code)
aegist train --hours 1              # train it on your CPU or NVIDIA GPU
aegist                              # open the session
```

## Read this first: what to expect

- **It's a small model you train yourself.** An hour on a laptop gives a
  model that has picked up the style and vocabulary of the code it read.
  Days on a GPU give a few-hundred-million-parameter model in the class of
  the small open code models of 2022. It is **not** comparable to the
  frontier AIs labs train on thousands of GPUs; that gap is about a million
  times more computing, not cleverer code. `aegist eval` measures exactly
  how good yours is.
- **It completes code; it doesn't chat.** It writes code that follows a
  comment, a docstring, a signature or the code around it. It can't answer
  questions or explain things in English. Ask it one and it says no plainly,
  rather than making something up.
- **It tells you when it might be wrong.** Every answer comes with checks:
  the language's own compiler or parser, a search for names it may have
  invented, the tests when there are any, and how sure the model was of each
  token. Unsure tokens are underlined. If the checks fail, nothing touches
  your files. A good average can't hide one made-up line (the weakest
  stretch counts too), it stops writing as soon as it loses track instead
  of guessing on, and when its separate attempts all disagree it says so.
- **It gets better the more it reads and trains.** `learn` adds code and
  `train` updates the model's weights: real self-training, on your machine.

## The session

`aegist` opens an interactive session in whatever folder you run it from.
It reads the project there (respecting `.gitignore`) so the model can see
related code from other files, and so the invented-name check knows your
project's names.

Just say what you want:

```
❯ write a function that checks whether a number is prime
❯ write a snake game as a web page
❯ complete src/app.py:42
❯ fix `pytest -q`
❯ run main.py
❯ where is parse_config defined
```

Or use a command (`/` shows the menu, tab completes):

| Command | What it does |
|---|---|
| `/write <file> <what it should do>` | Write a new file, or add to an existing one, checked before it's saved |
| `/complete <file>[:line]` | Fill in code at a line: a `TODO`, an empty line, or the end of the file |
| `/fix [command]` | Make a failing command pass (default: the project's tests) |
| `/run <command or file>` | Run a program and stream its output (`!command` works too) |
| `/test` | Find and run the project's tests (cargo, npm, go, pytest, make, ...) |
| `/start <command>` · `/jobs` · `/logs <job>` · `/stop <job>` | Keep programs running in the background: dev servers, games, watchers |
| `/serve [folder]` | Serve a website folder on localhost |
| `/preview <url, file or job>` | Load a page in a headless browser and draw it in the terminal, with its console errors |
| `/open <file>[:from-to]` · `/find` · `/grep` · `/tree` | Look around the code |
| `/diff` · `/undo` | Everything Aegist changed this session; put the last change back |
| `/show` | The last code Aegist refused to stand behind, marked unverified |
| `/learn` · `/train [hours]` · `/model` | Teach it more, train it, see its state and limits |
| `/screen` · `/windows` · `/focus` · `/launch` | See the screen (with coordinates), list windows, switch to one, open an app or website |
| `/click` · `/rclick` · `/dclick` · `/mouse` · `/drag` · `/scroll` | Be the mouse |
| `/type <text>` · `/key <combo>` · `/wait` | Be the keyboard |
| `/auto <steps>` (or `/do`) | Plan several steps, show the plan, then do them all |
| `/status` · `/config` · `/history` · `/export` · `/cd` · `/copy` · `/git` | Session state, settings, what you asked, save the transcript, change folder, clipboard, git |
| `/safety` · `/install` | What it will and won't do on your computer; add Aegist to your apps menu |

`/help` groups them all; `/help screen` shows one group.

Keys: **enter** sends, **shift/alt+enter** (or `\` then enter) adds a line,
**↑ ↓** history, **tab** completes commands and paths, **esc** or
**ctrl+c** stops the work in progress, **ctrl+c** twice leaves.

Output scrolls into your terminal's history like any program's; only the
input box and the work in progress are redrawn in place. Copy and paste and
scrollback keep working. Colors are 24-bit where the terminal supports
them, and fall back to 256 or 16 colors, or none with `NO_COLOR` or
`--no-color`.

### How it stays honest

A small model's words aren't evidence, so Aegist checks every answer:

| Check | How |
|---|---|
| **syntax** | The language's own tools: Python's parser, `node --check`, `rustc`, `gofmt`, `cc -fsyntax-only`, `ruby -c`, `php -l`, `bash -n`, ... |
| **names** | Every function, method and module the code uses must exist somewhere: in your project, in the code Aegist learned from, or in the language. A name found nowhere was probably made up, and Aegist names it. |
| **tests** | `/fix` runs your tests on every candidate and only keeps a change that makes them pass. |
| **certainty** | How likely the model found each token it wrote, pulled down when the surrounding code is unlike anything it has read. |

Each request writes several candidates (one careful, the rest more
adventurous), checks all of them, and offers the best that passes. The
verdict can only go down:

- **Verified**: every check passed and tests ran.
- **Checked**: every check it could run passed; nothing has run it yet.
- **Unverified**: something couldn't be checked (for example no checker installed); it's offered, clearly marked.
- **No**: a check failed or the model wasn't sure. Nothing is written, and it says why.

`aegist eval` measures how well these verdicts predict the truth: of the
code Aegist stood behind, how much passed its tests, and of the code it
refused, how much really did fail.

## Your screen: Aegist as the mouse and keyboard

Aegist can use your computer the way you do. Say what to do:

```
❯ open notepad, type hello world and press enter
❯ switch to firefox then scroll down 5
❯ take a screenshot
❯ click 640 400
❯ right click the center
❯ press ctrl+shift+t
❯ open github.com
```

One action runs straight away. Several steps become a **plan**: Aegist
reads the whole request first, shows you every step, and runs them only
when you say go. Each step is checked as it runs (did the app's window
appear? did the window come to the front? did the program exit cleanly?),
and the first one that fails stops the rest. Plans can mix the screen and
code: `fix the tests then run main.py`, `open paint then draw a red circle
in paint`. From a script: `aegist do "open calculator" --yes`.

It **can't read the text on your screen**, so it doesn't guess where
things are. "Click the OK button" gets "I don't know where that is", not a
click somewhere that looks likely. Tell it where instead: `/screen` draws
the screen in the terminal with pixel coordinates along the edges, and
`click 640 400` or `click 50% 20%` clicks there. Windows are found by their
title or by the program they belong to (`switch to xterm`).

Staying safe (`/safety` shows this too):

- It asks once per session before touching your mouse and keyboard
  (`agent.confirm`).
- Keys that close, save, print, lock or delete ask first, and so does typed
  text with line breaks. It never presses ctrl+alt+delete.
- It won't open anything to do with passwords, sign-ins, payments or
  banking, or anything that sends messages in your name.
- **Move the mouse and it lets go.** Esc stops it too, and one run stops
  after `agent.max_actions` actions.

It works on Windows and on Linux under X11. Wayland doesn't let programs
see or control the screen, and macOS isn't supported yet.

## Install

**Build from source** (needs [Rust](https://rustup.rs)):

```bash
git clone https://github.com/CodeStuffed/Aegist
cd Aegist
cargo build --release
./target/release/aegist            # or: cargo install --path .
```

**Or download a build** from the project's Releases page (Windows, macOS,
Linux), make it runnable (`chmod +x` on macOS/Linux) and put it on your
PATH.

**Open it like an app**: `aegist install` (or `/install` in the session)
adds Aegist to your apps menu and desktop, so it opens in its own window
with its own title. That's Windows Terminal when it's installed on Windows,
your usual terminal on Linux, and Terminal on macOS.

Aegist keeps its settings and trained model in its home folder: this repo
when run from a build inside it, otherwise `~/.aegist`. The code you work on
is wherever you run `aegist`. `aegist doctor` shows the paths.

Optional tools make it more capable: `git` (for `learn` from URLs and
packs), each language's own compiler or interpreter (for its syntax checks
and `/run`), and Chrome, Chromium or Edge (for `/preview`; set
`AEGIST_BROWSER` to use a specific one).

## Teaching it

```bash
aegist learn --list                 # the packs, and what's learned so far
aegist learn --pack python          # python, javascript, typescript, web, games, rust, go, c, cpp, java, os, ai, algorithms
aegist learn ~/code                 # your own projects
aegist learn https://github.com/owner/repo
aegist train --hours 8              # longer is smarter; ctrl+c stops and saves
aegist train --plan-hours 96 --hours 24   # several sessions on one schedule
aegist eval                         # measure it
```

Packs are sets of well-known, openly licensed projects, downloaded with git
straight from their repositories. `learn` keeps only real source code:
nothing git ignores, no dependency folders, no generated or minified files,
no exact duplicates.

`train` sizes a **new** model for the time you give it: the biggest model
that can still read about 20 tokens of code per parameter in that time.
Bigger isn't smarter if it can't finish learning, and a small corpus limits
the size too. `aegist doctor` lists every size with its memory and training
time on your machine.

**Have an NVIDIA GPU?** Training is on the order of 100 times faster.
[docs/GPU.md](docs/GPU.md) covers setup and a multi-day plan.

## How it's built

All of it is in `src/`, written from scratch:

- **The model** (`model.rs`, `kernels.rs`): a decoder-only transformer with
  RMSNorm, rotary positions, a SwiGLU MLP and tied embeddings. The forward
  and backward passes are written by hand, and every gradient is checked
  against finite differences in the tests. Matrix multiplies use AVX2,
  AVX-512 or NEON through the `gemm` crate, on every core.
- **Long context**: attention runs a chunk of queries at a time and, during
  training, keeps one number per row instead of the full attention matrix,
  recomputing it in the backward pass (the idea behind FlashAttention). So
  memory grows with the context, not its square, and contexts of 2,048 to
  4,096 tokens train on ordinary machines. When writing code, the model
  reads up to 4 times further than it trained on, by stretching its rotary
  positions (NTK scaling) only when a prompt needs it. And it searches your
  project for relevant code, so its memory reaches the whole repository.
- **Speed**: prompts are read in one batched pass that computes the output
  layer only where needed. Generation guesses ahead by copying from the
  prompt (prompt-lookup decoding) and checks the guesses in the same pass:
  every token is still exactly what the model would have written, but
  repeated code comes out several tokens per step. Weights run as int8 (4x
  smaller than f32, practically identical) or int4, with hand-written
  integer kernels. Candidates share the prompt's cached keys and values and
  are written in parallel.
- **The tokenizer** (`tokenizer.rs`): byte-level BPE trained on your
  corpus, so any code in any language can be encoded.
- **Training** (`trainer.rs`, `optim.rs`, `corpus.rs`): AdamW, warmup plus
  cosine learning rate, gradient clipping, held-out evaluation and
  checkpoints. The corpus is memory-mapped from disk, so billions of tokens
  cost disk space, not RAM. Each file is a document headed by its path, and
  half of them are trained as fill-in-the-middle examples, which is what
  lets the model complete code in the middle of a file.
- **The GPU** (`gpu/`): hand-written CUDA kernels compiled at run time with
  NVRTC, and cuBLAS for matrix multiplies, both loaded only if present. A
  self-check against the CPU runs before any training.
- **The honesty layer** (`verify.rs`, `assist.rs`, `bench.rs`): the checks,
  the verdicts, the candidate search, and the benchmark.
- **The workbench** (`app.rs`, `actions.rs`, `ide.rs`, `project.rs`,
  `ui/`): the session, commands, background jobs, the web server, the
  browser preview, the project index and undo journal, and the terminal
  interface (colors, syntax highlighting, diffs, the live screen region and
  the line editor).

Understanding plain requests is done by simple, predictable rules, not by
another model: a request it can't place gets "I don't know what you want",
never a guess.

## Tests

```bash
cargo test                          # the whole suite, in seconds, without internet
cargo test --features gpu-emulator  # plus the GPU kernels, on a CPU emulator
```

They cover gradient checks, chunked attention against plain attention,
generation (guessing ahead never changes the output), the tokenizer, the
corpus and fill-in-the-middle documents, training, the checks and verdicts,
the benchmark's problems, the project index and undo, the web server, jobs,
the browser preview (when a browser is installed), and the interface's
pieces.

## Layout

```
src/                 the program (aegist) and its library
  computer.rs        the mouse and keyboard: plain words to checked actions
  autopilot.rs       plans of several steps, all understood before any runs
  extras.rs          /status, /config, /history, /export, /git, `aegist install`
config/aegist.yaml   every tunable number
docs/GPU.md          training on an NVIDIA GPU
examples/            training speed per size, quantization's cost, tokenizer speed
data/                corpus, sources, names, model (gitignored)
```
