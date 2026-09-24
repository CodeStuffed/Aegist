# How to run council-engine

## What kind of program is this?

It's a **terminal (command-line) program**. It's not a website or an app with
buttons: you type a command, press Enter, and it prints the answer as text.

You can run it from any terminal:

- **VS Code** (easiest if you already use it): open the project folder, then
  **Terminal → New Terminal** from the top menu. A terminal opens at the bottom
  of the window, already inside the project folder.
- **macOS**: the Terminal app.
- **Windows**: PowerShell or Windows Terminal.
- **Linux**: any terminal.

Everything runs on your own computer. No internet is needed, except for
`research`, which reads Wikipedia.

---

## One-time setup

### 1. Install Python 3.11 or newer

Check what you have:

```bash
python --version
```

If that says "not found" or shows a version below 3.11, install Python from
<https://www.python.org/downloads/>. **On Windows, tick "Add python.exe to PATH"**
during install.

> On some Macs and Linux machines the command is `python3` instead of `python`.
> If `python` doesn't work, use `python3` everywhere below. On Windows you can
> also use `py`.

### 2. Get the code

**Option A: with Git**

```bash
git clone https://github.com/CodeStuffed/Aegist.git
cd Aegist
```

**Option B: without Git**. On the GitHub page, click the green **Code** button,
choose **Download ZIP**, unzip it, and open that folder in VS Code (or `cd`
into it in your terminal).

> Until pull request #1 is merged, the new code lives on the branch
> `claude/admiring-keller-bzw2gn`. With Git, run
> `git checkout claude/admiring-keller-bzw2gn`. Without Git, switch to that
> branch on GitHub before downloading the ZIP.

### 3. Create a private Python environment and install the pieces it needs

This keeps the project's packages separate from the rest of your computer.

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If Windows says running scripts is disabled, run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Once it's activated, your prompt starts with `(.venv)`. **Every time you open a
new terminal** to use the program, run the activate line again (`source
.venv/bin/activate` or `.venv\Scripts\Activate.ps1`). VS Code often does this for you.

### 4. Check it works

```bash
python cli.py doctor
```

This shows your computer's RAM and CPU, the model size it picked for your
machine, and that there's no trained model yet. That's expected.

---

## The commands

> For every option, annotated example output, and the settings behind each
> command, see **[COMMANDS.md](COMMANDS.md)**.

All commands start with `python cli.py`. Add `--help` to any of them to see its
options, for example `python cli.py train --help`.

### `doctor`: check status

```bash
python cli.py doctor
```

Shows your hardware and model size, how much text it has, how long the model has
trained, and how many passages are in its knowledge base. Safe to run anytime.

### `train`: teach the model from your text

```bash
python cli.py train --data "path/to/your/texts" --hours 2
```

- `--data` is a folder (or a single file) of `.txt` or `.md` files. They're
  copied into the project, so you only need `--data` once per folder.
- `--hours` is how long to train. Default 1. Decimals work (`--hours 0.5`).
- `--steps 500` trains for a fixed number of steps instead of a set time.
- Leave `--data` off to keep training on the text it already has:
  `python cli.py train --hours 3`.
- **Ctrl+C** stops early. Progress is saved; nothing is lost.

It needs **at least 200 KB of text** to start a model, and works much better
with several MB. Good sources: books from <https://www.gutenberg.org> (download
the "Plain Text UTF-8" version), articles, your own notes. Or skip this and use
`research` below.

While it trains you'll see lines like:

```
step    420 | loss 4.005 | held-out 4.782 | 3,500 tok/s | lr 9.3e-04
```

**Lower numbers mean it's learning.** "held-out" is the one that matters: it's
measured on text the model hasn't trained on. If "loss" keeps falling but
"held-out" stops falling, the model is memorizing your text. Give it more text.

### `ask`: put a claim in front of the council

```bash
python cli.py ask "We should switch to usage-based pricing"
```

Put the claim in quotes. You get:

- **Believer / Skeptic** (plus **Investor** if the claim is about money): each
  one's signal, confidence, a position written by the model, and any stored
  passages that pushed it toward its side.
- **Judge**: Yes / No / Undecided, why, and how confident it is.
- **Repeat run**: it answers a second time with some randomness switched on.
  If the two answers disagree, it says so and drops to Low confidence.
- **OVERALL CONFIDENCE** at the bottom.

Options:

- `--no-recheck` skips the repeat run (faster, less careful).
- `--json` prints the full raw result, for use by other programs.

A new or lightly trained model answers **Low** to almost everything. That's
deliberate: it won't pretend to know things it hasn't learned.

### `research`: let it read Wikipedia and train itself

```bash
python cli.py research --hours 4
```

It repeats this until time runs out:

1. picks a topic (from `seed_topics` in `config/settings.yaml`, plus words
   that keep coming up in your `ask` questions),
2. reads a couple of Wikipedia articles about it,
3. saves them to its knowledge base and training text,
4. trains on everything it has, until it's time for the next topic.

It fetches at most 6 topics an hour, to be polite to Wikipedia. It needs internet.
**Ctrl+C** stops it anytime; everything is saved. It's fine to leave it running
overnight (`--hours 8`).

---

## A good first session

```bash
python cli.py doctor
python cli.py research --hours 2        # or: python cli.py train --data my_texts --hours 2
python cli.py doctor                    # see how much it learned
python cli.py ask "Raising prices will reduce the number of customers"
```

Then keep feeding it: more `research` or `train` time makes it better.

---

## Where things are stored

Everything it learns lives in the `data/` folder inside the project:

| Folder / file | What it is |
|---|---|
| `data/corpus/` | all the text it trains on (yours plus Wikipedia) |
| `data/brain/` | the trained model itself |
| `data/knowledge_base/` | passages it can look up when answering |
| `data/sessions.jsonl` | a log of the claims you've asked about |

- **Start the model over** (for example after changing its size): delete `data/brain/`.
- **Start everything over**: delete the whole `data/` folder.

Settings (model size, thresholds, research topics) are in
`config/settings.yaml`. It's a plain text file you can edit in VS Code.

---

## If something goes wrong

| You see | What to do |
|---|---|
| `python: command not found` | Use `python3` (Mac/Linux) or `py` (Windows), or install Python (step 1). |
| `No module named numpy` (or another module) | Activate the environment (step 3), then `pip install -r requirements.txt`. |
| `No trained model ...` | Train first: `python cli.py train --data <folder> --hours 1` or `python cli.py research --hours 1`. |
| `Only 12 KB of text so far ...` | Give it more text (at least 200 KB), or run `research` longer. |
| `Couldn't fetch '...' from Wikipedia` | Check your internet connection. It keeps training and retries later. |
| Training is slow | Normal: it runs on your CPU. Close heavy programs, or pick a smaller tier in `config/settings.yaml`. |
| The written positions are nonsense | Normal for a small model early on. Judge it by the signals and confidence, and give it more text and training time. |

## For developers

Run the tests (takes about 30 seconds, no internet needed):

```bash
python -m pytest
```

Use it from other Python code instead of the terminal:

```python
from engine.orchestrator import evaluate

result = evaluate("We should switch to usage-based pricing")
print(result["judge"]["verdict"], result["overall_confidence"])
```
