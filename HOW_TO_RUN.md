# How to run council-engine

## What kind of program is this?

It's a **terminal (command-line) program** called `council`. It's not a
website or an app with buttons: you type a command, press Enter, and it
prints the answer as text.

You can run it from any terminal:

- **VS Code** (easiest if you already use it): open the project folder, then
  **Terminal → New Terminal** from the top menu. A terminal opens at the
  bottom of the window.
- **macOS**: the Terminal app.
- **Windows**: PowerShell or Windows Terminal.
- **Linux**: any terminal.

Everything runs on your own computer. No internet is needed, except for
`research`, which reads Wikipedia. It's written in Rust and uses every core
of your processor, with SIMD (AVX2 / AVX-512 / NEON) where the CPU has it.

There are two ways to get it. **Option A** needs no setup. **Option B**
builds it from the source code.

---

## Option A: download the ready-made program

1. On the project's GitHub page, open **Releases** (right-hand side) and
   download the file for your system:

   | System | File |
   |---|---|
   | Windows | `council-windows-x86_64.exe` |
   | Mac (Apple Silicon) | `council-macos-arm64` |
   | Linux | `council-linux-x86_64` |

2. Rename it to `council` (`council.exe` on Windows) and put it in a folder
   of its own.
3. Open a terminal in that folder and run:

   ```bash
   ./council doctor           # macOS / Linux
   .\council.exe doctor       # Windows
   ```

   - **macOS / Linux:** first make it runnable with `chmod +x council`.
   - **macOS:** if it says the developer can't be verified, run
     `xattr -d com.apple.quarantine council` once. Or right-click the file
     in Finder, choose **Open**, then confirm.

Run this way, it keeps its settings in `~/.council/config/` and everything it
learns in `~/.council/data/`. They're created on first run, and `doctor`
shows the exact paths.

> **No Releases yet?** They're published when the maintainer pushes a
> version tag (`git tag v0.2.0 && git push origin v0.2.0`). Every push also
> builds all three programs: GitHub → **Actions** → the latest **CI** run →
> **Artifacts** at the bottom. Or use Option B.

---

## Option B: build it yourself

### 1. Install Rust (one time)

Go to <https://rustup.rs> and follow the one-line instructions:

- **macOS / Linux:**

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

  Then close and reopen the terminal.
- **Windows:** download and run `rustup-init.exe`. When it offers to install
  the Visual Studio C++ build tools, say yes (Rust needs them to build
  programs). Then close and reopen the terminal.

Check it worked:

```bash
cargo --version
```

### 2. Get the code

**With Git:**

```bash
git clone https://github.com/CodeStuffed/Aegist.git
cd Aegist
```

**Without Git:** on GitHub click the green **Code** button → **Download
ZIP**, unzip it, and open the folder in VS Code (or `cd` into it).

> Until pull request #1 is merged, the new code lives on the branch
> `claude/admiring-keller-bzw2gn`. With Git, run
> `git checkout claude/admiring-keller-bzw2gn`. Without Git, switch to that
> branch on GitHub before downloading the ZIP.

### 3. Build and install the `council` command

Inside the project folder:

```bash
cargo install --path .
```

The first build downloads its libraries and compiles with full optimization,
which takes a few minutes. After that, `council` works in any new terminal.

(Prefer not to install it? `cargo build --release` builds it in place. Then
run it as `./target/release/council` on macOS/Linux, or
`.\target\release\council.exe` on Windows.)

### 4. Check it works

```bash
council doctor
```

It shows your RAM, CPU and SIMD support, the model size it picked for your
machine, where your settings and data live, and that there's no trained
model yet. That's expected.

**Run `council` from inside the project folder** to use the project's
`config/` and `data/` folders. Anywhere else it uses `~/.council/`.

---

## The commands

For every option, annotated example output, and the settings behind each
command, see **[COMMANDS.md](COMMANDS.md)**.

| Command | What it does |
|---|---|
| `council doctor` | Status: hardware, model, how much it has learned. |
| `council train --data <folder> --hours 2` | Teach the model from your `.txt` / `.md` files. |
| `council research --hours 4` | Let it read Wikipedia and train itself on what it finds. |
| `council ask "<claim>"` | Put a claim in front of the council. |
| `council import-wikipedia <dump.xml.bz2>` | Add a whole Wikipedia download to the training text and the knowledge base. |
| `council gpu-check` | Test an NVIDIA GPU for training (see [GPU_TRAINING.md](GPU_TRAINING.md)). |
| `council remember <file or folder>` | Add your own documents, of any size, for `ask` to look things up in. |
| `council eval` | Measure how often the model tells a true claim from a false one. |

**Ctrl+C** stops `train` or `research` at any time. It saves first, so nothing is lost.

## A good first session

```bash
council doctor
council research --hours 2        # or: council train --data my_texts --hours 2
council doctor                    # see how much it learned
council ask "Raising prices will reduce the number of customers"
```

Then keep feeding it: more `research` or `train` time makes it better. It
needs **at least 200 KB of text** to start a model, and works much better
with several MB. Good sources: books from <https://www.gutenberg.org> (the
"Plain Text UTF-8" version), articles, your own notes.

## Where things are stored

`council doctor` prints both locations under **Files**.

| Folder / file | What it is |
|---|---|
| `config/settings.yaml` | every setting: model sizes, thresholds, research topics |
| `config/personas/*.yaml` | what each persona writes and listens for |
| `data/corpus/` | all the text it trains on (yours plus Wikipedia) |
| `data/brain/` | the trained model itself |
| `data/knowledge_base/` | passages it can look up when answering (`index/` is their search index) |
| `data/brain/token_cache/` | the corpus turned into tokens, cached (safe to delete; it's rebuilt) |
| `data/sessions.jsonl` | a log of the claims you've asked about |

- **Start the model over** (for example after changing its size): delete `data/brain/`.
- **Start everything over**: delete the whole `data/` folder.

## If something goes wrong

| You see | What to do |
|---|---|
| `council: command not found` | Open a new terminal after installing. Or run it as `./target/release/council` (Option B) or `./council` (Option A). |
| Windows: `linker 'link.exe' not found` | Install the Visual Studio C++ build tools (rerun `rustup-init.exe` and accept them). |
| macOS: "cannot be opened because the developer cannot be verified" | `xattr -d com.apple.quarantine council`, or right-click → Open. |
| `No trained model ...` | Train first: `council train --data <folder> --hours 1` or `council research --hours 1`. |
| `Only 12 KB of text so far ...` | Give it more text (at least 200 KB), or run `research` longer. |
| `... made by an older version ...` | Your model came from the old Python version. Delete `data/brain/` and train again; your text and knowledge base are kept. |
| `Couldn't fetch '...' from Wikipedia` | Check your internet connection. It keeps training and retries later. |
| Training is slower than expected | On the CPU: close heavy programs, or pick a smaller size (`--tier`). With an NVIDIA GPU, see [GPU_TRAINING.md](GPU_TRAINING.md): `council gpu-check` says whether training can use it. |
| The written positions are nonsense | Normal for a small model early on. Judge it by the signals and confidence, and give it more text and training time. |

## For developers

```bash
cargo test                                  # the whole test suite, no internet needed
cargo test --features gpu-emulator --lib gpu  # the GPU code, on a CPU emulator (needs a C++20 compiler)
cargo run --release --example bench         # training speed on this machine
council ask --json "..."                    # machine-readable output for other programs
```

The council is also a Rust library: `council::council::evaluate(&brain, claim,
&settings, &options, &mut log)` is the same function `council ask` uses.
