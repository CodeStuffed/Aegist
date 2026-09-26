# Training Aegist on an NVIDIA GPU

Aegist can train its model on an NVIDIA graphics card. Everything is still
written from scratch: the GPU code in `src/gpu/` is hand-written CUDA
kernels plus NVIDIA's own matrix-multiply library (cuBLAS). There are no
pretrained weights and no outside AI.

A GPU is the single biggest step up in how smart Aegist can get. A 4-core
CPU trains a ~6M-parameter model well in about a day. A modern card trains
a few-hundred-million-parameter model in a few days.

This page covers:

1. setting up the card;
2. getting enough code to learn from;
3. a multi-day training plan;
4. what to expect at the end, honestly.

---

## 1. Set up the GPU (once)

You need two things from NVIDIA:

| What | Why | Where |
|---|---|---|
| **A recent NVIDIA driver** (RTX 50-series: 570 or newer) | Lets programs use the card | Usually already installed; otherwise the NVIDIA App or <https://www.nvidia.com/drivers> |
| **The CUDA Toolkit 12.8 or newer** | Provides cuBLAS (matrix multiplies) and NVRTC, which compiles Aegist's kernels for your card | <https://developer.nvidia.com/cuda-downloads> |

- **Windows:** pick *Windows → x86_64 → your version → exe (local)* and install with the default options. It sets `CUDA_PATH`, which is where Aegist looks. Open a **new** terminal afterwards.
- **Linux:** follow the page's instructions for your distribution (for example `sudo apt install cuda-toolkit`). Aegist looks in `/usr/local/cuda/lib64` and the standard library folders.
- **WSL2:** install the Windows driver as normal, and the CUDA Toolkit for WSL inside Linux.

### Check it

```bash
aegist gpu-check
```

It compiles the kernels for your card, runs a small model through **both
the GPU and the CPU**, and checks that every gradient matches. Then it
measures the card's speed and lists the model sizes that fit. You should
see `check passed`.

From then on, `aegist train` uses the GPU automatically (`--device auto`).
It runs the same self-check first and falls back to the CPU if it fails.
`--device gpu` or `--device cpu` chooses explicitly.

---

## 2. Get the code

A model is only as good as what it reads, and a GPU can read a lot. A few
days on a modern card reads several billion tokens, so give it a few GB of
code.

```bash
aegist learn --pack python           # well-known Python projects
aegist learn --pack javascript       # and so on: typescript, web, games, rust, go, c, cpp, java, os, ai, algorithms
aegist learn ~/code                  # all your own projects
aegist learn https://github.com/some/repo
```

`aegist learn --list` shows the packs and everything learned so far.
`learn` keeps only real source code: nothing git ignores, no dependency
folders, no generated or minified files, no exact duplicates.

For a really large corpus, clone many repositories into one folder
yourself and `aegist learn` that folder.

---

## 3. The plan

```bash
aegist gpu-check                   # 1. confirm the card works; see sizes and speed
aegist train --hours 96            # 2. train for 4 days
```

What `train` does with a new model:

1. **Picks the size** that will be smartest when the time is up, from the
   card's measured speed and memory. The rule is the biggest model that can
   still read about 20 tokens of code per parameter in that time. A bigger
   one would be cut off half-trained (and be *worse*); a smaller one would
   stop improving early. The log says which size it picked and why.
2. **Learns a 32,768-token vocabulary** from a sample of all the code.
3. **Turns the corpus into tokens once**; later runs reuse it. Half the
   files become fill-in-the-middle examples, which is what teaches the model
   to complete code in the middle of a file.
4. **Trains**, printing progress every 10 seconds and saving every 30
   minutes. Ctrl-C saves and stops; the same command continues.

The GPU sizes train on **2,048 or 4,096 tokens of context**, and Aegist
reads up to four times that when writing code, by stretching the rotary
positions it learned (`inference.context_scale`). Beyond that, it searches
your project for the code relevant to each request, so it can use your
whole repository, not just what fits in the context.

**Training in several sessions?** Give the total up front so the size fits
it: `aegist train --plan-hours 96 --hours 24` today, then
`aegist train --hours 24` three more times. The learning rate follows one
schedule across all four.

**Watch it:** the `held-out` number (loss on code it never trains on)
should keep falling. `aegist doctor` shows the history, and `aegist eval`
measures the model on coding problems by running their tests.

### Memory

- **GPU (16 GB on an RTX 5080):** only one layer's working values are kept
  at a time (the backward pass recomputes each layer), so ~300M-parameter
  models fit. `gpu-check` shows how many sequences fit per micro-batch for
  each size.
- **RAM:** the CPU keeps a copy of the weights for saving, about 12 bytes
  per parameter (4 GB for a 337M model).

---

## 4. What to expect, honestly

**Speed.** `aegist gpu-check` measures your card. As a rough guide, an RTX
5080 does on the order of 50 trillion TF32 operations a second, about twice
that in bf16, which training uses where the card supports it. That puts a
~200-350M-parameter model at roughly 10,000–20,000 tokens a second, or 3–7
billion tokens in 4 days. These are estimates from the hardware's specs,
not measurements: the real numbers print when training starts.

**How smart.** A model this size, trained this long on code, is in the
class of the small open code models of 2022 (a few hundred million
parameters):

- It completes lines, loops and small functions in the style of what it
  read, often correctly.
- It writes short, common functions from a docstring or comment some of
  the time. `aegist eval` measures exactly how often.
- Asking for several candidates and keeping the one that passes the checks
  (what Aegist does on every request) finds a working answer much more
  often than a single try.

It will **not**:

- follow complicated English instructions or hold a conversation;
- design a large program on its own;
- be anywhere near the frontier AI models that labs train on thousands of
  GPUs.

That gap is compute, not code. Frontier models are trained with 10²⁵
operations or more; four days on one card is about 10¹⁹, a million times
less. What Aegist adds on top is honesty: it checks everything it writes
and says no when the checks fail, so a small model's mistakes don't reach
your files unannounced.

---

## Troubleshooting

| Message | What to do |
|---|---|
| `no NVIDIA driver found` | Install or update the NVIDIA driver, then reboot. |
| `NVRTC ... wasn't found` / `cuBLAS ... wasn't found` | Install the CUDA Toolkit 12.8+ and open a new terminal. On Windows, check `CUDA_PATH` points at it. |
| `NVRTC couldn't compile the kernels` | The CUDA Toolkit is older than the card. RTX 50xx needs 12.8 or newer. |
| `the GPU's results don't match the CPU's` | Update the driver and CUDA Toolkit. Training stays on the CPU until the check passes. |
| `needs about X GB of GPU memory` | Close other programs using the GPU, or pick a smaller size with `--tier`. |
| Training uses the CPU although there's a GPU | Run `aegist gpu-check`; its first lines say why. |

## For developers

- `src/gpu/kernels.cu`: the CUDA kernels.
- `src/gpu/cuda.rs`: the driver, cuBLAS and NVRTC bindings, loaded at run time.
- `src/gpu/mod.rs`: the training step.

`cargo test --features gpu-emulator` runs the same kernels on the CPU,
through a small emulator (`src/gpu/emulate.cpp`, needs a C++20 compiler),
and checks the whole GPU training step against the CPU model. CI runs
these too.
