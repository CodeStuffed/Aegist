# Training on an NVIDIA GPU (the RTX 5080 plan)

`council` can train its model on an NVIDIA graphics card. Everything is
still written from scratch: the GPU code in `src/gpu/` is hand-written CUDA
kernels plus NVIDIA's own matrix-multiply library (cuBLAS). No pretrained
weights, no outside AI.

This page covers:
1. setting up the card;
2. getting a big pile of text (all of English Wikipedia);
3. a 4-day training plan;
4. what to expect at the end, honestly.

---

## 1. Set up the GPU (once)

You need two things from NVIDIA:

| What | Why | Where |
|---|---|---|
| **A recent NVIDIA driver** (for an RTX 50-series card: version 570 or newer) | Lets programs use the card | Usually already installed. Otherwise use the NVIDIA App, or <https://www.nvidia.com/drivers> |
| **The CUDA Toolkit 12.8 or newer** (12.9 or 13.x is fine too) | Provides cuBLAS (matrix multiplies) and NVRTC, which compiles `council`'s GPU code for your card | <https://developer.nvidia.com/cuda-downloads> |

**Windows:** in the download page pick *Windows → x86_64 → your version →
exe (local)* and run the installer with the default options. It sets
`CUDA_PATH`, which is where `council` looks for the libraries. Open a
**new** terminal afterwards.

**Linux:** follow the page's instructions for your distribution (it adds
NVIDIA's package repository), for example `sudo apt install cuda-toolkit`.
`council` looks in `/usr/local/cuda/lib64` and the standard library folders.

**WSL2 on Windows** works too: install the Windows driver as normal, and
the CUDA Toolkit for WSL inside Linux.

### Check it

```bash
council gpu-check
```

It compiles the GPU code for your card, runs a small model through **both
the GPU and the CPU**, and checks that every gradient matches: exactly in
fp32, and within bf16's rounding for the bf16 path. Then it
measures the card's speed and lists which model sizes fit. You should see
`Check passed`. If not, the message says what's missing (driver, CUDA
Toolkit, or a result that doesn't match).

From then on, `council train` and `council research` use the GPU
automatically (`--device auto`, the default). They run the same self-check
first, and fall back to the CPU if it fails. To choose yourself, use
`--device gpu` or `--device cpu`.

---

## 2. Get the text: all of English Wikipedia

A model is only as good as what it reads. English Wikipedia is about 20 GB
of clean text (4 to 5 billion words), which is about the right amount for
what a 5080 can train in a few days.

1. Download the dump from <https://dumps.wikimedia.org/enwiki/latest/>. The
   file is **`enwiki-latest-pages-articles-multistream.xml.bz2`** (about
   24 GB; don't unpack it). Other languages work the same way (`dewiki`,
   `frwiki`, ...).
2. Try a small slice first to see it working:
   ```bash
   council import-wikipedia enwiki-latest-pages-articles-multistream.xml.bz2 --max-articles 20000
   ```
3. Then import the rest. Running it again continues where it stopped:
   ```bash
   council import-wikipedia enwiki-latest-pages-articles-multistream.xml.bz2
   ```

What it does:
- unpacks the dump on every CPU core;
- turns the wiki markup into plain paragraphs (dropping infoboxes, tables,
  references and lists of links);
- writes the text to `data/corpus/wikipedia/` for training;
- stores every paragraph in the knowledge base (`data/knowledge_base/`),
  linked to its neighbors, so `ask` can look things up.

On a 4-core test machine it processed about 50 MB of wiki markup a second
and indexed a million passages in 45 seconds. At those rates the whole dump
takes roughly one to two hours, less with more cores. Ctrl-C is safe.

**Disk space:** plan for about **110 GB** free: the dump (24 GB), the text
(~20 GB), its tokens (~20 GB with the cache) and the knowledge base with its
search index (~40 GB). `--no-knowledge` skips the knowledge base if space is tight.

---

## 3. The 4-day plan

```bash
council gpu-check                 # 1. confirm the card works; see sizes and speed
council train --hours 96          # 2. train for 4 days
```

What `train` does with a new model:

1. **Picks the size** that will be smartest when the 96 hours are up, from
   the card's measured speed and memory. The rule: the biggest model that
   can still read ~20 tokens of text per parameter in that time. A bigger
   one would be cut off half-trained (and be *worse*); a smaller one would
   stop improving early. The log says which size it picked and why.
   `council gpu-check` shows the choice in advance.
2. **Learns a 32,768-token vocabulary** from a sample spread across all
   your text (seconds).
3. **Turns all the text into tokens once.** For all of Wikipedia that's
   minutes at the measured 60–75 MB/s on 4 cores, and later runs reuse it.
4. **Trains**, printing progress every 10 seconds and saving every 5
   minutes. Ctrl-C saves and stops; the same command continues.

**Training in several sessions?** Give the total up front so the size fits
it: `council train --plan-hours 96 --hours 24` today, then
`council train --hours 24` three more times. Only the first session of a
new model uses `--plan-hours`.

**Watch it:** the `held-out` number (loss on text it never trains on) should
keep falling. `council doctor` shows the whole history.

Once trained, `council ask "..."` uses it. The knowledge base puts the most
relevant Wikipedia paragraphs, and the paragraphs linked to them, in front
of every question.

Answering runs on the CPU (int8 weights, every core for big models). With
an untrained 235M-parameter model, a full `ask` (with the repeat run) took 40
seconds on **one** core of the 4-core test machine while it was also
training; with a desktop's free cores expect several seconds. `--no-recheck`
roughly halves it.

### Memory

- **GPU (16 GB on a 5080):** only one layer's working values are kept at a
  time (the backward pass recomputes each layer), so ~300M-parameter models
  fit comfortably. `gpu-check` shows how many sequences fit per
  micro-batch for each size.
- **RAM:** the CPU keeps a copy of the weights for saving: ~12 bytes per
  parameter (4 GB for a 337M model). 40 GB of RAM is plenty.

---

## 4. What to expect, honestly

**Speed.** `council gpu-check` measures your card. As a rough guide, an RTX
5080 does on the order of 50 trillion TF32 operations per second on matrix
multiplies, and about twice that in bf16. Training uses bf16 (with fp32
accumulation and fp32 weights) wherever the card supports it
(`gpu.precision`). Training reaches a good share of that speed, which puts a
~200-350M parameter model at roughly **10,000–20,000 tokens per second**.
In 4 days that's about 3 to 7 billion tokens: one to two reads through
English Wikipedia. These are estimates, not measurements (this code hasn't
run on a 5080 yet); the real numbers print when training starts.

**How smart.** A model this size trained on Wikipedia is in the same class
as GPT-2 (2019):
- It writes fluent, Wikipedia-style English.
- It knows common facts some of the time.
- Its sense of which statements "sound true" is much better than a
  CPU-trained model's, which is what the council measures.

It will **not**:
- follow instructions or chat;
- reason step by step;
- do reliable math;
- know anything after the dump's date.

**Why not a 72B or trillion-parameter model?** The biggest limit isn't
memory, it's compute:
- **Frontier models** are trained with about 10²⁵ operations or more.
- **Four days on a 5080** is about 10¹⁹, a million times less.
- **Spread over 72B parameters,** 10¹⁹ operations would be about 24M tokens:
  a third of a token per 1,000 parameters, which is noise.
- **Training a 72B model would also need about 1.1 TB of memory**, 70 times
  what the card has.

So the useful maximum is the size `train` picks. The rest of the
"intelligence" comes from reading more text and looking things up at
question time (the knowledge base) rather than from size.

**Long context.** The model reads 1,024 tokens at a time; that's what a card
this size can train in 4 days. The knowledge base is how it gets at more.
It holds all of Wikipedia on disk (billions of words) and hands the model
only the passages relevant to each question.

---

## Troubleshooting

| Message | What to do |
|---|---|
| `no NVIDIA driver found` | Install or update the NVIDIA driver, then reboot. |
| `NVRTC ... wasn't found` / `cuBLAS ... wasn't found` | Install the CUDA Toolkit 12.8+ and open a new terminal. On Windows, check `CUDA_PATH` points at it. |
| `NVRTC couldn't compile the kernels` | The CUDA Toolkit is older than the card. RTX 50xx needs 12.8 or newer. |
| `the GPU's results don't match the CPU's` | Update the driver and CUDA Toolkit. Training stays on the CPU until the check passes. |
| `needs about X GB of GPU memory` | Close other programs using the GPU (games, browsers with hardware acceleration), or pick a smaller size with `--tier`. |
| Training uses the CPU although there's a GPU | Run `council gpu-check`; the first lines say why. |

## For developers

- `src/gpu/kernels.cu`: the CUDA kernels.
- `src/gpu/cuda.rs`: the driver/cuBLAS/NVRTC bindings, loaded at run time.
- `src/gpu/mod.rs`: the training step.

`cargo test --features gpu-emulator` runs the same kernels on the CPU,
through a small emulator (`src/gpu/emulate.cpp`, needs a C++20 compiler),
and checks the whole GPU training step against the CPU model. It also
checks that a model trained on the "GPU" resumes on the CPU. CI runs these
too.
