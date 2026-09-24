> **Superseded.** This was the original plan, built on Claude and Ollama. The project
> has since changed direction: it now runs on a transformer written and trained from
> scratch, with no outside AI. See the README for the current design.

# The Council — Build Brief for Claude Opus 5.5

*Evaluation-pipeline module for your business-agent system: Believer / Skeptic / Investor / Judge, running on a self-researching, hardware-aware backend.*

## Scope check before you paste this into Opus

A few pieces of the original spec don't map onto how these systems actually work — worth knowing before Opus spends time on them:

- **"Self-trains for hours"** can't mean updating model weights — that takes compute only a handful of labs on Earth have. What's actually buildable, and what's below, is an autonomous *research loop*: it searches the web on its own, saves findings to a growing local knowledge base, and pulls that back into future evaluations. That's real, and it does make answers better over time — it's retrieval, not weight training.
- **"Many hidden layers"** is a from-scratch neural-net concept; it doesn't attach onto a system built on top of existing models like Claude or an open-weight local model. The actual lever for "smarter" here is the debate structure itself — Believer, Skeptic, and Judge forcing each other to be specific and cite evidence. That's where the real gains are, and it's fully built into the design below.
- **"Not predicting the next token"** isn't an option any current strong model has opted out of — Claude included. That's not a limitation a coding prompt fixes; the "actually smart" behavior comes from the reasoning structure wrapped around the model, not a different core mechanism.
- **"Local run"** auto-detects the machine's RAM/VRAM and picks a matching open-weight model to run through Ollama — it swaps in a *different, smaller model*, not a shrunk-down Claude. Claude itself is API-only.
- **"Best AI in the world"** — realistically, no, not from a coding prompt. A genuinely sharp tool that argues with you honestly instead of flattering your ideas is very buildable, though, and that's the actual target below.

"Can say no" and "can say it isn't certain" **are** both fully achievable, and they're built in structurally rather than left to hope: the Judge is required to allow a flat "no," and its confidence can never rate higher than the weakest input it relied on.

## What's in this doc

1. A file structure to give Opus as a starting skeleton
2. One prompt block, ready to paste into Opus 5.5, that builds it in order and checks in with you along the way

Claude Code is the smoothest way to run the prompt, since it can create the files and run commands itself as it goes — but it works as a plain chat prompt to Opus too, or pasted into Cursor if you point Cursor at Opus 5.5.

## File base

```
council-engine/
├── README.md
├── .env.example
├── requirements.txt
├── config/
│   ├── settings.yaml
│   └── personas/
│       ├── believer.md
│       ├── skeptic.md
│       ├── investor.md
│       └── judge.md
├── engine/
│   ├── router.py
│   ├── agents.py
│   ├── orchestrator.py
│   ├── uncertainty.py
│   └── memory/
│       ├── knowledge_store.py
│       └── research_loop.py
├── model_backend/
│   ├── hardware_detect.py
│   ├── local_runner.py
│   └── cloud_runner.py
├── data/
│   └── knowledge_base/        # gitignored, built at runtime
├── cli.py
└── tests/
```

## The prompt — paste this whole block into Opus 5.5

```
Build "council-engine": a CLI tool that takes any claim, idea, or argument and
evaluates it using four AI personas — Believer, Skeptic, Investor, Judge — then
improves its own knowledge over time through an autonomous web-research loop,
and can run either against the Anthropic API (cloud) or a local open-weight
model whose size is auto-picked from this machine's hardware.

SCOPE GUARDRAILS (read first)
- Don't invent a new model architecture or add custom neural-network layers
  anywhere. Every persona's "reasoning" is one LLM call — cloud via the
  Anthropic API, or local via Ollama — with a specific system prompt. That's
  the whole intelligence mechanism in this codebase.
- "Self-training" means an autonomous loop that researches topics on the web,
  summarizes findings, and stores them in a local vector database (RAG) that
  future sessions retrieve from — not weight updates to a model.
- "Local run" means picking an appropriately-sized open-weight model via
  Ollama based on detected RAM/VRAM. Claude is never run locally — it's
  API-only, used as the cloud backend.
- Build and show me each numbered step working before starting the next one.
  Stop after step 11 and wait for me before touching the stretch goal.

TECH STACK
- Python 3.11+
- `anthropic` SDK (Messages API) for the cloud backend. Model IDs:
  claude-opus-5-5 for the Judge's final synthesis, claude-sonnet-5 for the
  higher-volume Believer/Skeptic/Investor calls — both configurable in
  settings.yaml.
- Ollama's local HTTP API (http://localhost:11434) for the local backend.
  Assume Ollama is installed separately; pick reasonable current model tags
  for each hardware tier and note your choices in the README so they're easy
  to swap later.
- chromadb for the local knowledge base (file-based, no server needed)
- psutil for RAM/CPU detection; try torch.cuda for VRAM if torch is present,
  else shell out to `nvidia-smi --query-gpu=memory.total --format=csv,noheader`,
  else assume CPU-only
- argparse for the CLI, python-dotenv for the API key

CONFIG SHAPE (illustrative — adjust as needed)
backend:
  prefer: cloud   # cloud | local | auto
cloud:
  judge_model: claude-opus-5-5
  panel_model: claude-sonnet-5
local:
  tiers:
    tiny:   {max_ram_gb: 8,    ollama_model: "<pick a ~3B-class tag>"}
    small:  {max_ram_gb: 16,   ollama_model: "<pick a ~7-8B-class tag>"}
    medium: {max_ram_gb: 32,   ollama_model: "<pick a ~13-14B-class tag>"}
    large:  {max_ram_gb: 9999, ollama_model: "<pick a ~30B+-class tag>"}
research:
  max_calls_per_hour: 6
  seed_topics: []

FILE STRUCTURE
council-engine/
├── README.md
├── .env.example
├── requirements.txt
├── config/
│   ├── settings.yaml
│   └── personas/{believer,skeptic,investor,judge}.md
├── engine/
│   ├── router.py
│   ├── agents.py
│   ├── orchestrator.py
│   ├── uncertainty.py
│   └── memory/{knowledge_store.py,research_loop.py}
├── model_backend/{hardware_detect.py,local_runner.py,cloud_runner.py}
├── data/knowledge_base/   (gitignored)
├── cli.py
└── tests/

BUILD ORDER

1. model_backend/hardware_detect.py — detect total RAM, VRAM (0 if none), CPU
   core count. Map to a tier using thresholds read from settings.yaml, not
   hardcoded. Return the tier plus the raw numbers. Wire up `python cli.py
   doctor` to print this.

2. model_backend/local_runner.py and cloud_runner.py — both expose
   `generate(system: str, prompt: str) -> dict` returning at least {"text",
   "backend", "model"}. local_runner picks the Ollama tag for the detected
   tier from settings.yaml. cloud_runner calls the Anthropic Messages API
   with the configured model. Which backend is primary is a config flag,
   default: cloud if ANTHROPIC_API_KEY is set and reachable, else local.

3. config/personas/*.md — write these four files with exactly this content:

--- believer.md ---
You are the Believer on this council. Given a claim, build the strongest,
most specific case FOR it — steelman it. Find the best available evidence,
the soundest reasoning, and the clearest mechanism by which it would work,
and suggest how it could be extended or strengthened. This is advocacy, not
balance — the Skeptic supplies the other side — but stay accurate: vague
enthusiasm isn't a strong case, specifics are.
Respond as JSON: {"position": <2-4 sentences>, "key_points": [<specific
supporting points>], "confidence": "High"|"Medium"|"Low",
"what_would_change_my_mind": <one sentence>}

--- skeptic.md ---
You are the Skeptic on this council. Attack the claim as rigorously as its
toughest, best-informed critic would, so anything that survives you is
actually airtight. Look for unstated assumptions, missing or weak evidence,
logical gaps, stronger alternatives, edge cases, and second-order
consequences. Attack the argument, never the person. Name the exact weak
point — "this might not work" doesn't count, "the cost estimate assumes X,
which isn't shown" does.
Respond as JSON: {"position": <2-4 sentences>, "key_points": [<specific
objections>], "confidence": "High"|"Medium"|"Low" (how damaging your
strongest objection is), "what_would_change_my_mind": <one sentence>}

--- investor.md ---
You are the Investor on this council. You're only invoked when the router
flags this claim as a monetizable idea. Evaluate it the way a sharp,
skeptical early-stage investor would: market size, willingness to pay,
competition, unit economics, and what would have to be true for this to be
a good bet. Give concrete estimates where you can and say plainly when a
number is a guess.
Respond as JSON: {"position": <2-4 sentences>, "key_points": [<specific
financial/market points>], "confidence": "High"|"Medium"|"Low",
"what_would_change_my_mind": <the single number or fact that would most
change your view>}

--- judge.md ---
You are the Judge on this council. You receive the Believer's, Skeptic's,
and (if present) Investor's cases for one claim. Synthesize a final, honest
verdict — don't average or split the difference by default. Name where the
sides actually disagree on facts, not just emphasis, and say which has the
stronger case and why. A flat "no, this doesn't hold up" is exactly as
valid an output as "yes" — never manufacture a positive spin the evidence
doesn't support. Your confidence can never rate higher than the lowest
confidence among the inputs you relied on most.
Respond as JSON: {"verdict": <the actual answer, in plain language>,
"reasoning": <2-4 sentences citing specific points from each side>,
"confidence": "High"|"Medium"|"Low", "unresolved": <anything the sides
couldn't reconcile, or null>}

4. engine/router.py — one cheap LLM call that takes the input claim and
   returns {"is_money_idea": bool, "reason": "<one line>"}. This decides
   whether Investor runs.

5. engine/agents.py — an Agent class: persona name (loads the matching .md
   as system prompt) + backend module + `.respond(claim, context="") ->
   dict`, parsing the persona's JSON reply. If parsing fails, retry once
   with an explicit "reply with valid JSON only" nudge before surfacing the
   raw text.

6. engine/orchestrator.py — runs Believer and Skeptic always, Investor only
   if the router flagged it, then passes everything to Judge. Expose one
   clean function: `evaluate(claim: str) -> dict`, not just a CLI wrapper —
   I'll want to call this directly from other code later.

7. engine/uncertainty.py — after getting a Judge verdict, re-run the
   council once more (temperature > 0) and compare. If the two verdicts
   materially disagree — different confidence tier, or the position flips —
   mark the final result "Low confidence: the council didn't agree with
   itself on a repeat run" and show both instead of silently picking one.

8. engine/memory/knowledge_store.py — thin chromadb wrapper: add(text,
   metadata), search(query, k=5). Persist under data/knowledge_base/.

9. engine/memory/research_loop.py — use the Anthropic API's built-in
   web_search server tool for this (not a separate search API/key). Call
   the Messages API with web_search enabled and a prompt like "Research
   <topic>. Give me 5-8 sourced findings with URLs," then store each
   finding with its source URL and a timestamp. Pull topics from
   seed_topics in settings.yaml plus recurring subjects from past council
   sessions (a simple frequency count is fine for v1). Cap it with
   max_calls_per_hour in settings.yaml so it can't run up the API bill
   unattended. `python cli.py research --hours N` runs this for N hours (or
   until Ctrl-C), logging what it stores as it goes.

10. Wire retrieval into orchestrator.py: before running the council, search
    the knowledge base for anything relevant to the claim and pass the top
    few hits into each persona's context, so stored research actually gets
    used, not just stored.

11. cli.py — `ask "<claim>"` (full council run, pretty-printed: each
    persona's position + confidence, then the Judge's verdict, then overall
    confidence), `doctor` (hardware tier + active backend/model), `research
    --hours N`.

Show me `ask` working end-to-end on a test claim before building
research_loop.py or uncertainty.py.

STRETCH GOAL (only if I ask for it later — don't build by default)
LoRA fine-tuning of the local model on accumulated council transcripts, for
whoever has a GPU and wants the local model to actually adapt over time.
Separate module; never applies to the cloud/Claude backend.
```

## After v1 works

- If the research loop starts making a lot of calls, look at the Message Batches API (roughly half the cost for async, non-interactive calls) and prompt caching for the four persona system prompts, since they're reused on every single call: https://docs.claude.com/en/api/overview
- Since this is the evaluation stage of your bigger pipeline (idea generator → this council → coding agent → revenue tracking), the clean interface is `orchestrator.evaluate(claim: str) -> dict` — have your other agents import and call that directly rather than shelling out to the CLI once it's evaluating well.
- The LoRA stretch goal at the end of the prompt is the closest legitimate version of "it trains itself" if you want to go further later — it's real, it's just scoped to the small local model, on a GPU you own, and it's meaningfully more work than everything else here.
