# council-engine

Evaluation-pipeline module for a larger business-agent system: a claim,
idea, or argument goes in, and comes out judged by four AI personas -
Believer, Skeptic, Investor, and Judge - backed by a self-researching,
hardware-aware model backend.

## Status: scaffold

Folder structure, module contracts (docstrings + function signatures), the
four persona prompts, and a starter config are all in place. The actual
logic inside each module still needs to be written.

**To implement it:** open this repo in Claude Code (or paste
`docs/build-brief.md` into a Claude Opus 5.5 chat) and follow the build
order there - it explains each module's job and what NOT to build (no
custom model architectures, no literal weight training).

## Layout

- `config/` - settings and the four persona system prompts
- `engine/` - routing, agents, orchestration, uncertainty checking, memory
- `model_backend/` - hardware detection and the local (Ollama) / cloud
  (Anthropic API) inference backends
- `cli.py` - `ask`, `doctor`, `research` commands
- `docs/build-brief.md` - full spec and the build prompt for Opus 5.5

## Setup (once implemented)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
python cli.py doctor
python cli.py ask "we should switch to usage-based pricing"
```
