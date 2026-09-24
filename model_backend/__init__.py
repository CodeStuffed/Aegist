"""Model backends: local (Ollama) and cloud (Anthropic API) inference.

Both runner modules expose the same surface, so callers treat "a backend"
as any object with:
    NAME: str
    generate(system, prompt, *, role, temperature, json_mode) -> {"text", "backend", "model", ...}
    model_for(role) -> str
"""

from __future__ import annotations

import os

from config import load_settings
from model_backend import cloud_runner, local_runner

CHOICES = ("auto", "cloud", "local")


class BackendUnavailable(RuntimeError):
    pass


def select_backend(prefer: str | None = None, settings: dict | None = None):
    """Pick the runner module to use.

    Precedence: explicit `prefer` > COUNCIL_BACKEND env var > settings.yaml
    backend.prefer. "auto" means cloud if ANTHROPIC_API_KEY is set and the
    API answers, else local.
    """
    settings = settings if settings is not None else load_settings()
    prefer = prefer or os.environ.get("COUNCIL_BACKEND") or settings["backend"].get("prefer", "auto")
    if prefer not in CHOICES:
        raise BackendUnavailable(f"Unknown backend {prefer!r}; expected one of {CHOICES}")
    if prefer == "local":
        return local_runner
    if prefer == "cloud":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise BackendUnavailable(
                "Backend is set to 'cloud' but ANTHROPIC_API_KEY isn't set (see .env.example)."
            )
        return cloud_runner
    available, _ = cloud_runner.is_available(settings)
    return cloud_runner if available else local_runner
