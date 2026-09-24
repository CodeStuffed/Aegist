"""
Cloud inference backend. Calls the Anthropic Messages API with the model
configured in config/settings.yaml (claude-opus-5-5 for the Judge,
claude-sonnet-5 for the panel, by default).

Build brief: docs/build-brief.md, step 2.
"""

from __future__ import annotations

import os
from functools import lru_cache

import anthropic

from config import load_settings

NAME = "cloud"


class CloudBackendError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY from the environment (config/ loads .env).
    return anthropic.Anthropic()


def model_for(role: str = "panel", settings: dict | None = None) -> str:
    settings = settings if settings is not None else load_settings()
    cloud = settings["cloud"]
    if role == "judge":
        return cloud["judge_model"]
    if role == "router":
        return cloud.get("router_model") or cloud["panel_model"]
    if role == "research":
        return settings["research"]["model"]
    return cloud["panel_model"]


def _effort_for(role: str, settings: dict) -> str | None:
    if role == "research":
        return settings["research"].get("effort")
    return settings["cloud"].get("effort", {}).get(role)


def generate(
    system: str,
    prompt: str,
    *,
    role: str = "panel",
    temperature: float | None = None,
    json_mode: bool = True,
    settings: dict | None = None,
) -> dict:
    """Return {"text": str, "backend": "cloud", "model": str}.

    `temperature` is accepted for interface parity with local_runner but not
    sent: claude-opus-5-5 and claude-sonnet-5 reject sampling parameters
    (every call is already sampled). `json_mode` is likewise a no-op here -
    the persona prompts ask for JSON and the caller parses it.
    """
    settings = settings if settings is not None else load_settings()
    model = model_for(role, settings)
    kwargs: dict = {
        "model": model,
        "max_tokens": settings["cloud"].get("max_tokens", 16000),
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    effort = _effort_for(role, settings)
    if effort:
        kwargs["output_config"] = {"effort": effort}

    try:
        response = client().messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise CloudBackendError("ANTHROPIC_API_KEY was rejected.") from e
    except anthropic.NotFoundError as e:
        raise CloudBackendError(f"Model {model!r} not found - check config/settings.yaml.") from e
    except anthropic.RateLimitError as e:
        raise CloudBackendError("Rate limited by the Anthropic API (after SDK retries).") from e
    except anthropic.APIStatusError as e:
        raise CloudBackendError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise CloudBackendError("Couldn't connect to the Anthropic API.") from e

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise CloudBackendError(f"{model} declined this request (category: {category}).")

    # Thinking blocks come back too (empty by default); only text is the answer.
    text = "".join(block.text for block in response.content if block.type == "text")
    return {
        "text": text.strip(),
        "backend": NAME,
        "model": model,
        "stop_reason": response.stop_reason,
    }


@lru_cache(maxsize=1)
def _probe(timeout: float) -> tuple[bool, str]:
    try:
        client().with_options(timeout=timeout, max_retries=0).models.list(limit=1)
    except anthropic.AuthenticationError:
        return False, "ANTHROPIC_API_KEY was rejected"
    except anthropic.APIConnectionError:
        return False, "Anthropic API unreachable"
    except anthropic.APIStatusError as e:
        return False, f"Anthropic API returned {e.status_code}"
    return True, "API key accepted"


def is_available(settings: dict | None = None) -> tuple[bool, str]:
    """(usable?, why). Cached for the life of the process."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not set"
    settings = settings if settings is not None else load_settings()
    return _probe(float(settings["cloud"].get("reachability_timeout_s", 5)))
