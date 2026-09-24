"""
Local inference backend. Calls Ollama's HTTP API (http://localhost:11434)
using the model tag configured for the current hardware tier.

Build brief: docs/build-brief.md, step 2.
"""

from __future__ import annotations

import os
import re

import requests

from config import load_settings
from model_backend.hardware_detect import detect_hardware

NAME = "local"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalBackendError(RuntimeError):
    pass


def ollama_host(settings: dict | None = None) -> str:
    settings = settings if settings is not None else load_settings()
    host = os.environ.get("OLLAMA_HOST") or settings["local"].get("host", "http://localhost:11434")
    if "://" not in host:  # OLLAMA_HOST is often written as "127.0.0.1:11434"
        host = f"http://{host}"
    return host.rstrip("/")


def model_for(role: str = "panel", settings: dict | None = None) -> str:
    """Ollama tag to use. Every role shares one local model: the biggest one
    this machine's tier can hold."""
    settings = settings if settings is not None else load_settings()
    local = settings["local"]
    if local.get("model_override"):
        return local["model_override"]
    tier = detect_hardware(settings)["tier"]
    return local["tiers"][tier]["ollama_model"]


def generate(
    system: str,
    prompt: str,
    *,
    role: str = "panel",
    temperature: float | None = None,
    json_mode: bool = True,
    settings: dict | None = None,
) -> dict:
    """Return {"text": str, "backend": "local", "model": str}."""
    settings = settings if settings is not None else load_settings()
    local = settings["local"]
    host = ollama_host(settings)
    model = model_for(role, settings)

    options: dict = {"num_ctx": local.get("num_ctx", 8192)}
    if temperature is not None:
        options["temperature"] = temperature
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": options,
    }
    if json_mode:
        # Grammar-constrained decoding: small local models otherwise drift
        # out of JSON surprisingly often.
        payload["format"] = "json"
    if local.get("think") is not None:
        payload["think"] = local["think"]

    try:
        resp = requests.post(
            f"{host}/api/chat", json=payload, timeout=local.get("request_timeout_s", 600)
        )
    except requests.ConnectionError as e:
        raise LocalBackendError(
            f"Can't reach Ollama at {host}. Is it installed and running (`ollama serve`)?"
        ) from e
    except requests.Timeout as e:
        raise LocalBackendError(f"Ollama at {host} timed out running {model}.") from e

    if resp.status_code == 404:
        raise LocalBackendError(f"Ollama doesn't have {model}. Run: ollama pull {model}")
    if resp.status_code != 200:
        raise LocalBackendError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    text = _THINK_BLOCK.sub("", data.get("message", {}).get("content", "")).strip()
    return {
        "text": text,
        "backend": NAME,
        "model": model,
        "stop_reason": data.get("done_reason"),
    }


def status(settings: dict | None = None) -> dict:
    """For `doctor`: is Ollama up, and is the tier's model pulled?"""
    settings = settings if settings is not None else load_settings()
    host = ollama_host(settings)
    model = model_for("panel", settings)
    try:
        resp = requests.get(f"{host}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"host": host, "model": model, "reachable": False, "model_pulled": False,
                "detail": str(e.__class__.__name__)}
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    return {"host": host, "model": model, "reachable": True, "model_pulled": wanted in names,
            "detail": ""}
