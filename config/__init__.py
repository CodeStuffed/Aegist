"""
Settings loader. Everything tunable lives in config/settings.yaml; this
module just finds it, reads it, and knows where the project's data lives.

Environment overrides (handy for tests and for embedding the engine in
other code):
  COUNCIL_SETTINGS  - path to an alternate settings.yaml
  COUNCIL_DATA_DIR  - where runtime data (knowledge base, logs) is written
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PERSONA_DIR = CONFIG_DIR / "personas"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.yaml"

# Pick up ANTHROPIC_API_KEY (and friends) from a local .env. Existing
# environment variables win over the file.
load_dotenv(ROOT / ".env")


def settings_path() -> Path:
    return Path(os.environ.get("COUNCIL_SETTINGS", DEFAULT_SETTINGS_PATH))


def load_settings() -> dict:
    with open(settings_path(), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def data_dir(settings: dict | None = None) -> Path:
    """Root for everything written at runtime. Created on first use."""
    if "COUNCIL_DATA_DIR" in os.environ:
        path = Path(os.environ["COUNCIL_DATA_DIR"])
    else:
        settings = settings if settings is not None else load_settings()
        path = ROOT / settings.get("paths", {}).get("data_dir", "data")
    path.mkdir(parents=True, exist_ok=True)
    return path
