"""
Detects this machine's RAM and CPU core count and maps RAM to a hardware
tier (tiny/small/medium/large) using the thresholds in config/settings.yaml.
The tier decides how big a NEW model is: layers, width, context length.

The model trains on the CPU with NumPy, so GPU memory doesn't matter here.

Build brief: docs/build-brief.md, step 1.
"""

from __future__ import annotations

import os
from functools import lru_cache

import psutil

from config import load_settings

GIB = 1024**3
SIZE_KEYS = ("n_layer", "n_head", "d_model", "block_size", "vocab_size")


def pick_tier(memory_gb: float, tiers: dict) -> str:
    """First tier (smallest max_ram_gb first) that this much memory fits under.

    Anything larger than every threshold gets the biggest tier.
    """
    if not tiers:
        raise ValueError("settings.yaml has no model.tiers configured")
    ordered = sorted(tiers.items(), key=lambda kv: kv[1]["max_ram_gb"])
    for name, spec in ordered:
        if memory_gb <= spec["max_ram_gb"]:
            return name
    return ordered[-1][0]


@lru_cache(maxsize=1)
def _measure() -> tuple[float, int]:
    """The raw numbers don't change while the process runs, so probe once."""
    ram_gb = psutil.virtual_memory().total / GIB
    cpu_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    return ram_gb, cpu_cores


def detect_hardware(settings: dict | None = None) -> dict:
    """Return {"tier", "ram_gb", "cpu_cores", "model_size": {n_layer, ...}}."""
    settings = settings if settings is not None else load_settings()
    ram_gb, cpu_cores = _measure()
    tiers = settings["model"]["tiers"]
    tier = pick_tier(ram_gb, tiers)
    return {
        "tier": tier,
        "ram_gb": round(ram_gb, 1),
        "cpu_cores": cpu_cores,
        "model_size": {k: tiers[tier][k] for k in SIZE_KEYS},
    }
