"""
Detects this machine's RAM, VRAM, and CPU core count, and maps them to a
hardware tier (tiny/small/medium/large) using the thresholds in
config/settings.yaml. Used by `python cli.py doctor` and by
model_backend/local_runner.py to pick which local model to run.

Build brief: docs/build-brief.md, step 1.
"""

from __future__ import annotations


def detect_hardware() -> dict:
    """Return {"tier": str, "ram_gb": float, "vram_gb": float, "cpu_cores": int}."""
    raise NotImplementedError
