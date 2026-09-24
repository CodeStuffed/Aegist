"""
Detects this machine's RAM, VRAM, and CPU core count, and maps them to a
hardware tier (tiny/small/medium/large) using the thresholds in
config/settings.yaml. Used by `python cli.py doctor` and by
model_backend/local_runner.py to pick which local model to run.

Build brief: docs/build-brief.md, step 1.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import psutil

from config import load_settings

GIB = 1024**3


def _vram_from_torch() -> float | None:
    try:
        import torch  # optional; only present if the user installed it
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        total = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return None
    return total / GIB


def parse_nvidia_smi(output: str) -> float:
    """Sum `memory.total` lines (MiB, e.g. "24576 MiB") across all GPUs, in GiB."""
    total_mib = 0.0
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        total_mib += float(line.split()[0])
    return total_mib / 1024


def _vram_from_nvidia_smi() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return parse_nvidia_smi(out)
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


def detect_vram() -> tuple[float, str]:
    """Return (total VRAM in GiB, where the number came from)."""
    vram = _vram_from_torch()
    if vram is not None:
        return vram, "torch"
    vram = _vram_from_nvidia_smi()
    if vram is not None:
        return vram, "nvidia-smi"
    return 0.0, "none (CPU-only)"


def pick_tier(memory_gb: float, tiers: dict) -> str:
    """First tier (smallest max_ram_gb first) that this much memory fits under.

    Anything larger than every threshold gets the biggest tier.
    """
    if not tiers:
        raise ValueError("settings.yaml has no local.tiers configured")
    ordered = sorted(tiers.items(), key=lambda kv: kv[1]["max_ram_gb"])
    for name, spec in ordered:
        if memory_gb <= spec["max_ram_gb"]:
            return name
    return ordered[-1][0]


def detect_hardware(settings: dict | None = None) -> dict:
    """Return {"tier": str, "ram_gb": float, "vram_gb": float, "cpu_cores": int, ...}."""
    settings = settings if settings is not None else load_settings()
    ram_gb = psutil.virtual_memory().total / GIB
    vram_gb, vram_source = detect_vram()
    cpu_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    # A model has to fit somewhere: system RAM normally, or the GPU if it's
    # the bigger pool.
    tier = pick_tier(max(ram_gb, vram_gb), settings["local"]["tiers"])
    return {
        "tier": tier,
        "ram_gb": round(ram_gb, 1),
        "vram_gb": round(vram_gb, 1),
        "vram_source": vram_source,
        "cpu_cores": cpu_cores,
    }
