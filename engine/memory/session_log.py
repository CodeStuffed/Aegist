"""
A line per council session in data/sessions.jsonl. The research loop reads
it back to find subjects that keep coming up.
"""

from __future__ import annotations

import json
from collections import Counter

from config import data_dir


def _path(settings: dict | None = None):
    return data_dir(settings) / "sessions.jsonl"


def record(result: dict, settings: dict | None = None) -> None:
    entry = {
        "at": result["timestamp"],
        "claim": result["claim"],
        "keywords": result["routing"]["keywords"],
        "stance": result["judge"]["stance"],
        "confidence": result["overall_confidence"],
    }
    with open(_path(settings), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def sessions(settings: dict | None = None) -> list[dict]:
    path = _path(settings)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keyword_counts(settings: dict | None = None) -> Counter:
    """How often each keyword has appeared across all past claims."""
    return Counter(k for s in sessions(settings) for k in s.get("keywords", []))
