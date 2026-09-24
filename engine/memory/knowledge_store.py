"""
Thin wrapper around a local Chroma vector store, persisted under
data/knowledge_base/.

Build brief: docs/build-brief.md, step 8.
"""

from __future__ import annotations


def add(text: str, metadata: dict) -> None:
    raise NotImplementedError


def search(query: str, k: int = 5) -> list:
    raise NotImplementedError
