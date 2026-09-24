"""
The local knowledge base: passages of text with metadata, searchable with
BM25 (the classic ranking function behind most keyword search engines),
implemented here from scratch. No embedding model, no outside AI.

Stored as one JSON object per line in data/knowledge_base/passages.jsonl,
loaded into an inverted index in memory on first use.

Build brief: docs/build-brief.md, step 8.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from config import data_dir
from engine.text import content_words

K1, B = 1.5, 0.75  # standard BM25 constants


class KnowledgeStore:
    def __init__(self, path: Path):
        self.path = path
        self.docs: list[dict] = []
        self.ids: set[str] = set()
        self.postings: dict[str, dict[int, int]] = {}  # term -> {doc index: term count}
        self.lengths: list[int] = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._index(json.loads(line))

    def _index(self, doc: dict) -> None:
        i = len(self.docs)
        self.docs.append(doc)
        self.ids.add(doc["id"])
        terms = Counter(content_words(doc["text"]))
        self.lengths.append(sum(terms.values()))
        for term, count in terms.items():
            self.postings.setdefault(term, {})[i] = count

    def add(self, text: str, metadata: dict | None = None) -> str | None:
        """Store a passage. Returns its id, or None if it was already stored."""
        text = " ".join(text.split())
        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        if not text or doc_id in self.ids:
            return None
        doc = {"id": doc_id, "text": text, "metadata": metadata or {}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self._index(doc)
        return doc_id

    def idf(self, term: str) -> float:
        n, df = len(self.docs), len(self.postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _reference_idf(self, term: str) -> float:
        """idf, except a word no passage contains counts as if one did -
        otherwise one unheard-of word would swamp the rest of the query."""
        n = len(self.docs)
        return self.idf(term) if term in self.postings else math.log(1 + (n - 0.5) / 1.5)

    def search(self, query: str, k: int = 5, min_relevance: float = 0.0) -> list[dict]:
        """Top-k passages by BM25. `relevance` is the score as a fraction of
        the best score this query could get, so one cutoff works whether
        the store holds ten passages or a million."""
        terms = set(content_words(query))
        if not self.docs or not terms:
            return []
        avg_len = sum(self.lengths) / len(self.lengths) or 1.0
        scores: Counter = Counter()
        for term in terms:
            idf = self.idf(term)
            for i, tf in self.postings.get(term, {}).items():
                norm = tf + K1 * (1 - B + B * self.lengths[i] / avg_len)
                scores[i] += idf * tf * (K1 + 1) / norm
        # A typical-length passage containing every query word once.
        best_possible = sum(self._reference_idf(t) for t in terms)
        hits = [(i, s, min(1.0, s / best_possible)) for i, s in scores.most_common(k)]
        return [{**self.docs[i], "score": round(s, 3), "relevance": round(r, 3)}
                for i, s, r in hits if r >= min_relevance]

    def count(self) -> int:
        return len(self.docs)


_stores: dict[str, KnowledgeStore] = {}


def store(settings: dict | None = None) -> KnowledgeStore:
    path = data_dir(settings) / "knowledge_base" / "passages.jsonl"
    key = str(path)
    if key not in _stores:
        _stores[key] = KnowledgeStore(path)
    return _stores[key]


def add(text: str, metadata: dict | None = None, settings: dict | None = None) -> str | None:
    return store(settings).add(text, metadata)


def search(query: str, k: int = 5, settings: dict | None = None, min_relevance: float = 0.0) -> list:
    return store(settings).search(query, k, min_relevance)


def count(settings: dict | None = None) -> int:
    return store(settings).count()
