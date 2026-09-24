"""
Byte-level BPE tokenizer, trained from scratch on the corpus.

Text is split into word-ish chunks, each chunk becomes its UTF-8 bytes
(ids 0-255, so any text at all can be encoded), and training repeatedly
merges the most frequent adjacent pair into a new token.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

PRETOKENIZE = re.compile(r" ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+")
BYTE_VOCAB = 256


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_VOCAB)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[BYTE_VOCAB + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "Tokenizer":
        """Learn merges until the vocabulary reaches `vocab_size` (or nothing repeats)."""
        words = [tuple(chunk.encode("utf-8")) for chunk in PRETOKENIZE.findall(text)]
        freqs = Counter(words)
        seqs = [list(w) for w in freqs]
        counts = list(freqs.values())

        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> word indexes
        for wi, seq in enumerate(seqs):
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += counts[wi]
                where[pair].add(wi)

        merges = []
        while BYTE_VOCAB + len(merges) < vocab_size and pair_counts:
            best, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_count < 2:
                break
            new_id = BYTE_VOCAB + len(merges)
            merges.append(best)
            for wi in list(where.pop(best, ())):
                seq, c = seqs[wi], counts[wi]
                for pair in zip(seq, seq[1:]):  # retract this word's old pairs
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                merged, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                seqs[wi] = merged
                for pair in zip(merged, merged[1:]):  # and add the new ones
                    pair_counts[pair] += c
                    where[pair].add(wi)
            pair_counts.pop(best, None)
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        seq = list(chunk.encode("utf-8"))
        while len(seq) > 1:
            ranked = [(self.ranks.get(pair, 1 << 30), i) for i, pair in enumerate(zip(seq, seq[1:]))]
            rank, i = min(ranked)
            if rank == 1 << 30:
                break
            seq[i:i + 2] = [BYTE_VOCAB + rank]
        if len(self._cache) < 200_000:
            self._cache[chunk] = seq
        return seq

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in PRETOKENIZE.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")
