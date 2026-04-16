"""
Cache Memory — fast reuse of recent computations and tool outputs.

Uses a two-tier lookup:
  1. Exact hash match for deterministic cache hits
  2. Embedding cosine similarity for soft/semantic matches

Eviction policy: LRU with TTL expiry.  The MPC controller can also
trigger explicit eviction when the token budget is tight.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..config import CacheMemoryConfig
from .base import MemoryEntry, MemorySubstrate


def _cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class CacheMemory(MemorySubstrate):
    """
    LRU + TTL cache with exact-hash and semantic-similarity lookup.
    """

    def __init__(
        self,
        config: CacheMemoryConfig,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.cfg = config
        self.embed_fn = embed_fn

        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._hash_index: Dict[str, str] = {}

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def read(self, query: str, top_k: int = 1, **kw) -> List[MemoryEntry]:
        self._expire()

        h = self._content_hash(query)
        if h in self._hash_index:
            key = self._hash_index[h]
            if key in self._store:
                entry = self._store[key]
                entry.touch()
                self._store.move_to_end(key)
                self._hits += 1
                return [entry]

        if self.embed_fn is not None:
            q_emb = self.embed_fn(query)
            scored: List[tuple] = []
            for entry in self._store.values():
                if entry.embedding is not None:
                    sim = _cosine(q_emb, entry.embedding)
                    if sim >= self.cfg.semantic_threshold:
                        scored.append((entry, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored:
                results = []
                for entry, _ in scored[:top_k]:
                    entry.touch()
                    results.append(entry)
                self._hits += 1
                return results

        self._misses += 1
        return []

    def write(self, entry: MemoryEntry) -> bool:
        if entry.token_count == 0:
            entry.token_count = _estimate_tokens(entry.content)
        if entry.token_count > self.cfg.max_tokens_per_entry:
            return False

        if self.embed_fn is not None and entry.embedding is None:
            entry.embedding = self.embed_fn(entry.content)

        while len(self._store) >= self.cfg.max_entries:
            self._evict_lru()

        self._store[entry.key] = entry
        self._store.move_to_end(entry.key)
        h = self._content_hash(entry.content)
        self._hash_index[h] = entry.key
        return True

    def evict(self, key: str) -> bool:
        if key not in self._store:
            return False
        entry = self._store.pop(key)
        h = self._content_hash(entry.content)
        self._hash_index.pop(h, None)
        self._evictions += 1
        return True

    def size(self) -> int:
        return len(self._store)

    def token_footprint(self) -> int:
        return sum(e.token_count for e in self._store.values())

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "type": "cache",
            "entries": self.size(),
            "token_footprint": self.token_footprint(),
            "hit_rate": self._hits / max(total, 1),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    def _expire(self):
        now = time.time()
        expired = [
            k for k, e in self._store.items()
            if (now - e.timestamp) > self.cfg.ttl_seconds
        ]
        for k in expired:
            self.evict(k)

    def _evict_lru(self):
        if not self._store:
            return
        key, _ = self._store.popitem(last=False)
        self.evict(key) if key in self._store else None
        self._evictions += 1

    def clear(self):
        self._store.clear()
        self._hash_index.clear()
