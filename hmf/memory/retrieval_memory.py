"""
Retrieval Memory — semantic access to past experiences and external knowledge.

Backed by a lightweight in-process vector store (numpy cosine).
Scores entries with a weighted combination of:
    relevance  (embedding similarity)
    recency    (exponential decay)
    importance (utility-weighted access frequency)

The MPC controller decides *when* to retrieve vs. use cache/skill,
and can trigger eviction of low-scoring entries.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..config import RetrievalMemoryConfig
from .base import MemoryEntry, MemorySubstrate


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class RetrievalMemory(MemorySubstrate):
    """
    Semantic vector store with importance-weighted scoring and
    time-decay eviction.
    """

    def __init__(
        self,
        config: RetrievalMemoryConfig,
        embed_fn: Callable[[str], List[float]],
        persist_dir: Optional[str] = None,
    ):
        self.cfg = config
        self.embed_fn = embed_fn
        self.persist_dir = persist_dir

        self._entries: Dict[str, MemoryEntry] = {}
        self._load()

    def read(self, query: str, top_k: int = 0, **kw) -> List[MemoryEntry]:
        if not self._entries:
            return []
        top_k = top_k or self.cfg.top_k
        q_emb = self.embed_fn(query)
        now = time.time()

        scored: List[tuple] = []
        for entry in self._entries.values():
            if entry.embedding is None:
                continue
            relevance = _cosine(q_emb, entry.embedding)
            if relevance < self.cfg.relevance_threshold:
                continue

            age_hours = (now - entry.timestamp) / 3600.0
            recency = self.cfg.decay_factor ** age_hours

            freq = min(entry.access_count / 10.0, 1.0)
            importance = entry.importance * (0.5 + 0.5 * freq)

            score = (
                self.cfg.importance_weight * importance
                + self.cfg.recency_weight * recency
                + self.cfg.frequency_weight * relevance
            )
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for entry, _ in scored[:top_k]:
            entry.touch()
            results.append(entry)
        return results

    def write(self, entry: MemoryEntry) -> bool:
        if entry.token_count == 0:
            entry.token_count = _estimate_tokens(entry.content)
        if entry.embedding is None:
            entry.embedding = self.embed_fn(entry.content)

        if len(self._entries) >= self.cfg.max_documents:
            self._evict_lowest()

        self._entries[entry.key] = entry
        self._persist()
        return True

    def write_experience(
        self,
        task_goal: str,
        trajectory: str,
        success: bool,
        goal_type: str = "",
        metadata: Optional[Dict] = None,
    ) -> MemoryEntry:
        content = f"Goal: {task_goal}\nOutcome: {'SUCCESS' if success else 'FAILURE'}\n{trajectory}"
        entry = MemoryEntry(
            key=f"exp_{int(time.time()*1000)}_{hash(task_goal) % 10000}",
            content=content,
            importance=1.0 if success else 0.4,
            source="experience",
            metadata={
                "task_goal": task_goal,
                "success": success,
                "goal_type": goal_type,
                **(metadata or {}),
            },
        )
        self.write(entry)
        return entry

    def evict(self, key: str) -> bool:
        if key in self._entries:
            del self._entries[key]
            self._persist()
            return True
        return False

    def size(self) -> int:
        return len(self._entries)

    def token_footprint(self) -> int:
        return sum(e.token_count for e in self._entries.values())

    def stats(self) -> Dict[str, Any]:
        successes = sum(
            1 for e in self._entries.values()
            if e.metadata.get("success", False)
        )
        return {
            "type": "retrieval",
            "entries": self.size(),
            "token_footprint": self.token_footprint(),
            "success_entries": successes,
            "failure_entries": self.size() - successes,
        }

    def _evict_lowest(self):
        """Remove the entry with the lowest combined score."""
        if not self._entries:
            return
        now = time.time()
        worst_key, worst_score = None, float("inf")
        for key, entry in self._entries.items():
            age_hours = (now - entry.timestamp) / 3600.0
            recency = self.cfg.decay_factor ** age_hours
            score = entry.importance * recency * (1 + entry.access_count * 0.1)
            if score < worst_score:
                worst_score = score
                worst_key = key
        if worst_key:
            del self._entries[worst_key]

    def _persist(self):
        if not self.persist_dir:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        path = os.path.join(self.persist_dir, "retrieval_store.json")
        data = []
        for e in self._entries.values():
            data.append({
                "key": e.key,
                "content": e.content,
                "timestamp": e.timestamp,
                "access_count": e.access_count,
                "importance": e.importance,
                "source": e.source,
                "metadata": e.metadata,
                "token_count": e.token_count,
            })
        with open(path, "w") as f:
            json.dump(data, f)

    def _load(self):
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, "retrieval_store.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for d in data:
                entry = MemoryEntry(
                    key=d["key"],
                    content=d["content"],
                    timestamp=d.get("timestamp", time.time()),
                    access_count=d.get("access_count", 0),
                    importance=d.get("importance", 1.0),
                    source=d.get("source", ""),
                    metadata=d.get("metadata", {}),
                    token_count=d.get("token_count", 0),
                )
                entry.embedding = self.embed_fn(entry.content)
                self._entries[entry.key] = entry
        except Exception as e:
            print(f"[RetrievalMemory] load error: {e}")
