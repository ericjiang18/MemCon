"""
Base interfaces for memory substrates.

Every substrate exposes read / write / evict plus stats collection
so the MPC controller can reason uniformly across heterogeneous stores.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class MemoryActionType(Enum):
    CACHE_READ = auto()
    CACHE_WRITE = auto()
    RETRIEVE = auto()
    RETRIEVE_WRITE = auto()
    SKILL_INVOKE = auto()
    SKILL_CONSOLIDATE = auto()
    EVICT_CACHE = auto()
    EVICT_RETRIEVAL = auto()
    EVICT_SKILL = auto()
    LLM_GENERATE = auto()
    NO_OP = auto()


@dataclass
class MemoryAction:
    action_type: MemoryActionType
    query: str = ""
    payload: Any = None
    estimated_token_cost: int = 0
    estimated_latency_ms: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    key: str
    content: str
    embedding: Optional[List[float]] = None
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 1.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def touch(self):
        self.access_count += 1
        self.timestamp = time.time()


class MemorySubstrate(ABC):
    """Common interface that every memory layer must implement."""

    @abstractmethod
    def read(self, query: str, top_k: int = 5, **kw) -> List[MemoryEntry]:
        ...

    @abstractmethod
    def write(self, entry: MemoryEntry) -> bool:
        ...

    @abstractmethod
    def evict(self, key: str) -> bool:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def token_footprint(self) -> int:
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        ...
