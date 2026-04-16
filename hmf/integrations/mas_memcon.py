"""
MemCon (Memory as Controlled Process) — MASMemoryBase adapter.

Wraps G-Memory with the MemCon policy controller.
The policy learns ONLINE which retrieval parameters work best for each state.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.GMemory import GMemory
from mas.memory.common import MASMessage
from mas.llm import LLMCallable
from mas.utils import EmbeddingFunc

from ..mcp_framework.wrapper import MemConWrapper


@dataclass
class MemConMemory(MemConWrapper):
    """
    MemCon wrapping G-Memory.
    Instantiates G-Memory as the inner backend, then applies learned
    policy control on top.
    """

    def __post_init__(self):
        super().__post_init__()

        inner = GMemory(
            namespace="gmemory_inner",
            global_config=self.global_config,
            llm_model=self.llm_model,
            embedding_func=self.embedding_func,
        )
        self.set_inner(inner)

        print(f"[MemCon] wrapping G-Memory as inner backend")

    @property
    def memory_size(self):
        if self._inner:
            return self._inner.memory_size
        return 0
