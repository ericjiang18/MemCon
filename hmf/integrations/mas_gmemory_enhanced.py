"""
G-Memory Enhanced — Inherits G-Memory directly and adds two lightweight layers:

  1. Retrieval-level cache: caches the OUTPUT of retrieve_memory() keyed by
     goal-type prefix. When a similar task appears, if cache hits, skip the
     expensive Chroma + graph lookup entirely.

  2. Success-pattern shortcut: for each goal-type (clean/heat/put/cool/examine),
     tracks the most recent successful action sequence. Injects this directly
     as a "recommended plan" into the retrieval output, giving the agent a
     concrete action template in addition to G-Memory's trajectory + insights.

Design principles (learned from v1/v2 failures):
  - ZERO extra LLM calls (no refine, no skill distillation)
  - ZERO changes to save_task_context logic (G-Memory's add_memory is already good)
  - Only add value during RETRIEVAL (make it faster + richer)
  - Keep G-Memory's graph + insights + backward scoring 100% intact
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.GMemory import GMemory
from mas.memory.common import MASMessage


def _extract_goal_type(task: str) -> str:
    """Extract the verb/goal-type from an ALFWorld task description."""
    task_lower = task.lower()
    for verb in ("clean", "heat", "cool", "examine", "look", "put", "puttwo"):
        if verb in task_lower:
            return verb
    return "other"


@dataclass
class GMemoryEnhanced(GMemory):
    """
    G-Memory + retrieval cache + success-pattern shortcuts.
    Inherits ALL of G-Memory's graph/insight/backward logic unchanged.
    """

    def __post_init__(self):
        super().__post_init__()

        self._retrieval_cache: Dict[str, tuple] = {}
        self._cache_ttl = 3
        self._cache_uses: Dict[str, int] = defaultdict(int)

        self._success_patterns: Dict[str, Dict[str, Any]] = {}
        self._pattern_path = os.path.join(self.persist_dir, "success_patterns.json")
        self._load_patterns()

        self._current_step_actions: List[str] = []

        print(f"[GMemoryEnhanced] initialized (cache + success patterns on top of G-Memory)")

    # ============== Step tracking ==============

    def move_memory_state(self, action: str, observation: str, **kwargs):
        super().move_memory_state(action, observation, **kwargs)
        is_think = action.lower().startswith(("think", "thought"))
        if not is_think:
            self._current_step_actions.append(action)

    # ============== Enhanced save: record success patterns ==============

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        task_main = self.current_task_context.task_main
        goal_type = _extract_goal_type(task_main)

        if label and self._current_step_actions:
            self._success_patterns[goal_type] = {
                "task": task_main,
                "actions": list(self._current_step_actions[:20]),
                "steps": len(self._current_step_actions),
                "timestamp": time.time(),
            }
            self._save_patterns()

            cache_key = self._goal_cache_key(goal_type)
            self._retrieval_cache.pop(cache_key, None)

        self._current_step_actions = []

        return super().save_task_context(label=label, feedback=feedback)

    # ============== Enhanced retrieval ==============

    def retrieve_memory(
        self,
        query_task: str = "",
        successful_topk: int = 2,
        failed_topk: int = 1,
        insight_topk: int = 10,
        threshold: float = 0.3,
        **kwargs,
    ) -> tuple:
        goal_type = _extract_goal_type(query_task)
        cache_key = self._goal_cache_key(goal_type)

        # Check retrieval cache (skip expensive Chroma + graph lookup)
        if cache_key in self._retrieval_cache:
            cached = self._retrieval_cache[cache_key]
            self._cache_uses[cache_key] += 1
            if self._cache_uses[cache_key] <= self._cache_ttl:
                successful, failed, insights = cached
                insights = list(insights)
                pattern_hint = self._get_pattern_hint(goal_type, query_task)
                if pattern_hint:
                    insights.insert(0, pattern_hint)
                return successful, failed, insights

            del self._retrieval_cache[cache_key]
            self._cache_uses[cache_key] = 0

        # Cache miss → full G-Memory retrieval
        result = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk,
            failed_topk=failed_topk,
            insight_topk=insight_topk,
            threshold=threshold,
            **kwargs,
        )

        successful = result[0]
        failed = result[1]
        insights = list(result[2])

        # Cache the result for same goal-type reuse
        self._retrieval_cache[cache_key] = (successful, failed, insights)
        self._cache_uses[cache_key] = 1

        # Inject success-pattern shortcut
        pattern_hint = self._get_pattern_hint(goal_type, query_task)
        if pattern_hint:
            insights.insert(0, pattern_hint)

        return (successful, failed, insights) + result[3:] if len(result) > 3 else (successful, failed, insights)

    # ============== Success pattern hint ==============

    def _get_pattern_hint(self, goal_type: str, current_task: str) -> str:
        pattern = self._success_patterns.get(goal_type)
        if not pattern:
            return ""

        actions = pattern["actions"][:10]
        steps_str = "\n".join(f"    {i+1}. {a}" for i, a in enumerate(actions))

        return (
            f"[SUCCESS PATTERN for '{goal_type}' tasks — {pattern['steps']} steps, "
            f"from similar task: {pattern['task'][:80]}]\n"
            f"  Recommended action sequence:\n{steps_str}\n"
            f"  Adapt object names to your current task."
        )

    # ============== Helpers ==============

    @staticmethod
    def _goal_cache_key(goal_type: str) -> str:
        return f"goal_{goal_type}"

    def _save_patterns(self):
        os.makedirs(os.path.dirname(self._pattern_path) or self.persist_dir, exist_ok=True)
        with open(self._pattern_path, "w") as f:
            json.dump(self._success_patterns, f, indent=2)

    def _load_patterns(self):
        if os.path.exists(self._pattern_path):
            try:
                with open(self._pattern_path) as f:
                    self._success_patterns = json.load(f)
            except Exception:
                pass
