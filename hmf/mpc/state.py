"""
MPC State — tracks the joint state of all memory substrates
and the running budget consumption for the current task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..memory.base import MemoryActionType


@dataclass
class StepRecord:
    step: int
    action_type: MemoryActionType
    query: str = ""
    result_summary: str = ""
    token_cost: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    skill_used: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class MPCState:
    """
    Observable state visible to the MPC controller at each decision point.
    Encapsulates task progress, memory substrate states, and budget usage.
    """

    task_goal: str = ""
    task_type: str = ""
    current_observation: str = ""
    step_number: int = 0

    # Budget tracking
    tokens_used: int = 0
    tokens_budget: int = 8000
    latency_used_ms: float = 0.0
    latency_budget_ms: float = 30000.0

    # Memory substrate snapshots
    cache_size: int = 0
    cache_hit_rate: float = 0.0
    cache_token_footprint: int = 0
    retrieval_size: int = 0
    retrieval_token_footprint: int = 0
    skill_count: int = 0
    skill_avg_sr: float = 0.0
    trajectory_buffer_size: int = 0

    # Recent action history (for the controller to avoid repetition)
    recent_actions: List[MemoryActionType] = field(default_factory=list)
    step_history: List[StepRecord] = field(default_factory=list)

    # Task-level signals
    consecutive_failures: int = 0
    uncertainty: float = 0.5
    task_progress: float = 0.0

    @property
    def token_headroom(self) -> int:
        return max(0, self.tokens_budget - self.tokens_used)

    @property
    def latency_headroom_ms(self) -> float:
        return max(0.0, self.latency_budget_ms - self.latency_used_ms)

    @property
    def budget_fraction_used(self) -> float:
        token_frac = self.tokens_used / max(self.tokens_budget, 1)
        latency_frac = self.latency_used_ms / max(self.latency_budget_ms, 1.0)
        return max(token_frac, latency_frac)

    def record_step(self, rec: StepRecord):
        self.step_history.append(rec)
        self.tokens_used += rec.token_cost
        self.latency_used_ms += rec.latency_ms
        self.recent_actions.append(rec.action_type)
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        self.step_number += 1

    def to_prompt_context(self) -> str:
        """Serialize state into a compact string for the LLM-based MPC scorer."""
        lines = [
            f"Task: {self.task_goal}",
            f"Step: {self.step_number}, Progress: {self.task_progress:.0%}",
            f"Tokens: {self.tokens_used}/{self.tokens_budget} ({self.token_headroom} left)",
            f"Latency: {self.latency_used_ms:.0f}/{self.latency_budget_ms:.0f}ms",
            f"Cache: {self.cache_size} entries, hit_rate={self.cache_hit_rate:.0%}",
            f"Retrieval: {self.retrieval_size} docs",
            f"Skills: {self.skill_count} (avg SR={self.skill_avg_sr:.0%})",
            f"Uncertainty: {self.uncertainty:.2f}",
            f"Consecutive failures: {self.consecutive_failures}",
        ]
        if self.recent_actions:
            recent = [a.name for a in self.recent_actions[-5:]]
            lines.append(f"Recent actions: {', '.join(recent)}")
        return "\n".join(lines)

    def snapshot_memory_stats(self, cache_stats, retrieval_stats, skill_stats):
        self.cache_size = cache_stats.get("entries", 0)
        self.cache_hit_rate = cache_stats.get("hit_rate", 0.0)
        self.cache_token_footprint = cache_stats.get("token_footprint", 0)
        self.retrieval_size = retrieval_stats.get("entries", 0)
        self.retrieval_token_footprint = retrieval_stats.get("token_footprint", 0)
        self.skill_count = skill_stats.get("active_skills", 0)
        self.skill_avg_sr = skill_stats.get("avg_success_rate", 0.0)
        self.trajectory_buffer_size = skill_stats.get("trajectory_buffer", 0)
