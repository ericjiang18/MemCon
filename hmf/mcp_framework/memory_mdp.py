"""
Memory MDP — Formal definition.

We model the agent's memory operations as an MDP (M_mem):

  M_mem = (S, A, T, R, γ)

  S (State):  Joint state of task progress + memory system
    s = (s_task, s_mem) where:
      s_task = (goal_type, step_phase, is_stuck, objects_held, locations_visited, sub_goals_done)
      s_mem  = (mem_size, retrieval_hits_last, insight_count, insight_quality, plan_available)

  A (Actions): Memory operations with CONTINUOUS parameters
    a = (op, params) where:
      op ∈ {RETRIEVE, ENCODE, CONSOLIDATE, FORGET, PLAN_INJECT, RE_RETRIEVE, NO_OP}
      params = {top_k, insight_k, hop, query_rewrite, ...}

  T (Transition): Deterministic — executing memory op updates s_mem
  R (Reward):     Observed at task end: success(+1/-0.5) + efficiency_bonus
  γ (Discount):   Within-task discount for multi-step credit assignment

Key insight: Unlike prior work that treats memory actions as discrete
(retrieve or not), we control CONTINUOUS parameters (how many to retrieve,
how deep to search, which query to use). This is what makes the policy
learnable and the framework general.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class MemOp(Enum):
    """Memory operations the policy can choose."""
    RETRIEVE = auto()         # Query the underlying memory store
    ENCODE = auto()           # Write current experience to memory
    CONSOLIDATE = auto()      # Merge/abstract stored experiences
    FORGET = auto()           # Evict low-value entries
    PLAN_INJECT = auto()      # Inject a generalized success plan
    RE_RETRIEVE = auto()      # Re-query with modified parameters (when stuck)
    NO_OP = auto()


@dataclass
class MemoryAction:
    op: MemOp
    top_k: int = 2
    insight_k: int = 5
    hop: int = 1
    query_suffix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryState:
    """
    Observable state for the memory policy.
    Designed to be compact (hashable for tabular RL) yet informative.
    """
    # Task progress signals
    goal_type: str = "other"
    step_phase: str = "early"        # early (<8), mid (8-18), late (>18)
    is_stuck: bool = False           # repeated action detected
    objects_held: int = 0            # how many objects agent is holding
    locations_visited: int = 0       # unique locations explored
    sub_goals_done: int = 0          # for multi-goal tasks (puttwo)

    # Memory system signals
    mem_size: int = 0                # total entries in underlying store
    last_retrieval_useful: bool = True  # did last retrieval lead to progress?
    insight_count: int = 0
    insight_avg_score: float = 2.0
    plan_available: bool = False     # do we have a success plan for this goal type?
    task_index: int = 0              # which task number (learning progresses over time)

    def to_key(self) -> str:
        """Discretize into hashable key for tabular policy."""
        return (
            f"{self.goal_type}|"
            f"{self.step_phase}|"
            f"stuck={self.is_stuck}|"
            f"hold={min(self.objects_held, 2)}|"
            f"visited={min(self.locations_visited // 3, 4)}|"
            f"mem={min(self.mem_size // 10, 5)}|"
            f"plan={self.plan_available}|"
            f"phase={'warm' if self.task_index > 15 else 'cold'}"
        )


@dataclass
class MemoryActionSpace:
    """
    Defines the set of possible memory actions and their parameter ranges.
    The policy selects from this space.
    """

    ACTIONS: List[MemoryAction] = field(default_factory=lambda: [
        # Standard retrieval with varying depth
        MemoryAction(op=MemOp.RETRIEVE, top_k=1, insight_k=3, hop=1),
        MemoryAction(op=MemOp.RETRIEVE, top_k=2, insight_k=5, hop=1),
        MemoryAction(op=MemOp.RETRIEVE, top_k=3, insight_k=8, hop=2),

        # Plan injection (use stored success pattern)
        MemoryAction(op=MemOp.PLAN_INJECT, top_k=1, insight_k=3),

        # Re-retrieval with alternative query
        MemoryAction(op=MemOp.RE_RETRIEVE, top_k=2, insight_k=5, hop=2,
                     query_suffix="alternative approach"),

        # Consolidation (trigger insight merge)
        MemoryAction(op=MemOp.CONSOLIDATE),

        # Forget (prune low-scoring insights)
        MemoryAction(op=MemOp.FORGET),

        # Minimal retrieval (save tokens)
        MemoryAction(op=MemOp.RETRIEVE, top_k=1, insight_k=2, hop=0),

        # No memory operation
        MemoryAction(op=MemOp.NO_OP),
    ])

    def size(self) -> int:
        return len(self.ACTIONS)


class MemoryMDP:
    """
    The Memory MDP environment.
    Wraps the actual memory backend and provides state/action/reward interface
    for the policy to learn from.
    """

    def __init__(self):
        self.state = MemoryState()
        self.action_space = MemoryActionSpace()
        self._episode_actions: List[Tuple[str, int]] = []
        self._step_count = 0

    def observe(self, **kwargs) -> MemoryState:
        """Update state from external signals."""
        for k, v in kwargs.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)

        step = kwargs.get("step", self._step_count)
        if step < 8:
            self.state.step_phase = "early"
        elif step < 18:
            self.state.step_phase = "mid"
        else:
            self.state.step_phase = "late"

        return self.state

    def record_action(self, action_idx: int):
        key = self.state.to_key()
        self._episode_actions.append((key, action_idx))
        self._step_count += 1

    def get_episode_actions(self) -> List[Tuple[str, int]]:
        return self._episode_actions

    def reset(self, goal_type: str = "", task_index: int = 0):
        self.state = MemoryState(goal_type=goal_type, task_index=task_index)
        self._episode_actions = []
        self._step_count = 0
