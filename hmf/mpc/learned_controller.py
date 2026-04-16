"""
Learned MPC Controller — replaces hand-tuned heuristics with online learning.

Uses a contextual bandit approach:
  - State features: [cache_hit_rate, retrieval_size, skill_count, skill_avg_sr,
                      budget_fraction, uncertainty, task_progress, ...]
  - Actions: memory action types (CACHE_READ, RETRIEVE, SKILL_INVOKE, etc.)
  - Reward: task success (+1) or failure (-0.5), weighted by token efficiency

Each (feature_bin, action) pair maintains a Q-value updated online via
exponential moving average. Action selection uses UCB (Upper Confidence Bound)
for exploration-exploitation balance.

No pretraining needed — learns from task-by-task feedback during evaluation.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..memory.base import MemoryAction, MemoryActionType
from .state import MPCState


_ACTIONS = [
    MemoryActionType.CACHE_READ,
    MemoryActionType.RETRIEVE,
    MemoryActionType.SKILL_INVOKE,
    MemoryActionType.LLM_GENERATE,
    MemoryActionType.SKILL_CONSOLIDATE,
    MemoryActionType.EVICT_CACHE,
    MemoryActionType.NO_OP,
]

_ACTION_TOKEN_COST = {
    MemoryActionType.CACHE_READ: 0,
    MemoryActionType.RETRIEVE: 50,
    MemoryActionType.SKILL_INVOKE: 100,
    MemoryActionType.LLM_GENERATE: 800,
    MemoryActionType.SKILL_CONSOLIDATE: 500,
    MemoryActionType.EVICT_CACHE: 0,
    MemoryActionType.NO_OP: 0,
}

_ACTION_LATENCY = {
    MemoryActionType.CACHE_READ: 5.0,
    MemoryActionType.RETRIEVE: 100.0,
    MemoryActionType.SKILL_INVOKE: 200.0,
    MemoryActionType.LLM_GENERATE: 3000.0,
    MemoryActionType.SKILL_CONSOLIDATE: 2000.0,
    MemoryActionType.EVICT_CACHE: 1.0,
    MemoryActionType.NO_OP: 0.0,
}


@dataclass
class LearnedControllerConfig:
    learning_rate: float = 0.2
    ucb_c: float = 1.5
    discount: float = 0.95
    feature_bins: int = 4
    persist_path: Optional[str] = None


def _discretize(value: float, bins: int = 4) -> int:
    """Map a [0, 1] float to a discrete bin."""
    return min(int(value * bins), bins - 1)


class LearnedMPCController:
    """
    Contextual bandit over memory actions with UCB exploration.

    State is discretized into a compact feature vector; Q-values are
    updated online after each task completes.
    """

    def __init__(self, config: Optional[LearnedControllerConfig] = None):
        self.cfg = config or LearnedControllerConfig()

        self._q: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_steps = 0

        self._current_context: Optional[str] = None
        self._current_action: Optional[MemoryActionType] = None
        self._episode_actions: List[tuple] = []

        self._stats = {"updates": 0, "total_reward": 0.0}

        if self.cfg.persist_path:
            self._load()

    def _state_features(self, state: MPCState) -> str:
        """Discretize state into a hashable context key."""
        bins = self.cfg.feature_bins
        features = (
            _discretize(state.cache_hit_rate, bins),
            _discretize(min(state.retrieval_size / 50.0, 1.0), bins),
            _discretize(min(state.skill_count / 20.0, 1.0), bins),
            _discretize(state.skill_avg_sr, bins),
            _discretize(state.budget_fraction_used, bins),
            _discretize(state.uncertainty, bins),
        )
        return str(features)

    def _feasible_actions(self, state: MPCState) -> List[MemoryActionType]:
        feasible = []
        for a in _ACTIONS:
            tok = _ACTION_TOKEN_COST.get(a, 200)
            lat = _ACTION_LATENCY.get(a, 500)
            if state.token_headroom < tok:
                continue
            if state.latency_headroom_ms < lat:
                continue
            if a == MemoryActionType.SKILL_INVOKE and state.skill_count == 0:
                continue
            if a == MemoryActionType.CACHE_READ and state.cache_size == 0:
                continue
            if (a == MemoryActionType.SKILL_CONSOLIDATE
                    and state.trajectory_buffer_size < 3):
                continue
            feasible.append(a)
        return feasible or [MemoryActionType.LLM_GENERATE]

    def select_action(
        self,
        state: MPCState,
        observation: str = "",
        **kwargs,
    ) -> MemoryAction:
        """UCB-based action selection."""
        ctx = self._state_features(state)
        feasible = self._feasible_actions(state)

        self._total_steps += 1

        best_action = None
        best_score = -float("inf")

        for a in feasible:
            a_key = a.name
            q = self._q[ctx][a_key]
            n = self._counts[ctx][a_key]

            if n == 0:
                ucb = float("inf")
            else:
                ucb = q + self.cfg.ucb_c * math.sqrt(
                    math.log(self._total_steps + 1) / n
                )

            if ucb > best_score:
                best_score = ucb
                best_action = a

        if best_action is None:
            best_action = MemoryActionType.LLM_GENERATE

        self._current_context = ctx
        self._current_action = best_action
        self._episode_actions.append((ctx, best_action))

        return MemoryAction(
            action_type=best_action,
            query=observation,
            estimated_token_cost=_ACTION_TOKEN_COST.get(best_action, 200),
            estimated_latency_ms=_ACTION_LATENCY.get(best_action, 500),
            confidence=min(self._counts[ctx].get(best_action.name, 0) / 10.0, 1.0),
        )

    def update(self, reward: float):
        """
        Update Q-values for all actions taken in the current episode.
        Called once per task with reward = +1 (success) or -0.5 (failure).
        """
        lr = self.cfg.learning_rate
        gamma = self.cfg.discount

        for i, (ctx, action) in enumerate(reversed(self._episode_actions)):
            discounted_reward = reward * (gamma ** i)
            a_key = action.name

            old_q = self._q[ctx][a_key]
            self._q[ctx][a_key] = old_q + lr * (discounted_reward - old_q)
            self._counts[ctx][a_key] += 1

        self._stats["updates"] += 1
        self._stats["total_reward"] += reward
        self._episode_actions = []

        if self.cfg.persist_path and self._stats["updates"] % 10 == 0:
            self._save()

    def plan_maintenance(self, state: MPCState) -> List[MemoryAction]:
        """Learned maintenance: consolidate if Q-value for CONSOLIDATE is high."""
        actions = []
        ctx = self._state_features(state)
        consol_q = self._q[ctx].get(MemoryActionType.SKILL_CONSOLIDATE.name, 0)
        evict_q = self._q[ctx].get(MemoryActionType.EVICT_CACHE.name, 0)

        if state.trajectory_buffer_size >= 5 and consol_q > -0.1:
            actions.append(MemoryAction(
                action_type=MemoryActionType.SKILL_CONSOLIDATE,
            ))

        if state.budget_fraction_used > 0.7 and evict_q > -0.2:
            actions.append(MemoryAction(
                action_type=MemoryActionType.EVICT_CACHE,
            ))

        return actions

    def stats(self) -> Dict[str, Any]:
        n_contexts = len(self._q)
        return {
            **self._stats,
            "contexts_seen": n_contexts,
            "total_steps": self._total_steps,
        }

    def _save(self):
        if not self.cfg.persist_path:
            return
        os.makedirs(os.path.dirname(self.cfg.persist_path) or ".", exist_ok=True)
        data = {
            "q": {k: dict(v) for k, v in self._q.items()},
            "counts": {k: dict(v) for k, v in self._counts.items()},
            "total_steps": self._total_steps,
            "stats": self._stats,
        }
        with open(self.cfg.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not self.cfg.persist_path or not os.path.exists(self.cfg.persist_path):
            return
        try:
            with open(self.cfg.persist_path) as f:
                data = json.load(f)
            for k, v in data.get("q", {}).items():
                self._q[k].update(v)
            for k, v in data.get("counts", {}).items():
                self._counts[k].update(v)
            self._total_steps = data.get("total_steps", 0)
            self._stats = data.get("stats", self._stats)
        except Exception as e:
            print(f"[LearnedMPC] load error: {e}")
