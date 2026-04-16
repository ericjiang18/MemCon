"""
Memory Policy — Online contextual bandit with UCB + decay.

Learns Q(state_key, action_idx) from task success/failure feedback.
No pretraining needed; improves as more tasks are completed.

Design choices:
  - Tabular Q-learning (not neural) — fast, interpretable, no GPU needed
  - UCB exploration — balances trying new actions vs. exploiting known good ones
  - Decayed credit assignment — actions taken earlier in the episode get less credit
  - Warm-start heuristic — provides reasonable defaults before any learning
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .memory_mdp import MemOp, MemoryAction, MemoryActionSpace, MemoryState


# Warm-start heuristic: initial Q-values based on domain knowledge
_WARM_START = {
    MemOp.RETRIEVE: 0.5,        # retrieval is usually helpful
    MemOp.PLAN_INJECT: 0.3,     # plans help for some task types
    MemOp.RE_RETRIEVE: 0.1,     # useful when stuck
    MemOp.CONSOLIDATE: 0.0,     # neutral
    MemOp.FORGET: -0.1,         # slightly risky
    MemOp.NO_OP: -0.2,          # doing nothing is usually bad
}


@dataclass
class PolicyConfig:
    learning_rate: float = 0.15
    ucb_c: float = 1.4
    discount: float = 0.9
    warm_start: bool = True
    persist_path: Optional[str] = None


class MemoryPolicy:
    """
    Contextual bandit policy over memory actions.

    At each decision point:
      1. Compute state key from MemoryState
      2. For each action, compute: score = Q[state][action] + UCB_bonus
      3. Select action with highest score
      4. After task ends, update Q-values with discounted reward
    """

    def __init__(self, action_space: MemoryActionSpace, config: Optional[PolicyConfig] = None):
        self.cfg = config or PolicyConfig()
        self.action_space = action_space

        self._q: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._total_selections = 0

        self._stats = {"updates": 0, "total_reward": 0.0, "explorations": 0, "exploitations": 0}

        if self.cfg.persist_path:
            self._load()

    def select(self, state: MemoryState) -> Tuple[int, MemoryAction]:
        """Select best memory action for the given state via UCB."""
        key = state.to_key()
        n_actions = self.action_space.size()
        self._total_selections += 1

        best_idx = 0
        best_score = -float("inf")
        explored = False

        for i in range(n_actions):
            q = self._get_q(key, i)
            n = self._counts[key][i]

            if n == 0:
                ucb = float("inf")
                explored = True
            else:
                ucb = q + self.cfg.ucb_c * math.sqrt(
                    math.log(self._total_selections + 1) / n
                )

            if ucb > best_score:
                best_score = ucb
                best_idx = i

        self._counts[key][best_idx] += 1
        if explored:
            self._stats["explorations"] += 1
        else:
            self._stats["exploitations"] += 1

        return best_idx, self.action_space.ACTIONS[best_idx]

    def update(self, episode_actions: List[Tuple[str, int]], reward: float):
        """
        Update Q-values for all (state, action) pairs in the episode.
        Uses reverse-discounted credit assignment.
        """
        lr = self.cfg.learning_rate
        gamma = self.cfg.discount

        for i, (state_key, action_idx) in enumerate(reversed(episode_actions)):
            discounted_r = reward * (gamma ** i)
            old_q = self._get_q(state_key, action_idx)
            self._q[state_key][action_idx] = old_q + lr * (discounted_r - old_q)

        self._stats["updates"] += 1
        self._stats["total_reward"] += reward

        if self.cfg.persist_path and self._stats["updates"] % 5 == 0:
            self._save()

    def _get_q(self, key: str, action_idx: int) -> float:
        if action_idx in self._q[key]:
            return self._q[key][action_idx]
        if self.cfg.warm_start:
            action = self.action_space.ACTIONS[action_idx]
            return _WARM_START.get(action.op, 0.0)
        return 0.0

    def get_learned_preferences(self, state: MemoryState) -> Dict[str, float]:
        """Return Q-values for debugging/analysis."""
        key = state.to_key()
        result = {}
        for i, action in enumerate(self.action_space.ACTIONS):
            q = self._get_q(key, i)
            n = self._counts[key].get(i, 0)
            result[f"{action.op.name}(k={action.top_k},h={action.hop})"] = round(q, 3)
        return result

    def stats(self) -> Dict:
        n_contexts = len(self._q)
        total_entries = sum(len(v) for v in self._q.values())
        return {
            **self._stats,
            "contexts_learned": n_contexts,
            "q_entries": total_entries,
            "total_selections": self._total_selections,
        }

    def _save(self):
        if not self.cfg.persist_path:
            return
        os.makedirs(os.path.dirname(self.cfg.persist_path) or ".", exist_ok=True)
        data = {
            "q": {k: dict(v) for k, v in self._q.items()},
            "counts": {k: dict(v) for k, v in self._counts.items()},
            "total": self._total_selections,
            "stats": self._stats,
        }
        with open(self.cfg.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        if not self.cfg.persist_path or not os.path.exists(self.cfg.persist_path):
            return
        try:
            with open(self.cfg.persist_path) as f:
                data = json.load(f)
            for k, v in data.get("q", {}).items():
                self._q[k] = {int(ki): vi for ki, vi in v.items()}
            for k, v in data.get("counts", {}).items():
                self._counts[k] = {int(ki): vi for ki, vi in v.items()}
            self._total_selections = data.get("total", 0)
            self._stats = data.get("stats", self._stats)
        except Exception:
            pass
