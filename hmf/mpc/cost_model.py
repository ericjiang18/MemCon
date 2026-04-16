"""
Cost Model — estimates token cost, latency, and expected utility
for each candidate memory action.

The MPC controller uses these estimates to score candidate action
sequences over the planning horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..config import MPCConfig
from ..memory.base import MemoryActionType
from .state import MPCState


# Typical cost profiles for each action type (tokens, latency_ms)
_ACTION_PROFILES: Dict[MemoryActionType, tuple] = {
    MemoryActionType.CACHE_READ:         (0,    5),
    MemoryActionType.CACHE_WRITE:        (0,    5),
    MemoryActionType.RETRIEVE:           (50,   100),
    MemoryActionType.RETRIEVE_WRITE:     (10,   50),
    MemoryActionType.SKILL_INVOKE:       (100,  200),
    MemoryActionType.SKILL_CONSOLIDATE:  (500,  2000),
    MemoryActionType.EVICT_CACHE:        (0,    1),
    MemoryActionType.EVICT_RETRIEVAL:    (0,    1),
    MemoryActionType.EVICT_SKILL:        (0,    1),
    MemoryActionType.LLM_GENERATE:       (800,  3000),
    MemoryActionType.NO_OP:              (0,    0),
}


class CostModel:
    """
    Estimates the multi-objective cost of a memory action given the current state.

    cost(a) = α·token_cost + β·latency_cost + γ·uncertainty_penalty - δ·utility
    """

    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg

    def estimate_cost(
        self,
        action: MemoryActionType,
        state: MPCState,
        extra: dict | None = None,
    ) -> float:
        extra = extra or {}
        tokens, latency = _ACTION_PROFILES.get(action, (200, 500))

        token_cost = tokens / max(state.token_headroom, 1)
        latency_cost = latency / max(state.latency_headroom_ms, 1.0)
        uncertainty = self._uncertainty_penalty(action, state)
        utility = self._expected_utility(action, state, extra)

        cost = (
            self.cfg.alpha_token * token_cost
            + self.cfg.beta_latency * latency_cost
            + self.cfg.gamma_uncertainty * uncertainty
            - self.cfg.delta_utility * utility
        )
        return cost

    def estimate_tokens(self, action: MemoryActionType) -> int:
        return _ACTION_PROFILES.get(action, (200, 500))[0]

    def estimate_latency(self, action: MemoryActionType) -> float:
        return float(_ACTION_PROFILES.get(action, (200, 500))[1])

    def is_feasible(self, action: MemoryActionType, state: MPCState) -> bool:
        tokens, latency = _ACTION_PROFILES.get(action, (200, 500))
        if state.token_headroom < tokens:
            return False
        if state.latency_headroom_ms < latency:
            return False
        return True

    def _uncertainty_penalty(self, action: MemoryActionType, state: MPCState) -> float:
        """Higher penalty when uncertainty is high and the action is cheap/shallow."""
        if action in (MemoryActionType.CACHE_READ, MemoryActionType.NO_OP):
            return state.uncertainty * 0.8
        if action == MemoryActionType.RETRIEVE:
            return state.uncertainty * 0.3
        if action == MemoryActionType.SKILL_INVOKE:
            return state.uncertainty * 0.2 * (1 - state.skill_avg_sr)
        if action == MemoryActionType.LLM_GENERATE:
            return state.uncertainty * 0.1
        return state.uncertainty * 0.5

    def _expected_utility(
        self,
        action: MemoryActionType,
        state: MPCState,
        extra: dict,
    ) -> float:
        """Heuristic utility based on state signals."""
        if action == MemoryActionType.CACHE_READ:
            return state.cache_hit_rate * 0.9

        if action == MemoryActionType.RETRIEVE:
            base = 0.5
            if state.uncertainty > 0.5:
                base += 0.3
            if state.retrieval_size > 0:
                base += 0.1
            return min(base, 1.0)

        if action == MemoryActionType.SKILL_INVOKE:
            if state.skill_count == 0:
                return 0.0
            return state.skill_avg_sr * 0.8

        if action == MemoryActionType.LLM_GENERATE:
            base = 0.6
            if state.budget_fraction_used > 0.8:
                base -= 0.3
            return max(base, 0.1)

        if action == MemoryActionType.SKILL_CONSOLIDATE:
            if state.trajectory_buffer_size >= 3:
                return 0.5
            return 0.1

        if action in (
            MemoryActionType.EVICT_CACHE,
            MemoryActionType.EVICT_RETRIEVAL,
            MemoryActionType.EVICT_SKILL,
        ):
            if state.budget_fraction_used > 0.7:
                return 0.4
            return 0.1

        if action == MemoryActionType.CACHE_WRITE:
            return 0.3

        if action == MemoryActionType.RETRIEVE_WRITE:
            return 0.3

        return 0.0
