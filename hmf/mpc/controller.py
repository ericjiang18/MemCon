"""
MPC Controller — the core predictive-control layer.

At each decision step the controller:
1. Observes the joint memory + task state
2. Enumerates feasible action sequences of length H (horizon)
3. Scores each sequence via cost_model (heuristic) or LLM (learned)
4. Executes only the first action, then re-plans

The LLM-based scorer (inspired by LLMPC) treats the language model
as an implicit cost-function optimizer: given the state context and
candidate plans, the LLM ranks them.
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import MPCConfig
from ..memory.base import MemoryAction, MemoryActionType
from .cost_model import CostModel
from .state import MPCState, StepRecord


# Candidate actions the controller can propose at any step
_CANDIDATE_ACTIONS = [
    MemoryActionType.CACHE_READ,
    MemoryActionType.RETRIEVE,
    MemoryActionType.SKILL_INVOKE,
    MemoryActionType.LLM_GENERATE,
    MemoryActionType.SKILL_CONSOLIDATE,
    MemoryActionType.EVICT_CACHE,
    MemoryActionType.NO_OP,
]

_MPC_SCORING_PROMPT = """\
You are a memory controller for an AI agent. Given the agent's current state and \
candidate memory-action plans, score each plan from 0 (worst) to 10 (best).

Consider:
- Token efficiency: prefer cheaper actions when budget is tight
- Latency: prefer fast actions under time pressure
- Accuracy: prefer actions that reduce uncertainty
- Redundancy: penalize repeated identical actions

Current state:
{state_context}

Current observation:
{observation}

Candidate plans (each plan = sequence of memory actions):
{plans_text}

Output a JSON list of scores, one per plan, e.g. [7, 3, 9, ...].
Output ONLY the JSON list, no explanation."""


class MPCController:
    """
    Model Predictive Control over heterogeneous memory substrates.

    Supports two scoring modes:
      - heuristic: fast analytical cost model (default fallback)
      - llm: uses a lightweight LLM call to rank candidate plans
    """

    def __init__(
        self,
        config: MPCConfig,
        cost_model: Optional[CostModel] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.cfg = config
        self.cost_model = cost_model or CostModel(config)
        self.llm_fn = llm_fn

        self._plan_cache: Dict[str, List[MemoryActionType]] = {}
        self._stats = {"plans_evaluated": 0, "llm_scores": 0, "heuristic_scores": 0}

    def select_action(
        self,
        state: MPCState,
        observation: str = "",
        available_actions: Optional[List[MemoryActionType]] = None,
    ) -> MemoryAction:
        """
        Main entry: pick the best next memory action given current state.
        Plans over a horizon but returns only the first action.
        """
        candidates = available_actions or _CANDIDATE_ACTIONS
        feasible = [a for a in candidates if self.cost_model.is_feasible(a, state)]
        if not feasible:
            return MemoryAction(action_type=MemoryActionType.NO_OP)

        plans = self._generate_plans(feasible, state)
        if not plans:
            return MemoryAction(action_type=MemoryActionType.LLM_GENERATE)

        scores = self._score_plans(plans, state, observation)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_plan = plans[best_idx]
        chosen = best_plan[0]

        self._stats["plans_evaluated"] += len(plans)

        return MemoryAction(
            action_type=chosen,
            query=observation,
            estimated_token_cost=self.cost_model.estimate_tokens(chosen),
            estimated_latency_ms=self.cost_model.estimate_latency(chosen),
            confidence=scores[best_idx] / 10.0 if scores[best_idx] > 0 else 0.0,
        )

    def _generate_plans(
        self,
        feasible: List[MemoryActionType],
        state: MPCState,
    ) -> List[List[MemoryActionType]]:
        """
        Generate candidate action sequences of length up to H.
        Uses pruning to keep the number manageable.
        """
        horizon = min(self.cfg.horizon, 3)

        if len(feasible) <= 4:
            plans = list(itertools.product(feasible, repeat=horizon))
        else:
            plans = []
            for first in feasible:
                for second in feasible[:4]:
                    plan = [first, second]
                    if horizon >= 3:
                        plan.append(MemoryActionType.LLM_GENERATE)
                    plans.append(plan)

        # Prune obviously bad plans
        pruned = []
        for plan in plans:
            if self._plan_is_valid(plan, state):
                pruned.append(list(plan))

        if len(pruned) > self.cfg.num_candidates * 3:
            costs = []
            for p in pruned:
                c = sum(self.cost_model.estimate_cost(a, state) for a in p)
                costs.append(c)
            indexed = sorted(enumerate(costs), key=lambda x: x[1])
            pruned = [pruned[i] for i, _ in indexed[: self.cfg.num_candidates * 3]]

        return pruned[: self.cfg.num_candidates * 3]

    def _plan_is_valid(self, plan: List[MemoryActionType], state: MPCState) -> bool:
        if len(plan) >= 2 and plan[0] == plan[1] == MemoryActionType.NO_OP:
            return False
        if plan[0] == MemoryActionType.SKILL_INVOKE and state.skill_count == 0:
            return False
        if plan[0] == MemoryActionType.CACHE_READ and state.cache_size == 0:
            return False
        if (plan[0] == MemoryActionType.SKILL_CONSOLIDATE
                and state.trajectory_buffer_size < 3):
            return False
        return True

    def _score_plans(
        self,
        plans: List[List[MemoryActionType]],
        state: MPCState,
        observation: str,
    ) -> List[float]:
        """Score plans: try LLM first, fall back to heuristic."""
        if self.cfg.use_llm_scoring and self.llm_fn:
            try:
                scores = self._score_with_llm(plans, state, observation)
                if scores and len(scores) == len(plans):
                    self._stats["llm_scores"] += 1
                    return scores
            except Exception:
                pass

        self._stats["heuristic_scores"] += 1
        return self._score_with_heuristic(plans, state)

    def _score_with_heuristic(
        self,
        plans: List[List[MemoryActionType]],
        state: MPCState,
    ) -> List[float]:
        """Analytical scoring: negate the sum of costs, scale to [0, 10]."""
        raw_costs = []
        for plan in plans:
            cost = sum(self.cost_model.estimate_cost(a, state) for a in plan)
            raw_costs.append(cost)

        if not raw_costs:
            return []

        min_c, max_c = min(raw_costs), max(raw_costs)
        span = max_c - min_c if max_c > min_c else 1.0
        scores = [10.0 * (1.0 - (c - min_c) / span) for c in raw_costs]
        return scores

    def _score_with_llm(
        self,
        plans: List[List[MemoryActionType]],
        state: MPCState,
        observation: str,
    ) -> List[float]:
        """Use the LLM as an implicit cost-function optimizer (LLMPC-style)."""
        plans_text = ""
        for i, plan in enumerate(plans):
            actions = " → ".join(a.name for a in plan)
            plans_text += f"  Plan {i+1}: {actions}\n"

        prompt = _MPC_SCORING_PROMPT.format(
            state_context=state.to_prompt_context(),
            observation=observation[:500],
            plans_text=plans_text,
        )

        response = self.llm_fn(prompt)

        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            scores = json.loads(response[start:end])
            return [float(s) for s in scores]

        return []

    def plan_maintenance(self, state: MPCState) -> List[MemoryAction]:
        """
        Between-step maintenance planning.
        Returns a list of housekeeping actions (evictions, consolidations)
        that should be executed before the next decision step.
        """
        actions = []

        # Token pressure → evict stale cache
        total_footprint = (
            state.cache_token_footprint
            + state.retrieval_token_footprint
        )
        if total_footprint > state.tokens_budget * 0.8:
            if state.cache_size > 10:
                actions.append(MemoryAction(
                    action_type=MemoryActionType.EVICT_CACHE,
                    metadata={"reason": "token_pressure"},
                ))

        # Enough trajectories → try consolidation
        if state.trajectory_buffer_size >= 5 and state.skill_count < 50:
            actions.append(MemoryAction(
                action_type=MemoryActionType.SKILL_CONSOLIDATE,
                metadata={"reason": "buffer_full"},
            ))

        return actions

    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)
