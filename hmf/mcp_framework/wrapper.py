"""
MemCon Wrapper — wraps ANY MASMemoryBase backend with learned policy control.

This is the key architectural contribution: the wrapper is AGNOSTIC to the
underlying memory implementation. It can wrap:
  - G-Memory (graph + Chroma + insights)
  - Flat vector store
  - SkillMemory
  - Any future memory backend

The wrapper intercepts retrieve_memory() and save_task_context() calls,
lets the policy decide the parameters, and learns from outcomes.
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage

from .memory_mdp import MemOp, MemoryAction, MemoryActionSpace, MemoryMDP, MemoryState
from .policy import MemoryPolicy, PolicyConfig


def _extract_goal_type(task: str) -> str:
    task_lower = task.lower()
    for verb in ("puttwo", "clean", "heat", "cool", "examine", "look at", "put"):
        if verb in task_lower:
            return verb
    return "other"


@dataclass
class MemConWrapper(MASMemoryBase):
    """
    Memory as a Controlled Process.

    Wraps an inner MASMemoryBase backend and adds:
    1. Learned retrieval policy (what params to use when querying)
    2. Generalized success plans (abstracted action templates per goal type)
    3. Adaptive consolidation (policy decides when to merge insights)
    4. Goal decomposition for complex tasks
    """

    def __post_init__(self):
        super().__post_init__()

        memcon_dir = os.path.join(self.persist_dir, "memcon")
        os.makedirs(memcon_dir, exist_ok=True)

        # The wrapped inner backend (G-Memory or any other)
        self._inner: Optional[MASMemoryBase] = None

        # MDP + Policy
        self._action_space = MemoryActionSpace()
        self._mdp = MemoryMDP()
        self._policy = MemoryPolicy(
            action_space=self._action_space,
            config=PolicyConfig(
                learning_rate=0.15,
                ucb_c=1.4,
                persist_path=os.path.join(memcon_dir, "policy_q.json"),
            ),
        )

        # Success plans: goal_type → generalized action sequence
        self._success_plans: Dict[str, List[str]] = {}
        self._plans_path = os.path.join(memcon_dir, "success_plans.json")
        self._load_plans()

        # Step tracking
        self._step_history: List[Dict[str, Any]] = []
        self._visited: set = set()
        self._last_action: str = ""
        self._stuck_count: int = 0
        self._task_index: int = 0
        self._current_goal_type: str = ""

        print(f"[MemCon] initialized, policy entries: {self._policy.stats()['q_entries']}")

    def set_inner(self, inner: MASMemoryBase):
        """Attach the underlying memory backend."""
        self._inner = inner

    # ============== Task Lifecycle ==============

    def init_task_context(self, task_main, task_description=None, **kw):
        self._task_index += 1
        self._current_goal_type = _extract_goal_type(task_main)
        self._step_history = []
        self._visited = set()
        self._last_action = ""
        self._stuck_count = 0

        self._mdp.reset(goal_type=self._current_goal_type, task_index=self._task_index)

        if self._inner:
            self._inner.init_task_context(task_main[:300], (task_description or "")[:500], **kw)
        return super().init_task_context(task_main[:300], (task_description or "")[:500])

    def move_memory_state(self, action: str, observation: str, **kwargs):
        super().move_memory_state(action, observation, **kwargs)
        if self._inner:
            self._inner.move_memory_state(action, observation, **kwargs)

        is_think = action.lower().startswith(("think", "thought"))
        if not is_think:
            self._step_history.append({"action": action, "observation": observation})

            m = re.search(r"arrive at (.+?)[\.\!]", observation.lower())
            if m:
                self._visited.add(m.group(1))

            if action == self._last_action:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_action = action

    # ============== Policy-Controlled Retrieval ==============

    def retrieve_memory(self, query_task="", successful_topk=2, failed_topk=1,
                        insight_topk=10, threshold=0.3, **kwargs) -> tuple:
        if not self._inner:
            return ([], [], [])

        goal_type = self._current_goal_type or _extract_goal_type(query_task)

        # Update MDP state
        inner_size = getattr(self._inner, 'memory_size', 0)
        if callable(inner_size):
            inner_size = inner_size()
        elif isinstance(inner_size, property):
            inner_size = 0

        self._mdp.observe(
            goal_type=goal_type,
            step=len(self._step_history),
            is_stuck=self._stuck_count >= 2,
            objects_held=0,
            locations_visited=len(self._visited),
            mem_size=inner_size if isinstance(inner_size, int) else 0,
            plan_available=goal_type in self._success_plans,
            task_index=self._task_index,
        )

        # Policy selects memory action
        action_idx, mem_action = self._policy.select(self._mdp.state)
        self._mdp.record_action(action_idx)

        # Execute the selected action
        if mem_action.op == MemOp.NO_OP:
            return ([], [], [])

        if mem_action.op == MemOp.CONSOLIDATE:
            if hasattr(self._inner, 'insights_layer'):
                try:
                    self._inner.insights_layer.merge_insights()
                except Exception:
                    pass
            return self._do_retrieve(query_task, mem_action, threshold)

        if mem_action.op == MemOp.FORGET:
            if hasattr(self._inner, 'insights_layer'):
                try:
                    self._inner.insights_layer.clear_insights()
                except Exception:
                    pass
            return self._do_retrieve(query_task, mem_action, threshold)

        # RETRIEVE, RE_RETRIEVE, PLAN_INJECT all involve retrieval
        result = self._do_retrieve(query_task, mem_action, threshold)

        successful = list(result[0])
        failed = list(result[1]) if len(result) > 1 else []
        insights = list(result[2]) if len(result) > 2 else []

        # Plan injection: add generalized success plan
        if mem_action.op in (MemOp.PLAN_INJECT, MemOp.RETRIEVE) and goal_type in self._success_plans:
            plan = self._success_plans[goal_type]
            plan_text = "\n".join(f"    {i+1}. {s}" for i, s in enumerate(plan[:8]))
            insights.insert(0,
                f"[Proven plan for '{goal_type}' tasks]\n{plan_text}\n"
                f"  Adapt object/location names to your current task."
            )

        # Goal decomposition for multi-object tasks
        if goal_type == "puttwo":
            insights.insert(0,
                "[TWO-OBJECT TASK: Complete all steps for object 1 first, then repeat for object 2]\n"
                "  1. Find & take first object → put it at target\n"
                "  2. Find & take second object → put it at target"
            )
            # Also retrieve for simple "put" to get relevant trajectory
            if "put" in self._success_plans:
                put_plan = self._success_plans["put"]
                insights.append(
                    f"[Reference: single-put plan]\n"
                    + "\n".join(f"    {i+1}. {s}" for i, s in enumerate(put_plan[:5]))
                )

        return (successful, failed, insights[:insight_topk]) + tuple(result[3:]) if len(result) > 3 else (successful, failed, insights[:insight_topk])

    def _do_retrieve(self, query: str, action: MemoryAction, threshold: float) -> tuple:
        """Execute retrieval on the inner backend with policy-chosen parameters."""
        q = query[:300]  # Truncate to avoid graph node errors with long contexts
        if action.op == MemOp.RE_RETRIEVE and action.query_suffix:
            q = f"{q} {action.query_suffix}"

        return self._inner.retrieve_memory(
            query_task=q,
            successful_topk=action.top_k,
            failed_topk=0,
            insight_topk=action.insight_k,
            threshold=threshold,
            hop=action.hop if hasattr(self._inner, '_hop') else None,
        )

    # ============== Save + Learn ==============

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        goal_type = self._current_goal_type

        # Store generalized success plan
        if label and self._step_history:
            actions = [h["action"] for h in self._step_history]
            generalized = self._generalize_actions(actions[:15])
            self._success_plans[goal_type] = generalized
            self._save_plans()

        # Compute reward
        reward = 1.0 if label else -0.5
        n_steps = len(self._step_history)
        if label and n_steps > 0:
            efficiency = max(0, 1.0 - n_steps / 30.0)
            reward += 0.3 * efficiency

        # Update policy from this episode
        episode_actions = self._mdp.get_episode_actions()
        if episode_actions:
            self._policy.update(episode_actions, reward)

        # Delegate to inner backend
        self._step_history = []
        if self._inner:
            self._inner.save_task_context(label=label, feedback=feedback)
        return super().save_task_context(label=label, feedback=feedback)

    # ============== Generalization ==============

    @staticmethod
    def _generalize_actions(actions: List[str]) -> List[str]:
        """
        Generalize an action sequence by replacing specific object/location IDs
        with placeholders. E.g., "go to shelf 3" → "go to [container]"
        """
        generalized = []
        for action in actions:
            g = action
            # Replace numbered locations with category
            g = re.sub(r"(shelf|drawer|cabinet|countertop|desk|bed|armchair|sofa|dresser|sidetable|coffeetable|diningtable|garbagecan|safe|bathtubbasin|sinkbasin|toilet|fridge|microwave|stoveburner|toaster)\s*\d+",
                       r"[\1]", g, flags=re.IGNORECASE)
            # Replace specific object instances
            g = re.sub(r"(mug|cup|plate|bowl|pan|pot|egg|apple|tomato|potato|bread|lettuce|knife|fork|spoon|spatula|cd|book|pen|pencil|cellphone|remotecontrol|creditcard|newspaper|keychain|vase|pillow|tissuebox|toiletpaper|soapbar|cloth|towel|watch|alarmclock|laptop|statue|box|candle|peppershaker|saltshaker|winebottle|wateringcan|houseplant|floorlamp|desklamp|lightswitch|laundrypan|dishsponge|butterknife|baseballbat|tennisracket|basketBall)\s*\d*",
                       r"[\1]", g, flags=re.IGNORECASE)
            generalized.append(g)
        return generalized

    # ============== Delegate remaining methods ==============

    def add_memory(self, mas_message: MASMessage):
        if self._inner:
            self._inner.add_memory(mas_message)

    def backward(self, reward, **kwargs):
        if self._inner:
            self._inner.backward(reward, **kwargs)

    # ============== Persistence ==============

    def _save_plans(self):
        with open(self._plans_path, "w") as f:
            json.dump(self._success_plans, f, indent=2)

    def _load_plans(self):
        if os.path.exists(self._plans_path):
            try:
                with open(self._plans_path) as f:
                    self._success_plans = json.load(f)
            except Exception:
                pass
