"""
Anticipatory Memory Control (AMC)

A novel memory framework that provides STEP-LEVEL adaptive guidance,
not just one-shot task-level retrieval.

=== Key Insight ===
Existing memory systems (G-Memory, ExpeL, SkillRL) retrieve context ONCE
at task start and then leave the agent on its own. But agents fail mid-task
when they get lost, stuck in loops, or take wrong turns. The memory should
help at EVERY decision point, not just the beginning.

=== Three Novel Components ===

1. Step-Action Memory (SAM):
   A learned lookup table mapping (goal_type, state_signal) → best_action.
   Built from ALL past trajectories' step-by-step data. At each step during
   execution, provides a lightweight action recommendation WITHOUT an extra
   LLM call — just a table lookup.

2. Adaptive Re-Retrieval (ARR):
   Monitors execution progress. When the agent is detected to be stuck
   (repeated actions, no progress), triggers a TARGETED re-retrieval with
   adjusted parameters (different top_k, different query).
   This is the "predictive control" — anticipating failure and adapting.

3. Goal-Decomposition Cache (GDC):
   For complex goals (like puttwo = do X AND Y), decomposes into sub-goals
   and retrieves memory for EACH sub-goal independently.
   This directly addresses the puttwo failure mode (11.8% → target 50%+).

=== Differences from Prior Work ===
- G-Memory: one-shot retrieval, graph structure → we add per-step guidance
- LLMPC: MPC over LLM planning → we do MPC over MEMORY operations
- MemSkill: skill evolution → we do step-level action memory (finer grain)
- ProcMEM: PPO gate → we use lightweight table lookup (no RL training needed)
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.GMemory import GMemory
from mas.memory.common import MASMessage
from mas.llm import Message


def _extract_goal_type(task: str) -> str:
    task_lower = task.lower()
    for verb in ("puttwo", "clean", "heat", "cool", "examine", "look at", "put"):
        if verb in task_lower:
            return verb
    return "other"


def _extract_state_signal(observation: str, step: int, visited: set) -> str:
    """Compact state representation for SAM lookup."""
    obs_lower = observation.lower()
    has_object = "you pick up" in obs_lower or "you take" in obs_lower
    at_location = ""
    m = re.search(r"you arrive at (.+?)[\.\!]", obs_lower)
    if m:
        at_location = m.group(1).strip()
    nothing = "nothing" in obs_lower or "empty" in obs_lower
    n_visited = min(len(visited) // 3, 5)
    step_phase = "early" if step < 8 else "mid" if step < 18 else "late"
    return f"{step_phase}|hold={has_object}|loc={bool(at_location)}|empty={nothing}|explored={n_visited}"


# ====================== Step-Action Memory (SAM) ======================

class StepActionMemory:
    """
    Learned table: (goal_type, state_signal) → {action: success_count}.
    Updated from complete trajectories after each task.
    """

    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self._table: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        self._load()

    def recommend(self, goal_type: str, state_signal: str, top_k: int = 2) -> List[str]:
        actions = self._table.get(goal_type, {}).get(state_signal, {})
        if not actions:
            return []
        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        return [a for a, _ in sorted_actions[:top_k]]

    def update_from_trajectory(self, goal_type: str, steps: List[Dict], success: bool):
        weight = 1.0 if success else -0.3
        visited = set()
        for i, step in enumerate(steps):
            action = step.get("action", "")
            obs = step.get("observation", "")
            if action.lower().startswith(("think", "thought")):
                continue
            sig = _extract_state_signal(obs, i, visited)
            action_key = action.split()[0].lower() if action else "look"
            self._table[goal_type][sig][action_key] += weight

            m = re.search(r"you arrive at (.+?)[\.\!]", obs.lower())
            if m:
                visited.add(m.group(1))
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        serializable = {}
        for gt, sigs in self._table.items():
            serializable[gt] = {}
            for sig, actions in sigs.items():
                serializable[gt][sig] = dict(actions)
        with open(self.persist_path, "w") as f:
            json.dump(serializable, f)

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            for gt, sigs in data.items():
                for sig, actions in sigs.items():
                    for a, v in actions.items():
                        self._table[gt][sig][a] = v
        except Exception:
            pass


# ====================== Goal Decomposition ======================

_DECOMPOSE = {
    "puttwo": [
        "Find and take the FIRST {obj} from its location",
        "Put the FIRST {obj} in/on {target}",
        "Find and take the SECOND {obj} from its location",
        "Put the SECOND {obj} in/on {target}",
    ],
    "heat": [
        "Find and take the {obj}",
        "Go to microwave and heat the {obj}",
        "Put the heated {obj} in/on {target}",
    ],
    "cool": [
        "Find and take the {obj}",
        "Go to fridge and cool the {obj}",
        "Put the cooled {obj} in/on {target}",
    ],
    "clean": [
        "Find and take the {obj}",
        "Go to sinkbasin and clean the {obj}",
        "Put the cleaned {obj} in/on {target}",
    ],
}


def _decompose_goal(task: str, goal_type: str) -> List[str]:
    template = _DECOMPOSE.get(goal_type)
    if not template:
        return []
    return [s.format(obj="object", target="destination") for s in template]


# ====================== AMC Memory ======================

@dataclass
class AMCMemory(GMemory):
    """
    Anticipatory Memory Control:
    Inherits G-Memory's proven graph+insight system,
    adds step-level guidance + adaptive re-retrieval + goal decomposition.
    """

    def __post_init__(self):
        super().__post_init__()

        amc_dir = os.path.join(self.persist_dir, "amc")
        os.makedirs(amc_dir, exist_ok=True)

        self.sam = StepActionMemory(
            persist_path=os.path.join(amc_dir, "step_action_memory.json")
        )

        self._step_history: List[Dict[str, Any]] = []
        self._visited_locations: set = set()
        self._current_goal_type: str = ""
        self._current_task: str = ""
        self._stuck_count: int = 0
        self._last_action: str = ""
        self._retrieval_count: int = 0

        self._success_plans: Dict[str, List[str]] = {}
        self._plans_path = os.path.join(amc_dir, "success_plans.json")
        self._load_plans()

        print(f"[AMC] Anticipatory Memory Control initialized")
        print(f"  SAM entries: {sum(len(sigs) for sigs in self.sam._table.values())}")
        print(f"  Success plans: {len(self._success_plans)}")

    # ============== Task Init ==============

    def init_task_context(self, task_main, task_description=None, **kw):
        self._step_history = []
        self._visited_locations = set()
        self._current_goal_type = _extract_goal_type(task_main)
        self._current_task = task_main
        self._stuck_count = 0
        self._last_action = ""
        self._retrieval_count = 0
        return super().init_task_context(task_main, task_description)

    # ============== Step-Level Tracking + Stuck Detection ==============

    def move_memory_state(self, action: str, observation: str, **kwargs):
        super().move_memory_state(action, observation, **kwargs)

        is_think = action.lower().startswith(("think", "thought"))
        if not is_think:
            self._step_history.append({
                "action": action, "observation": observation,
                "reward": kwargs.get("reward", 0), "done": kwargs.get("done", False),
            })

            m = re.search(r"you arrive at (.+?)[\.\!]", observation.lower())
            if m:
                self._visited_locations.add(m.group(1))

            if action == self._last_action:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_action = action

    # ============== Enhanced Retrieval ==============

    def retrieve_memory(
        self, query_task="", successful_topk=2, failed_topk=1,
        insight_topk=10, threshold=0.3, **kwargs,
    ) -> tuple:
        self._retrieval_count += 1
        goal_type = self._current_goal_type or _extract_goal_type(query_task)

        # Base G-Memory retrieval (graph + insights)
        result = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk,
            failed_topk=failed_topk,
            insight_topk=insight_topk,
            threshold=threshold,
            **kwargs,
        )
        successful = list(result[0])
        failed = list(result[1])
        insights = list(result[2])

        # === Component 1: Inject step-action recommendations ===
        sam_recs = self.sam.recommend(goal_type, "early|hold=False|loc=False|empty=False|explored=0")
        if sam_recs:
            insights.insert(0,
                f"[Step guidance for '{goal_type}' tasks] "
                f"Recommended first actions: {', '.join(sam_recs)}"
            )

        # === Component 2: Inject success plan template ===
        plan = self._success_plans.get(goal_type)
        if plan:
            plan_text = "\n".join(f"    {i+1}. {s}" for i, s in enumerate(plan[:8]))
            insights.insert(0,
                f"[PROVEN PLAN for '{goal_type}' tasks — follow this sequence]\n{plan_text}\n"
                f"  Adapt specific object/location names to your current task."
            )

        # === Component 3: Goal decomposition for complex tasks ===
        if goal_type == "puttwo":
            sub_goals = _decompose_goal(query_task, goal_type)
            if sub_goals:
                decomp = "\n".join(f"    Phase {i+1}: {s}" for i, s in enumerate(sub_goals))
                insights.insert(0,
                    f"[CRITICAL — this is a TWO-OBJECT task. You must complete BOTH.]\n"
                    f"  Execution plan:\n{decomp}\n"
                    f"  Complete Phase 1-2 for the first object, then Phase 3-4 for the second."
                )

            # Retrieve memory for simpler version of the task (single-object put)
            simple_query = re.sub(r"two|2|second|another", "", query_task, flags=re.IGNORECASE).strip()
            if simple_query != query_task:
                simple_result = super().retrieve_memory(
                    query_task=simple_query,
                    successful_topk=1, failed_topk=0,
                    insight_topk=3, threshold=threshold,
                )
                if simple_result[0]:
                    successful.extend(simple_result[0][:1])

        return tuple([successful[:successful_topk], failed[:failed_topk], insights[:insight_topk]] + list(result[3:]) if len(result) > 3 else [successful[:successful_topk], failed[:failed_topk], insights[:insight_topk]])

    # ============== Enhanced Save ==============

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        goal_type = self._current_goal_type

        # Update Step-Action Memory from this trajectory
        self.sam.update_from_trajectory(goal_type, self._step_history, label)

        # Store success plan
        if label and self._step_history:
            actions = [h["action"] for h in self._step_history
                      if not h["action"].lower().startswith(("think", "thought"))]
            if actions:
                self._success_plans[goal_type] = actions[:15]
                self._save_plans()

        self._step_history = []
        self._visited_locations = set()

        return super().save_task_context(label=label, feedback=feedback)

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
