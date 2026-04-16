"""
Trajectory Manager — collects and structures step-by-step execution traces.

Used by HMFAgent to:
  1. Build sliding-window prompts during execution
  2. Provide raw material for skill consolidation
  3. Compute post-task summaries for retrieval memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    action: str
    observation: str
    reward: float = 0.0
    done: bool = False
    is_think: bool = False
    memory_action: str = ""
    token_cost: int = 0
    latency_ms: float = 0.0


@dataclass
class TrajectoryManager:
    task_goal: str = ""
    task_type: str = ""
    steps: List[Step] = field(default_factory=list)
    success: Optional[bool] = None
    total_tokens: int = 0
    total_latency_ms: float = 0.0

    def add_step(self, step: Step):
        self.steps.append(step)
        self.total_tokens += step.token_cost
        self.total_latency_ms += step.latency_ms

    def physical_steps(self) -> List[Step]:
        return [s for s in self.steps if not s.is_think]

    def action_list(self) -> List[str]:
        return [s.action for s in self.physical_steps()]

    def observation_list(self) -> List[str]:
        return [s.observation for s in self.physical_steps()]

    def sliding_window(self, window_size: int = 7) -> List[tuple]:
        """Return last N (action, observation) pairs for the prompt."""
        physical = self.physical_steps()
        window = physical[-window_size:]
        return [(s.action, s.observation) for s in window]

    def summary(self, max_actions: int = 15) -> str:
        actions = self.action_list()[:max_actions]
        outcome = "SUCCESS" if self.success else "FAILURE" if self.success is not None else "IN_PROGRESS"
        lines = [
            f"Task: {self.task_goal}",
            f"Type: {self.task_type}",
            f"Outcome: {outcome}",
            f"Steps: {len(self.physical_steps())}",
            f"Tokens: {self.total_tokens}",
            "Key actions:",
        ]
        for a in actions:
            lines.append(f"  - {a}")
        return "\n".join(lines)

    def reset(self, task_goal: str = "", task_type: str = ""):
        self.task_goal = task_goal
        self.task_type = task_type
        self.steps.clear()
        self.success = None
        self.total_tokens = 0
        self.total_latency_ms = 0.0
