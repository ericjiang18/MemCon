"""
HMF-Static — Ablation: same three memory substrates, NO MPC controller.

Uses a fixed pipeline instead of adaptive MPC decisions:
  retrieve_memory: always cache → retrieval → skill (all three, fixed order)
  save_task_context: always write all three + consolidate every 5 tasks
  move_memory_state: always cache every observation

This exists to isolate the contribution of MPC vs. just having heterogeneous memory.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage
from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc

from ..config import CacheMemoryConfig, RetrievalMemoryConfig, SkillMemoryConfig
from ..memory.base import MemoryEntry
from ..memory.cache_memory import CacheMemory
from ..memory.retrieval_memory import RetrievalMemory
from ..memory.skill_memory import SkillMemory as HMFSkillMemory, Skill as HMFSkill, TrajectoryBuffer


class _SkillProxy:
    def __init__(self, hmf_skill: HMFSkill):
        self.skill_id = hmf_skill.skill_id
        self.name = hmf_skill.name
        self.description = hmf_skill.description
        self.steps = hmf_skill.steps
        self.preconditions = hmf_skill.preconditions
        self.postconditions = hmf_skill.postconditions
        self.success_rate = hmf_skill.success_rate
        self.usage_count = hmf_skill.usage_count
        self.active = hmf_skill.active


@dataclass
class HMFStaticMemory(MASMemoryBase):
    """
    Same three memory layers as HMFMemory, but with a FIXED pipeline:
      - Always query all three layers in order
      - Always write to all layers
      - Consolidate skills every N tasks (fixed schedule)
      - No cost-aware action selection
      - No adaptive eviction
    """

    def __post_init__(self):
        super().__post_init__()

        self._embed_fn_wrapper = lambda text: self.embedding_func.embed_query(text)

        def _llm_fn(prompt: str) -> str:
            return self.llm_model(
                messages=[
                    Message("system", "You are a concise helper."),
                    Message("user", prompt),
                ],
                temperature=0.3,
                max_tokens=1024,
            )

        self._llm_fn = _llm_fn
        hmf_dir = os.path.join(self.persist_dir, "hmf_static")

        self.cache = CacheMemory(
            config=CacheMemoryConfig(),
            embed_fn=self._embed_fn_wrapper,
        )
        self.retrieval = RetrievalMemory(
            config=RetrievalMemoryConfig(),
            embed_fn=self._embed_fn_wrapper,
            persist_dir=os.path.join(hmf_dir, "retrieval"),
        )
        self.skill_mem = HMFSkillMemory(
            config=SkillMemoryConfig(),
            embed_fn=self._embed_fn_wrapper,
            llm_fn=self._llm_fn,
            persist_dir=os.path.join(hmf_dir, "skills"),
        )

        self.memory_size: int = 0
        self._step_history: List[Dict[str, Any]] = []
        self._all_memories: List[MASMessage] = []
        self._consolidate_every = 5

        print(f"[HMFStatic] initialized (no MPC) at {hmf_dir}")

    # ======================== Task Lifecycle ========================

    def init_task_context(self, task_main, task_description=None, **kw):
        self._step_history = []
        return super().init_task_context(task_main, task_description)

    def move_memory_state(self, action: str, observation: str, **kwargs):
        super().move_memory_state(action, observation, **kwargs)
        reward = kwargs.get("reward", 0.0)
        done = kwargs.get("done", False)
        self._step_history.append({
            "action": action, "observation": observation,
            "reward": reward, "done": done,
        })
        # Static policy: always cache every non-think step
        is_think = action.lower().startswith(("think", "thought"))
        if not is_think and observation:
            entry = MemoryEntry(
                key=f"step_{len(self._step_history)}",
                content=f"Action: {action}\nResult: {observation[:300]}",
                source="step_cache",
            )
            self.cache.write(entry)

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        task_main = self.current_task_context.task_main
        trajectory = self.current_task_context.task_trajectory or ""

        # Static: always write to retrieval
        self.retrieval.write_experience(
            task_goal=task_main,
            trajectory=trajectory[:800],
            success=label,
            goal_type="",
            metadata={"steps": len(self._step_history), "feedback": (feedback or "")[:300]},
        )

        # Static: always buffer trajectory
        actions = [h["action"] for h in self._step_history if not h["action"].lower().startswith(("think", "thought"))]
        observations = [h["observation"] for h in self._step_history]
        self.skill_mem.record_trajectory(TrajectoryBuffer(
            task_goal=task_main,
            goal_type="",
            actions=actions,
            observations=observations,
            success=label,
            total_steps=len(actions),
        ))

        # Static: consolidate on fixed schedule (every N tasks), no cost check
        if self.memory_size > 0 and self.memory_size % self._consolidate_every == 0:
            self.skill_mem.consolidate()

        # Static: always cache success
        if label:
            summary = f"Goal: {task_main}\nSteps: {len(actions)}\nActions: {'; '.join(actions[:10])}"
            entry = MemoryEntry(
                key=f"success_{self.memory_size}",
                content=summary,
                importance=1.5,
                source="success_cache",
            )
            self.cache.write(entry)

        # NO eviction logic — memory grows unbounded
        self._step_history = []
        return super().save_task_context(label=label, feedback=feedback)

    # ======================== Cross-Trial Memory ========================

    def add_memory(self, mas_message: MASMessage):
        self._all_memories.append(mas_message)
        self.memory_size = len(self._all_memories)

    def retrieve_memory(
        self,
        query_task: str = "",
        successful_topk: int = 2,
        failed_topk: int = 0,
        insight_topk: int = 5,
        skill_topk: int = 3,
        threshold: float = 0.3,
        **kwargs,
    ) -> Tuple[List[MASMessage], List[MASMessage], List[str], List[Any]]:
        """
        Static pipeline: always run all three layers, no MPC scoring.
        Fixed order: cache → retrieval → skill.
        """
        successful_trajs: List[MASMessage] = []
        failed_trajs: List[MASMessage] = []
        insights: List[str] = []
        skills: List[Any] = []

        # Step 1: Always check cache
        cached = self.cache.read(query_task)
        if cached:
            insights.append(f"[Cached] {cached[0].content[:300]}")

        # Step 2: Always retrieve experiences
        experiences = self.retrieval.read(query_task, top_k=successful_topk + 3)
        for exp in experiences:
            if exp.metadata.get("success", False):
                fake_msg = MASMessage(
                    task_main=exp.metadata.get("task_goal", query_task),
                    task_description=exp.content[:500],
                    task_trajectory=exp.content,
                    label=True,
                )
                successful_trajs.append(fake_msg)
            else:
                insights.append(f"[Past failure] {exp.content[:200]}")
        successful_trajs = successful_trajs[:successful_topk]

        for exp in experiences[:insight_topk]:
            if exp.source == "experience" and exp.content:
                snippet = exp.content.split("\n")[0][:200]
                if snippet not in insights:
                    insights.append(snippet)
        insights = insights[:insight_topk]

        # Step 3: Always retrieve skills
        skill_entries = self.skill_mem.read(query_task, top_k=skill_topk)
        for entry in skill_entries:
            skill_data = entry.metadata.get("skill")
            if skill_data:
                hmf_skill = HMFSkill.from_dict(skill_data)
                skills.append(_SkillProxy(hmf_skill))

        return (successful_trajs, failed_trajs, insights, skills)

    def backward(self, reward, **kwargs):
        pass
