"""
HMF Memory — MASMemoryBase adapter.

Plugs the Hierarchical Memory Framework (cache + retrieval + skill + MPC)
into the original tasks/run.py execution system so that all existing
benchmarks (alfworld, pddl, sciworld, aime, gpqa, mmlu, etc.) can be
run with HMF as the memory backend.

This class bridges two APIs:
  - MASMemoryBase  (what SkillMAS expects)
  - HMFAgent       (cache / retrieval / skill / MPC internals)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage, AgentMessage
from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc

from ..config import HMFConfig, CacheMemoryConfig, RetrievalMemoryConfig, SkillMemoryConfig, MPCConfig
from ..memory.base import MemoryEntry, MemoryActionType
from ..memory.cache_memory import CacheMemory
from ..memory.retrieval_memory import RetrievalMemory
from ..memory.skill_memory import SkillMemory as HMFSkillMemory, Skill as HMFSkill, TrajectoryBuffer
from ..mpc.controller import MPCController
from ..mpc.cost_model import CostModel
from ..mpc.state import MPCState, StepRecord


class _SkillProxy:
    """Minimal skill object compatible with SkillMAS's expectations."""
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
        self._hmf_skill = hmf_skill


@dataclass
class HMFMemory(MASMemoryBase):
    """
    MASMemoryBase adapter backed by the Hierarchical Memory Framework.

    At init_task_context, the MPC controller decides the optimal memory
    query strategy (cache / retrieve / skill) under budget constraints.
    At each step via move_memory_state, observations are cached.
    At save_task_context, experiences are stored and skills evolved.
    """

    def __post_init__(self):
        super().__post_init__()

        self._embed_fn_wrapper = lambda text: self.embedding_func.embed_query(text)

        def _llm_fn(prompt: str) -> str:
            resp = self.llm_model(
                messages=[
                    Message("system", "You are a concise helper."),
                    Message("user", prompt),
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp

        self._llm_fn = _llm_fn

        hmf_dir = os.path.join(self.persist_dir, "hmf")

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

        mpc_cfg = MPCConfig(use_llm_scoring=False)
        self.cost_model = CostModel(mpc_cfg)
        self.mpc = MPCController(config=mpc_cfg, cost_model=self.cost_model)
        self.mpc_state = MPCState(
            tokens_budget=mpc_cfg.token_budget,
            latency_budget_ms=mpc_cfg.latency_budget_ms,
        )

        self.memory_size: int = 0
        self._current_task_id: str = ""
        self._step_history: List[Dict[str, Any]] = []
        self._all_memories: List[MASMessage] = []

        print(f"[HMFMemory] initialized at {hmf_dir}")

    def _sync_mpc_state(self):
        self.mpc_state.snapshot_memory_stats(
            self.cache.stats(),
            self.retrieval.stats(),
            self.skill_mem.stats(),
        )

    # ======================== Task Lifecycle ========================

    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
        **kwargs,
    ) -> MASMessage:
        mas_msg = super().init_task_context(task_main, task_description)

        self._current_task_id = f"task_{self.memory_size}"
        self._step_history = []
        self.mpc_state = MPCState(
            task_goal=task_main,
            task_type="",
            tokens_budget=8000,
            latency_budget_ms=30000,
        )
        self._sync_mpc_state()

        return mas_msg

    def move_memory_state(self, action: str, observation: str, **kwargs) -> None:
        super().move_memory_state(action, observation, **kwargs)

        reward = kwargs.get("reward", 0.0)
        done = kwargs.get("done", False)
        self._step_history.append({
            "action": action, "observation": observation,
            "reward": reward, "done": done,
        })

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

        # Store in retrieval memory
        self.retrieval.write_experience(
            task_goal=task_main,
            trajectory=trajectory[:800],
            success=label,
            goal_type="",
            metadata={"steps": len(self._step_history), "feedback": (feedback or "")[:300]},
        )

        # Buffer for skill consolidation
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

        # MPC maintenance
        self._sync_mpc_state()
        maintenance = self.mpc.plan_maintenance(self.mpc_state)
        for ma in maintenance:
            if ma.action_type == MemoryActionType.SKILL_CONSOLIDATE:
                self.skill_mem.consolidate()
            elif ma.action_type == MemoryActionType.EVICT_CACHE:
                self.cache._expire()

        # Cache successful pattern
        if label:
            summary = f"Goal: {task_main}\nSteps: {len(actions)}\nActions: {'; '.join(actions[:10])}"
            entry = MemoryEntry(
                key=f"success_{self.memory_size}",
                content=summary,
                importance=1.5,
                source="success_cache",
            )
            self.cache.write(entry)

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
        MPC-guided retrieval: the controller decides how to combine
        cache / retrieval / skill results.
        """
        self._sync_mpc_state()
        self.mpc_state.current_observation = query_task

        successful_trajs: List[MASMessage] = []
        failed_trajs: List[MASMessage] = []
        insights: List[str] = []
        skills: List[Any] = []

        # MPC decides retrieval strategy
        action = self.mpc.select_action(self.mpc_state, query_task)
        self.mpc_state.record_step(StepRecord(
            step=0, action_type=action.action_type,
            query=query_task[:200],
        ))

        # 1. Always try cache first (near-zero cost)
        t0 = time.time()
        cached = self.cache.read(query_task)
        cache_ms = (time.time() - t0) * 1000
        if cached:
            insights.append(f"[Cached] {cached[0].content[:300]}")

        # 2. Retrieve from experience memory
        t0 = time.time()
        experiences = self.retrieval.read(query_task, top_k=successful_topk + 3)
        ret_ms = (time.time() - t0) * 1000

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

        # 3. Retrieve and format insights from retrieval hits
        for exp in experiences[:insight_topk]:
            if exp.source == "experience" and exp.content:
                snippet = exp.content.split("\n")[0][:200]
                if snippet not in insights:
                    insights.append(snippet)
        insights = insights[:insight_topk]

        # 4. Retrieve matching skills
        t0 = time.time()
        skill_entries = self.skill_mem.read(query_task, top_k=skill_topk)
        skill_ms = (time.time() - t0) * 1000

        for entry in skill_entries:
            skill_data = entry.metadata.get("skill")
            if skill_data:
                hmf_skill = HMFSkill.from_dict(skill_data)
                skills.append(_SkillProxy(hmf_skill))

        return (successful_trajs, failed_trajs, insights, skills)

    def backward(self, reward, **kwargs) -> None:
        pass

    # ======================== Stats ========================

    def get_stats(self) -> Dict[str, Any]:
        return {
            "memory_size": self.memory_size,
            "cache": self.cache.stats(),
            "retrieval": self.retrieval.stats(),
            "skill": self.skill_mem.stats(),
            "mpc": self.mpc.stats(),
        }
