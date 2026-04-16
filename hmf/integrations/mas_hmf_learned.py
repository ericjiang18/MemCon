"""
HMF-Learned — Same three memory substrates, with LEARNED MPC controller.

Instead of hand-tuned cost weights (α, β, γ, δ), uses a contextual bandit
that learns Q(state_features, action) online from task success/failure signals.

Key differences from HMFMemory:
  - No hand-tuned cost model
  - UCB exploration for action selection
  - Q-values updated after each task via exponential moving average
  - Learns which memory action works best in which state
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
from ..memory.base import MemoryEntry, MemoryActionType
from ..memory.cache_memory import CacheMemory
from ..memory.retrieval_memory import RetrievalMemory
from ..memory.skill_memory import SkillMemory as HMFSkillMemory, Skill as HMFSkill, TrajectoryBuffer
from ..mpc.learned_controller import LearnedMPCController, LearnedControllerConfig
from ..mpc.state import MPCState, StepRecord


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
class HMFLearnedMemory(MASMemoryBase):
    """
    Three-layer memory with learned (contextual bandit) controller.
    Q-values persist across tasks and improve over time.
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
        hmf_dir = os.path.join(self.persist_dir, "hmf_learned")

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

        self.mpc = LearnedMPCController(
            config=LearnedControllerConfig(
                learning_rate=0.15,
                ucb_c=1.5,
                persist_path=os.path.join(hmf_dir, "learned_q.json"),
            )
        )
        self.mpc_state = MPCState(tokens_budget=8000, latency_budget_ms=30000)

        self.memory_size: int = 0
        self._step_history: List[Dict[str, Any]] = []
        self._all_memories: List[MASMessage] = []

        print(f"[HMFLearned] initialized at {hmf_dir}")

    def _sync_mpc_state(self):
        self.mpc_state.snapshot_memory_stats(
            self.cache.stats(),
            self.retrieval.stats(),
            self.skill_mem.stats(),
        )

    # ======================== Task Lifecycle ========================

    def init_task_context(self, task_main, task_description=None, **kw):
        self._step_history = []
        self.mpc_state = MPCState(
            task_goal=task_main, tokens_budget=8000, latency_budget_ms=30000,
        )
        self._sync_mpc_state()
        return super().init_task_context(task_main, task_description)

    def move_memory_state(self, action: str, observation: str, **kwargs):
        super().move_memory_state(action, observation, **kwargs)
        self._step_history.append({
            "action": action, "observation": observation,
            "reward": kwargs.get("reward", 0.0),
            "done": kwargs.get("done", False),
        })
        is_think = action.lower().startswith(("think", "thought"))
        if not is_think and observation:
            self.cache.write(MemoryEntry(
                key=f"step_{len(self._step_history)}",
                content=f"Action: {action}\nResult: {observation[:300]}",
                source="step_cache",
            ))

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        task_main = self.current_task_context.task_main
        trajectory = self.current_task_context.task_trajectory or ""

        # Always write experience
        self.retrieval.write_experience(
            task_goal=task_main, trajectory=trajectory[:800],
            success=label, goal_type="",
            metadata={"steps": len(self._step_history), "feedback": (feedback or "")[:300]},
        )

        actions = [h["action"] for h in self._step_history if not h["action"].lower().startswith(("think", "thought"))]
        observations = [h["observation"] for h in self._step_history]
        self.skill_mem.record_trajectory(TrajectoryBuffer(
            task_goal=task_main, goal_type="",
            actions=actions, observations=observations,
            success=label, total_steps=len(actions),
        ))

        # Learned maintenance
        self._sync_mpc_state()
        for ma in self.mpc.plan_maintenance(self.mpc_state):
            if ma.action_type == MemoryActionType.SKILL_CONSOLIDATE:
                self.skill_mem.consolidate()
            elif ma.action_type == MemoryActionType.EVICT_CACHE:
                self.cache._expire()

        if label:
            self.cache.write(MemoryEntry(
                key=f"success_{self.memory_size}",
                content=f"Goal: {task_main}\nSteps: {len(actions)}\nActions: {'; '.join(actions[:10])}",
                importance=1.5, source="success_cache",
            ))

        # KEY: update learned Q-values from task outcome
        reward = 1.0 if label else -0.5
        step_count = len(self._step_history)
        if step_count > 0:
            token_efficiency = max(0, 1.0 - step_count / 30.0)
            reward += 0.3 * token_efficiency if label else 0
        self.mpc.update(reward)

        self._step_history = []
        return super().save_task_context(label=label, feedback=feedback)

    # ======================== Cross-Trial Memory ========================

    def add_memory(self, mas_message: MASMessage):
        self._all_memories.append(mas_message)
        self.memory_size = len(self._all_memories)

    def retrieve_memory(
        self, query_task="", successful_topk=2, failed_topk=0,
        insight_topk=5, skill_topk=3, threshold=0.3, **kw,
    ) -> Tuple[List[MASMessage], List[MASMessage], List[str], List[Any]]:
        """Learned controller decides retrieval strategy via UCB."""
        self._sync_mpc_state()
        self.mpc_state.current_observation = query_task

        successful_trajs, failed_trajs, insights, skills = [], [], [], []

        # Learned action selection
        action = self.mpc.select_action(self.mpc_state, query_task)

        # Execute based on learned decision — but always do at least cache + retrieval
        # (the learned part is WHICH to emphasize and whether to also do skill)

        # Cache (always, near-zero cost)
        cached = self.cache.read(query_task)
        if cached:
            insights.append(f"[Cached] {cached[0].content[:300]}")

        # Retrieval
        if action.action_type in (
            MemoryActionType.RETRIEVE, MemoryActionType.LLM_GENERATE,
            MemoryActionType.CACHE_READ, MemoryActionType.NO_OP,
        ):
            experiences = self.retrieval.read(query_task, top_k=successful_topk + 3)
            for exp in experiences:
                if exp.metadata.get("success", False):
                    successful_trajs.append(MASMessage(
                        task_main=exp.metadata.get("task_goal", query_task),
                        task_description=exp.content[:500],
                        task_trajectory=exp.content, label=True,
                    ))
                else:
                    insights.append(f"[Past failure] {exp.content[:200]}")
            successful_trajs = successful_trajs[:successful_topk]

            for exp in experiences[:insight_topk]:
                if exp.source == "experience" and exp.content:
                    snippet = exp.content.split("\n")[0][:200]
                    if snippet not in insights:
                        insights.append(snippet)
            insights = insights[:insight_topk]

        # Skill — only if learned controller selects it or we have good skills
        if action.action_type == MemoryActionType.SKILL_INVOKE or self.mpc_state.skill_avg_sr > 0.5:
            skill_entries = self.skill_mem.read(query_task, top_k=skill_topk)
            for entry in skill_entries:
                skill_data = entry.metadata.get("skill")
                if skill_data:
                    skills.append(_SkillProxy(HMFSkill.from_dict(skill_data)))

        return (successful_trajs, failed_trajs, insights, skills)

    def backward(self, reward, **kwargs):
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "memory_size": self.memory_size,
            "cache": self.cache.stats(),
            "retrieval": self.retrieval.stats(),
            "skill": self.skill_mem.stats(),
            "learned_mpc": self.mpc.stats(),
        }
