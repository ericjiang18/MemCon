"""
HMF Agent — main orchestration layer.

Ties together:
  - Three memory substrates  (cache / retrieval / skill)
  - MPC controller           (adaptive memory action selection)
  - LLM backbone             (task reasoning)
  - Trajectory manager       (execution trace collection)

Lifecycle per task:
  1. init_task        → set goal, query all memory layers via MPC
  2. step             → MPC decides memory action → execute → env step
  3. finish_task      → write experience, update skills, run maintenance
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from ..config import HMFConfig
from ..memory.base import MemoryActionType, MemoryEntry
from ..memory.cache_memory import CacheMemory
from ..memory.retrieval_memory import RetrievalMemory
from ..memory.skill_memory import SkillMemory, Skill, TrajectoryBuffer
from ..mpc.controller import MPCController
from ..mpc.cost_model import CostModel
from ..mpc.state import MPCState, StepRecord
from .trajectory import TrajectoryManager, Step


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class HMFAgent:
    """
    Hierarchical Memory Framework agent with MPC-controlled memory access.
    Designed to plug into any agentic loop (LangGraph, Agent-Framework, Lobster,
    or a standalone Think-Act loop).
    """

    def __init__(self, config: HMFConfig):
        self.cfg = config
        self._setup_llm()
        self._setup_memory()
        self._setup_mpc()

        self.trajectory = TrajectoryManager()
        self.mpc_state = MPCState(
            tokens_budget=config.mpc.token_budget,
            latency_budget_ms=config.mpc.latency_budget_ms,
        )

        self._task_count = 0
        self._cumulative_stats: Dict[str, Any] = {
            "tasks": 0, "successes": 0, "total_tokens": 0,
            "cache_hits": 0, "skill_uses": 0,
        }

    def _setup_llm(self):
        from dotenv import load_dotenv
        load_dotenv()

        api_key = self.cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
        api_base = self.cfg.api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = self.cfg.model_name

        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer(self.cfg.embedding_model)

    def _embed_fn(self, text: str) -> List[float]:
        return self._embed_model.encode(text, normalize_embeddings=True).tolist()

    def _llm_call(self, prompt: str, system: str = "", **kw) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        params = dict(
            model=self._model,
            messages=messages,
            temperature=kw.get("temperature", 0.3),
        )
        tok_limit = kw.get("max_tokens", 2048)
        if any(tag in self._model for tag in ("gpt-5", "o3", "o4")):
            params["max_completion_tokens"] = tok_limit
        else:
            params["max_tokens"] = tok_limit
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    def _llm_fn_simple(self, prompt: str) -> str:
        """Thin wrapper for components that need a Callable[[str], str]."""
        return self._llm_call(prompt)

    def _setup_memory(self):
        base_dir = self.cfg.working_dir
        os.makedirs(base_dir, exist_ok=True)

        self.cache = CacheMemory(
            config=self.cfg.cache,
            embed_fn=self._embed_fn,
        )
        self.retrieval = RetrievalMemory(
            config=self.cfg.retrieval,
            embed_fn=self._embed_fn,
            persist_dir=os.path.join(base_dir, "retrieval"),
        )
        self.skill_mem = SkillMemory(
            config=self.cfg.skill,
            embed_fn=self._embed_fn,
            llm_fn=self._llm_fn_simple,
            persist_dir=os.path.join(base_dir, "skills"),
        )

    def _setup_mpc(self):
        self.cost_model = CostModel(self.cfg.mpc)
        self.mpc = MPCController(
            config=self.cfg.mpc,
            cost_model=self.cost_model,
            llm_fn=self._llm_fn_simple if self.cfg.mpc.use_llm_scoring else None,
        )

    def _sync_mpc_state(self):
        self.mpc_state.snapshot_memory_stats(
            self.cache.stats(),
            self.retrieval.stats(),
            self.skill_mem.stats(),
        )

    # ======================== Task Lifecycle ========================

    def init_task(self, task_goal: str, task_type: str = "", task_description: str = "") -> Dict[str, Any]:
        """
        Begin a new task. Returns a context dict with retrieved memories
        that should be injected into the agent's prompt.
        """
        self._task_count += 1
        self.trajectory.reset(task_goal, task_type)
        self.mpc_state = MPCState(
            task_goal=task_goal,
            task_type=task_type,
            tokens_budget=self.cfg.mpc.token_budget,
            latency_budget_ms=self.cfg.mpc.latency_budget_ms,
        )
        self._sync_mpc_state()

        context: Dict[str, Any] = {
            "experiences": [],
            "skills": [],
            "cached": [],
            "insights": [],
        }

        # MPC decides the initial memory query strategy
        query = task_goal
        if task_description:
            query = f"{task_goal}\n{task_description}"

        # Try cache first
        t0 = time.time()
        cached = self.cache.read(query)
        cache_ms = (time.time() - t0) * 1000
        if cached:
            context["cached"] = [e.content for e in cached]
            self.mpc_state.record_step(StepRecord(
                step=0, action_type=MemoryActionType.CACHE_READ,
                query=query, result_summary=f"cache hit: {len(cached)} entries",
                token_cost=0, latency_ms=cache_ms, cache_hit=True,
            ))
            self._cumulative_stats["cache_hits"] += 1

        # Retrieve past experiences
        t0 = time.time()
        experiences = self.retrieval.read(query, top_k=self.cfg.retrieval.top_k)
        ret_ms = (time.time() - t0) * 1000
        if experiences:
            context["experiences"] = [e.content for e in experiences]
            self.mpc_state.record_step(StepRecord(
                step=0, action_type=MemoryActionType.RETRIEVE,
                query=query, result_summary=f"retrieved {len(experiences)} experiences",
                token_cost=_estimate_tokens(" ".join(e.content for e in experiences)),
                latency_ms=ret_ms,
            ))

        # Retrieve matching skills
        t0 = time.time()
        skill_entries = self.skill_mem.read(query, top_k=3)
        skill_ms = (time.time() - t0) * 1000
        if skill_entries:
            context["skills"] = [e.content for e in skill_entries]
            self._cumulative_stats["skill_uses"] += 1
            self.mpc_state.record_step(StepRecord(
                step=0, action_type=MemoryActionType.SKILL_INVOKE,
                query=query, result_summary=f"matched {len(skill_entries)} skills",
                token_cost=_estimate_tokens(" ".join(e.content for e in skill_entries)),
                latency_ms=skill_ms,
                skill_used=skill_entries[0].key if skill_entries else None,
            ))

        self._sync_mpc_state()
        return context

    def step_decision(self, observation: str) -> MemoryAction:
        """
        Let the MPC controller decide what memory action to take
        given the current observation. Returns the chosen MemoryAction.
        """
        self.mpc_state.current_observation = observation
        self._sync_mpc_state()
        from ..memory.base import MemoryAction as MA
        return self.mpc.select_action(self.mpc_state, observation)

    def execute_memory_action(self, action_type: MemoryActionType, query: str = "") -> Dict[str, Any]:
        """Execute a specific memory action and return results."""
        t0 = time.time()
        result: Dict[str, Any] = {"action": action_type.name, "data": []}

        if action_type == MemoryActionType.CACHE_READ:
            entries = self.cache.read(query)
            result["data"] = [e.content for e in entries]
            result["hit"] = len(entries) > 0

        elif action_type == MemoryActionType.CACHE_WRITE:
            entry = MemoryEntry(
                key=f"cache_{int(time.time()*1000)}",
                content=query,
            )
            self.cache.write(entry)
            result["written"] = True

        elif action_type == MemoryActionType.RETRIEVE:
            entries = self.retrieval.read(query)
            result["data"] = [e.content for e in entries]

        elif action_type == MemoryActionType.RETRIEVE_WRITE:
            entry = MemoryEntry(
                key=f"ret_{int(time.time()*1000)}",
                content=query,
                source="step_observation",
            )
            self.retrieval.write(entry)
            result["written"] = True

        elif action_type == MemoryActionType.SKILL_INVOKE:
            entries = self.skill_mem.read(query, top_k=1)
            if entries:
                result["data"] = [entries[0].content]
                result["skill_id"] = entries[0].key

        elif action_type == MemoryActionType.SKILL_CONSOLIDATE:
            new_skills = self.skill_mem.consolidate()
            result["new_skills"] = len(new_skills)

        elif action_type == MemoryActionType.EVICT_CACHE:
            if self.cache.size() > 0:
                self.cache._evict_lru()

        elif action_type == MemoryActionType.EVICT_RETRIEVAL:
            if self.retrieval.size() > 0:
                self.retrieval._evict_lowest()

        elif action_type == MemoryActionType.LLM_GENERATE:
            pass  # handled by the outer loop

        elapsed_ms = (time.time() - t0) * 1000
        result["latency_ms"] = elapsed_ms
        result["token_cost"] = self.cost_model.estimate_tokens(action_type)

        self.mpc_state.record_step(StepRecord(
            step=self.mpc_state.step_number,
            action_type=action_type,
            query=query[:200],
            result_summary=str(result.get("data", ""))[:200],
            token_cost=result["token_cost"],
            latency_ms=elapsed_ms,
            cache_hit=result.get("hit", False),
            skill_used=result.get("skill_id"),
        ))

        return result

    def record_step(
        self,
        action: str,
        observation: str,
        reward: float = 0.0,
        done: bool = False,
        is_think: bool = False,
        token_cost: int = 0,
        latency_ms: float = 0.0,
    ):
        """Record an execution step in the trajectory."""
        step = Step(
            action=action,
            observation=observation,
            reward=reward,
            done=done,
            is_think=is_think,
            token_cost=token_cost,
            latency_ms=latency_ms,
        )
        self.trajectory.add_step(step)

        # Cache the observation for future reuse
        if not is_think and observation:
            cache_key = f"obs_{self.mpc_state.step_number}"
            entry = MemoryEntry(
                key=cache_key,
                content=f"Action: {action}\nResult: {observation}",
                source="step_cache",
            )
            self.cache.write(entry)

        if done and not reward:
            self.mpc_state.consecutive_failures += 1
        elif done and reward:
            self.mpc_state.consecutive_failures = 0

    def finish_task(self, success: bool, feedback: str = "") -> Dict[str, Any]:
        """
        End-of-task processing:
          1. Store trajectory in retrieval memory
          2. Buffer trajectory for skill consolidation
          3. Evolve used skills
          4. Run MPC maintenance
          5. Cache successful approach for reuse
        """
        self.trajectory.success = success

        # 1. Store experience in retrieval memory
        self.retrieval.write_experience(
            task_goal=self.trajectory.task_goal,
            trajectory=self.trajectory.summary(),
            success=success,
            goal_type=self.trajectory.task_type,
            metadata={"steps": len(self.trajectory.physical_steps()), "feedback": feedback[:300]},
        )

        # 2. Buffer for skill consolidation
        self.skill_mem.record_trajectory(TrajectoryBuffer(
            task_goal=self.trajectory.task_goal,
            goal_type=self.trajectory.task_type,
            actions=self.trajectory.action_list(),
            observations=self.trajectory.observation_list(),
            success=success,
            total_steps=len(self.trajectory.physical_steps()),
        ))

        # 3. Evolve skills that were used
        for rec in self.mpc_state.step_history:
            if rec.skill_used:
                self.skill_mem.evolve_skill(rec.skill_used, feedback, success)

        # 4. MPC maintenance (eviction, consolidation)
        self._sync_mpc_state()
        maintenance_actions = self.mpc.plan_maintenance(self.mpc_state)
        for ma in maintenance_actions:
            self.execute_memory_action(ma.action_type, ma.query)

        # 5. Cache successful approach
        if success:
            summary = self.trajectory.summary()
            entry = MemoryEntry(
                key=f"success_{self._task_count}",
                content=f"Successful approach:\n{summary}",
                importance=1.5,
                source="success_cache",
            )
            self.cache.write(entry)

        # Update cumulative stats
        self._cumulative_stats["tasks"] += 1
        if success:
            self._cumulative_stats["successes"] += 1
        self._cumulative_stats["total_tokens"] += self.trajectory.total_tokens

        return self.get_stats()

    # ======================== Prompt Construction ========================

    def build_memory_context(self, context: Dict[str, Any]) -> str:
        """Format retrieved memory context for injection into the agent prompt."""
        parts = []

        if context.get("cached"):
            parts.append("=== Cached Results ===")
            for c in context["cached"][:2]:
                parts.append(c[:500])

        if context.get("experiences"):
            parts.append("\n=== Relevant Past Experiences ===")
            for e in context["experiences"][:3]:
                parts.append(e[:500])

        if context.get("skills"):
            parts.append("\n=== Available Skills ===")
            for s in context["skills"][:2]:
                parts.append(s[:500])

        return "\n".join(parts) if parts else ""

    # ======================== Stats ========================

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cumulative": dict(self._cumulative_stats),
            "cache": self.cache.stats(),
            "retrieval": self.retrieval.stats(),
            "skill": self.skill_mem.stats(),
            "mpc": self.mpc.stats(),
            "current_task": {
                "steps": len(self.trajectory.steps),
                "tokens": self.trajectory.total_tokens,
                "success": self.trajectory.success,
            },
        }
