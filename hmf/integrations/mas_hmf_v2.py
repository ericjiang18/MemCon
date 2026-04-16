"""
HMF v2 — Improved hierarchical memory that incorporates G-Memory's key advantages:

  1. Graph-structured task layer (NetworkX) — captures task-type relationships
     via k-hop expansion, not just flat vector similarity
  2. Scored InsightsManager — accumulates rules with success/failure scoring,
     periodic LLM-driven merge/prune (like G-Memory's InsightsManager)
  3. LLM post-task refine — extracts key steps and failure reasons
  4. Cache + Skill layers retained from HMF for fast reuse and procedural knowledge
  5. NO MPC overhead — uses proven static pipeline (cache → graph-retrieval → insights → skill)

Key difference from G-Memory: we ADD cache + skill layers ON TOP of the graph structure.
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
import pickle
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mas.memory.mas_memory.memory_base import MASMemoryBase
from mas.memory.common import MASMessage
from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc

from ..config import CacheMemoryConfig, SkillMemoryConfig
from ..memory.base import MemoryEntry
from ..memory.cache_memory import CacheMemory
from ..memory.skill_memory import SkillMemory as HMFSkillMemory, Skill as HMFSkill, TrajectoryBuffer


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


class _SkillProxy:
    def __init__(self, s: HMFSkill):
        self.skill_id = s.skill_id
        self.name = s.name
        self.description = s.description
        self.steps = s.steps
        self.preconditions = s.preconditions
        self.postconditions = s.postconditions
        self.success_rate = s.success_rate
        self.usage_count = s.usage_count
        self.active = s.active


# ====================== Task Graph ======================

class TaskGraph:
    """Lightweight NetworkX graph linking similar tasks for k-hop retrieval."""

    def __init__(self, embed_fn, persist_path: str, similarity_threshold: float = 0.65):
        self.embed_fn = embed_fn
        self.persist_path = persist_path
        self.sim_thresh = similarity_threshold

        self._embeddings: Dict[str, List[float]] = {}
        self._tasks: Dict[str, Dict[str, Any]] = {}

        import networkx as nx
        if os.path.exists(persist_path):
            with open(persist_path, "rb") as f:
                saved = pickle.load(f)
                self.graph = saved.get("graph", nx.Graph())
                self._embeddings = saved.get("embeddings", {})
                self._tasks = saved.get("tasks", {})
        else:
            self.graph = nx.Graph()

    def add_task(self, task_main: str, label: bool, trajectory: str = "", key_steps: str = ""):
        if task_main in self._embeddings:
            return

        emb = self.embed_fn(task_main)
        self._embeddings[task_main] = emb
        self._tasks[task_main] = {
            "label": label, "trajectory": trajectory[:600],
            "key_steps": key_steps, "timestamp": time.time(),
        }

        self.graph.add_node(task_main)
        for other, other_emb in self._embeddings.items():
            if other == task_main:
                continue
            sim = _cosine(emb, other_emb)
            if sim >= self.sim_thresh:
                self.graph.add_edge(task_main, other, weight=sim)

        self._save()

    def retrieve(self, query: str, top_k: int = 3, hop: int = 1) -> List[Dict[str, Any]]:
        if not self._embeddings:
            return []

        q_emb = self.embed_fn(query)
        scored = []
        for task_main, emb in self._embeddings.items():
            sim = _cosine(q_emb, emb)
            scored.append((task_main, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_nodes = [t for t, _ in scored[:top_k]]

        import networkx as nx
        expanded = set(top_nodes)
        for node in top_nodes:
            if node in self.graph:
                neighbors = nx.single_source_shortest_path_length(self.graph, node, cutoff=hop).keys()
                expanded.update(neighbors)

        results = []
        for node in expanded:
            info = self._tasks.get(node, {})
            if info:
                results.append({
                    "task_main": node,
                    "label": info.get("label"),
                    "trajectory": info.get("trajectory", ""),
                    "key_steps": info.get("key_steps", ""),
                    "sim": _cosine(q_emb, self._embeddings.get(node, q_emb)),
                })

        results.sort(key=lambda x: (-int(x.get("label", False)), -x["sim"]))
        return results

    def _save(self):
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump({"graph": self.graph, "embeddings": self._embeddings, "tasks": self._tasks}, f)


# ====================== Scored Insights ======================

class ScoredInsights:
    """Insight rules with success/failure scoring and periodic LLM merge."""

    def __init__(self, llm_fn, persist_path: str):
        self.llm_fn = llm_fn
        self.persist_path = persist_path
        self.insights: List[Dict[str, Any]] = []
        self._load()

    def add_insight(self, rule: str, task_main: str, success: bool):
        self.insights.append({
            "rule": rule,
            "score": 3 if success else 1,
            "positive_tasks": [task_main] if success else [],
            "negative_tasks": [] if success else [task_main],
        })
        self._prune()
        self._save()

    def backward(self, used_insights: List[str], success: bool):
        delta = 0.5 if success else -1.0
        for ins in self.insights:
            if ins["rule"] in used_insights:
                ins["score"] += delta
        self._prune()
        self._save()

    def query(self, task_query: str, task_mains: List[str] = None, top_k: int = 5) -> List[str]:
        if not self.insights:
            return []
        scored = []
        for ins in self.insights:
            relevance = 0
            if task_mains:
                relevance = sum(1 for t in task_mains if t in ins.get("positive_tasks", []))
            base = ins.get("score", 1) + relevance * 2
            scored.append((ins["rule"], base))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]

    def merge(self, every_n: int = 20):
        """LLM-driven merge of redundant insights."""
        if len(self.insights) < every_n:
            return
        rules = [ins["rule"] for ins in self.insights]
        if len(rules) <= 5:
            return
        prompt = (
            f"Merge the following {len(rules)} rules into at most {len(rules)//3 + 1} concise, non-redundant rules.\n\n"
            + "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))
            + "\n\nOutput numbered list only."
        )
        try:
            response = self.llm_fn(prompt)
            merged = re.findall(r"\d+\.\s+(.+?)(?=\n\d+\.|\Z)", response.strip(), re.DOTALL)
            if merged and len(merged) < len(rules):
                self.insights = [{"rule": r.strip(), "score": 2, "positive_tasks": [], "negative_tasks": []} for r in merged]
                self._save()
        except Exception:
            pass

    def _prune(self):
        self.insights = [i for i in self.insights if i.get("score", 0) > 0]

    def _save(self):
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(self.insights, f, indent=2)

    def _load(self):
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path) as f:
                    self.insights = json.load(f)
            except Exception:
                pass


# ====================== HMF v2 Memory ======================

_REFINE_PROMPT = """\
Task: {goal}
Outcome: {outcome}
Key actions: {actions}

Extract:
1. KEY_STEPS: The essential steps that led to success (or should have been taken).
2. INSIGHT: One concise rule about this task type.
3. AVOID: What to avoid (if failed).

Format:
KEY_STEPS: <numbered list>
INSIGHT: <one sentence>
AVOID: <one sentence or NONE>"""


@dataclass
class HMFv2Memory(MASMemoryBase):
    """
    HMF v2: Graph-enhanced retrieval + scored insights + cache + skills.

    Combines G-Memory's strengths (graph structure, insight scoring)
    with HMF's additions (cache layer, skill distillation).
    """

    def __post_init__(self):
        super().__post_init__()

        self._embed_fn = lambda text: self.embedding_func.embed_query(text)

        def _llm_fn(prompt: str) -> str:
            return self.llm_model(
                messages=[Message("system", "Be concise."), Message("user", prompt)],
                temperature=0.2, max_tokens=512,
            )

        self._llm_fn = _llm_fn
        v2_dir = os.path.join(self.persist_dir, "hmf_v2")
        os.makedirs(v2_dir, exist_ok=True)

        self.cache = CacheMemory(
            config=CacheMemoryConfig(max_entries=128, ttl_seconds=1200),
            embed_fn=self._embed_fn,
        )
        self.task_graph = TaskGraph(
            embed_fn=self._embed_fn,
            persist_path=os.path.join(v2_dir, "task_graph.pkl"),
            similarity_threshold=0.65,
        )
        self.scored_insights = ScoredInsights(
            llm_fn=self._llm_fn,
            persist_path=os.path.join(v2_dir, "insights.json"),
        )
        self.skill_mem = HMFSkillMemory(
            config=SkillMemoryConfig(consolidation_threshold=3),
            embed_fn=self._embed_fn,
            llm_fn=self._llm_fn,
            persist_dir=os.path.join(v2_dir, "skills"),
        )

        self.memory_size: int = 0
        self._step_history: List[Dict[str, Any]] = []
        self._all_memories: List[MASMessage] = []
        self._insights_cache: List[str] = []

        print(f"[HMFv2] initialized at {v2_dir}")

    # ======================== Task Lifecycle ========================

    def init_task_context(self, task_main, task_description=None, **kw):
        self._step_history = []
        self._insights_cache = []
        return super().init_task_context(task_main, task_description)

    def move_memory_state(self, action: str, observation: str, **kwargs):
        super().move_memory_state(action, observation, **kwargs)
        self._step_history.append({
            "action": action, "observation": observation,
            "reward": kwargs.get("reward", 0.0), "done": kwargs.get("done", False),
        })
        is_think = action.lower().startswith(("think", "thought"))
        if not is_think and observation:
            self.cache.write(MemoryEntry(
                key=f"s{len(self._step_history)}",
                content=f"Action: {action}\nResult: {observation[:300]}",
                source="step",
            ))

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        task_main = self.current_task_context.task_main
        trajectory = self.current_task_context.task_trajectory or ""

        actions = [h["action"] for h in self._step_history if not h["action"].lower().startswith(("think", "thought"))]

        # 1. LLM refine — extract key steps + insight
        key_steps, insight, avoid = self._llm_refine(task_main, label, actions)

        # 2. Add to task graph (with key_steps for richer retrieval)
        self.task_graph.add_task(task_main, label, trajectory[:600], key_steps)

        # 3. Add insight with scoring
        if insight:
            self.scored_insights.add_insight(insight, task_main, label)
        if avoid:
            self.scored_insights.add_insight(f"AVOID: {avoid}", task_main, False)

        # 4. Backward: update scores for insights used this task
        self.scored_insights.backward(self._insights_cache, label)

        # 5. Periodic insight merge
        if self.memory_size > 0 and self.memory_size % 15 == 0:
            self.scored_insights.merge()

        # 6. Buffer trajectory for skill consolidation
        observations = [h["observation"] for h in self._step_history]
        self.skill_mem.record_trajectory(TrajectoryBuffer(
            task_goal=task_main, goal_type="",
            actions=actions, observations=observations,
            success=label, total_steps=len(actions),
        ))
        if self.memory_size > 0 and self.memory_size % 5 == 0:
            self.skill_mem.consolidate()

        # 7. Cache successful approach
        if label:
            summary = f"Goal: {task_main}\nKey steps: {key_steps}\nActions: {'; '.join(actions[:8])}"
            self.cache.write(MemoryEntry(
                key=f"success_{self.memory_size}", content=summary,
                importance=1.5, source="success",
            ))

        self._step_history = []
        self._insights_cache = []
        return super().save_task_context(label=label, feedback=feedback)

    # ======================== Retrieval ========================

    def retrieve_memory(
        self, query_task="", successful_topk=2, failed_topk=0,
        insight_topk=5, skill_topk=3, threshold=0.3, **kw,
    ) -> Tuple[List[MASMessage], List[MASMessage], List[str], List[Any]]:

        successful_trajs, failed_trajs, insights, skills = [], [], [], []

        # Layer 1: Cache (instant)
        cached = self.cache.read(query_task)
        if cached:
            insights.append(f"[Cached approach] {cached[0].content[:300]}")

        # Layer 2: Graph-enhanced retrieval (k-hop expansion)
        graph_results = self.task_graph.retrieve(query_task, top_k=3, hop=1)
        related_task_mains = []
        for r in graph_results:
            related_task_mains.append(r["task_main"])
            if r.get("label") and r.get("trajectory"):
                msg = MASMessage(
                    task_main=r["task_main"],
                    task_description=r.get("trajectory", "")[:500],
                    task_trajectory=r.get("trajectory", ""),
                    label=True,
                )
                msg.add_extra_field("key_steps", r.get("key_steps", ""))
                successful_trajs.append(msg)

        successful_trajs = successful_trajs[:successful_topk]

        # Layer 3: Scored insights (with task-type relevance boosting)
        insight_rules = self.scored_insights.query(query_task, related_task_mains, top_k=insight_topk)
        insights.extend(insight_rules)
        self._insights_cache = list(insight_rules)

        # Layer 4: Skills
        skill_entries = self.skill_mem.read(query_task, top_k=skill_topk)
        for entry in skill_entries:
            skill_data = entry.metadata.get("skill")
            if skill_data:
                skills.append(_SkillProxy(HMFSkill.from_dict(skill_data)))

        return (successful_trajs, failed_trajs, insights[:insight_topk], skills)

    # ======================== Post-Task Refine ========================

    def _llm_refine(self, goal: str, success: bool, actions: List[str]) -> Tuple[str, str, str]:
        action_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions[:15]))
        prompt = _REFINE_PROMPT.format(
            goal=goal,
            outcome="SUCCESS" if success else "FAILURE",
            actions=action_str or "(none)",
        )
        try:
            resp = self._llm_fn(prompt)
            key_steps = ""
            insight = ""
            avoid = ""
            m = re.search(r"KEY_STEPS:\s*(.+?)(?=INSIGHT:|$)", resp, re.DOTALL | re.IGNORECASE)
            if m:
                key_steps = m.group(1).strip()
            m = re.search(r"INSIGHT:\s*(.+?)(?=AVOID:|$)", resp, re.DOTALL | re.IGNORECASE)
            if m:
                insight = m.group(1).strip()
            m = re.search(r"AVOID:\s*(.+)", resp, re.IGNORECASE)
            if m:
                v = m.group(1).strip()
                if v.upper() != "NONE":
                    avoid = v
            return key_steps, insight, avoid
        except Exception:
            return "", "", ""

    # ======================== Cross-Trial ========================

    def add_memory(self, mas_message: MASMessage):
        self._all_memories.append(mas_message)
        self.memory_size = len(self._all_memories)

    def backward(self, reward, **kwargs):
        pass
