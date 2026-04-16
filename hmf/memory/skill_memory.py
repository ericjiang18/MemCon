"""
Skill Memory — reusable procedural knowledge distilled from trajectories.

Each Skill is a parameterized program with:
  - preconditions  (when to apply)
  - steps          (what to do)
  - postconditions (expected outcome)
  - statistics     (success rate, usage count, avg steps)

Consolidation: the MPC controller triggers trajectory → skill distillation
when enough similar successful trajectories accumulate.

Evolution: on failure the skill steps are refined via an LLM call;
on repeated failure the skill is deactivated.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..config import SkillMemoryConfig
from .base import MemoryEntry, MemorySubstrate


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    preconditions: List[str]
    steps: List[str]
    postconditions: List[str]
    goal_pattern: str = ""

    success_count: int = 0
    failure_count: int = 0
    usage_count: int = 0
    avg_steps: float = 0.0
    active: bool = True
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def total_uses(self) -> int:
        return self.success_count + self.failure_count

    def record_outcome(self, success: bool, steps_taken: int = 0):
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        if steps_taken > 0:
            alpha = 0.3
            self.avg_steps = alpha * steps_taken + (1 - alpha) * self.avg_steps
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "postconditions": self.postconditions,
            "goal_pattern": self.goal_pattern,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "usage_count": self.usage_count,
            "avg_steps": self.avg_steps,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryBuffer:
    """Holds raw trajectories pending consolidation into skills."""
    task_goal: str
    goal_type: str
    actions: List[str]
    observations: List[str]
    success: bool
    total_steps: int
    timestamp: float = field(default_factory=time.time)


class SkillMemory(MemorySubstrate):
    """
    Procedural knowledge store with trajectory-to-skill consolidation
    and closed-loop skill evolution.
    """

    def __init__(
        self,
        config: SkillMemoryConfig,
        embed_fn: Callable[[str], List[float]],
        llm_fn: Optional[Callable] = None,
        persist_dir: Optional[str] = None,
    ):
        self.cfg = config
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.persist_dir = persist_dir

        self._skills: Dict[str, Skill] = {}
        self._trajectory_buffer: List[TrajectoryBuffer] = []
        self._load()

    def read(self, query: str, top_k: int = 3, **kw) -> List[MemoryEntry]:
        """Return matching skills as MemoryEntry objects."""
        if not self._skills:
            return []
        q_emb = self.embed_fn(query)
        scored = []
        for skill in self._skills.values():
            if not skill.active:
                continue
            if skill.embedding is None:
                skill.embedding = self.embed_fn(skill.description)
            sim = self._cosine(q_emb, skill.embedding)
            utility = skill.success_rate * (1 + 0.1 * skill.usage_count)
            score = 0.6 * sim + 0.4 * utility
            scored.append((skill, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for skill, score in scored[:top_k]:
            entry = MemoryEntry(
                key=skill.skill_id,
                content=self._format_skill(skill),
                importance=score,
                source="skill",
                metadata={"skill": skill.to_dict(), "match_score": score},
            )
            results.append(entry)
        return results

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def get_active_skills(self) -> List[Skill]:
        return [s for s in self._skills.values() if s.active]

    def write(self, entry: MemoryEntry) -> bool:
        """Write a skill from a MemoryEntry (for interface compliance)."""
        skill_dict = entry.metadata.get("skill")
        if skill_dict:
            skill = Skill.from_dict(skill_dict)
            return self.add_skill(skill)
        return False

    def add_skill(self, skill: Skill) -> bool:
        if len(self._skills) >= self.cfg.max_skills:
            self._evict_worst_skill()
        if skill.embedding is None:
            skill.embedding = self.embed_fn(skill.description)
        self._skills[skill.skill_id] = skill
        self._persist()
        return True

    def record_trajectory(self, traj: TrajectoryBuffer):
        self._trajectory_buffer.append(traj)
        if len(self._trajectory_buffer) > 500:
            self._trajectory_buffer = self._trajectory_buffer[-500:]

    def consolidate(self, goal_type: Optional[str] = None) -> List[Skill]:
        """
        Distill buffered trajectories into reusable skills.
        Triggered by the MPC controller when enough similar successes accumulate.
        """
        if not self.llm_fn:
            return []

        success_trajs = [
            t for t in self._trajectory_buffer
            if t.success and (goal_type is None or t.goal_type == goal_type)
        ]
        if len(success_trajs) < self.cfg.consolidation_threshold:
            return []

        groups = self._cluster_trajectories(success_trajs)
        new_skills = []
        for group in groups:
            if len(group) < self.cfg.consolidation_threshold:
                continue
            skill = self._distill_skill(group)
            if skill:
                self.add_skill(skill)
                new_skills.append(skill)
                for t in group:
                    if t in self._trajectory_buffer:
                        self._trajectory_buffer.remove(t)

        return new_skills

    def evolve_skill(self, skill_id: str, feedback: str, success: bool) -> bool:
        """Update skill steps based on execution feedback."""
        skill = self._skills.get(skill_id)
        if not skill or not self.cfg.evolution_enabled:
            return False

        skill.record_outcome(success)

        if skill.success_rate < 0.2 and skill.total_uses >= 5:
            skill.active = False
            self._persist()
            return True

        if not success and self.llm_fn:
            prompt = (
                f"A procedural skill failed. Update its steps.\n\n"
                f"Skill: {skill.name}\n"
                f"Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(skill.steps)) +
                f"\n\nFeedback: {feedback}\n\n"
                f"Provide ONLY the updated step list, one step per line, prefixed with numbers."
            )
            try:
                response = self.llm_fn(prompt)
                new_steps = self._parse_steps(response)
                if new_steps:
                    skill.steps = new_steps
                    skill.embedding = self.embed_fn(skill.description)
                    skill.updated_at = time.time()
            except Exception:
                pass

        self._persist()
        return True

    def evict(self, key: str) -> bool:
        if key in self._skills:
            del self._skills[key]
            self._persist()
            return True
        return False

    def size(self) -> int:
        return len(self._skills)

    def token_footprint(self) -> int:
        total = 0
        for s in self._skills.values():
            text = self._format_skill(s)
            total += max(1, len(text) // 4)
        return total

    def stats(self) -> Dict[str, Any]:
        active = sum(1 for s in self._skills.values() if s.active)
        avg_sr = (
            np.mean([s.success_rate for s in self._skills.values()])
            if self._skills else 0.0
        )
        return {
            "type": "skill",
            "total_skills": self.size(),
            "active_skills": active,
            "avg_success_rate": float(avg_sr),
            "trajectory_buffer": len(self._trajectory_buffer),
            "token_footprint": self.token_footprint(),
        }

    def _cluster_trajectories(self, trajs: List[TrajectoryBuffer]) -> List[List[TrajectoryBuffer]]:
        """Group trajectories by goal type for consolidation."""
        groups: Dict[str, List[TrajectoryBuffer]] = {}
        for t in trajs:
            groups.setdefault(t.goal_type or "general", []).append(t)
        return list(groups.values())

    def _distill_skill(self, group: List[TrajectoryBuffer]) -> Optional[Skill]:
        """Use LLM to distill a group of similar trajectories into one skill."""
        if not self.llm_fn:
            return None

        examples = ""
        for i, t in enumerate(group[:5]):
            actions_str = "\n".join(f"    {a}" for a in t.actions[:15])
            examples += f"\n  Example {i+1} (goal: {t.task_goal}):\n{actions_str}\n"

        prompt = (
            f"Distill the following successful task trajectories into ONE reusable skill.\n\n"
            f"Trajectories:{examples}\n"
            f"Produce a JSON object with keys: name, description, preconditions (list), "
            f"steps (list), postconditions (list), goal_pattern.\n"
            f"Be concise. Output ONLY valid JSON."
        )

        try:
            response = self.llm_fn(prompt)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                skill_id = f"skill_{int(time.time())}_{hash(data.get('name', '')) % 10000}"
                return Skill(
                    skill_id=skill_id,
                    name=data.get("name", "unnamed"),
                    description=data.get("description", ""),
                    preconditions=data.get("preconditions", []),
                    steps=data.get("steps", []),
                    postconditions=data.get("postconditions", []),
                    goal_pattern=data.get("goal_pattern", group[0].goal_type),
                )
        except Exception as e:
            print(f"[SkillMemory] distill error: {e}")
        return None

    def _evict_worst_skill(self):
        if not self._skills:
            return
        worst_id = min(
            self._skills,
            key=lambda k: self._skills[k].success_rate * (1 + self._skills[k].usage_count * 0.1),
        )
        del self._skills[worst_id]

    @staticmethod
    def _cosine(a, b) -> float:
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

    @staticmethod
    def _format_skill(skill: Skill) -> str:
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(skill.steps))
        return (
            f"[Skill: {skill.name}] (SR={skill.success_rate:.0%}, uses={skill.usage_count})\n"
            f"{skill.description}\n"
            f"Preconditions: {', '.join(skill.preconditions) if skill.preconditions else 'any'}\n"
            f"Steps:\n{steps}\n"
            f"Expected outcome: {', '.join(skill.postconditions) if skill.postconditions else 'task completion'}"
        )

    @staticmethod
    def _parse_steps(text: str) -> List[str]:
        import re
        lines = text.strip().split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            cleaned = re.sub(r"^\d+[.)]\s*", "", line)
            if cleaned:
                steps.append(cleaned)
        return steps

    def _persist(self):
        if not self.persist_dir:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        path = os.path.join(self.persist_dir, "skills.json")
        data = [s.to_dict() for s in self._skills.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        buf_path = os.path.join(self.persist_dir, "trajectory_buffer.json")
        buf_data = [
            {
                "task_goal": t.task_goal,
                "goal_type": t.goal_type,
                "actions": t.actions[:30],
                "success": t.success,
                "total_steps": t.total_steps,
                "timestamp": t.timestamp,
            }
            for t in self._trajectory_buffer[-200:]
        ]
        with open(buf_path, "w") as f:
            json.dump(buf_data, f)

    def _load(self):
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, "skills.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                for d in data:
                    skill = Skill.from_dict(d)
                    skill.embedding = self.embed_fn(skill.description)
                    self._skills[skill.skill_id] = skill
            except Exception as e:
                print(f"[SkillMemory] load skills error: {e}")

        buf_path = os.path.join(self.persist_dir, "trajectory_buffer.json")
        if os.path.exists(buf_path):
            try:
                with open(buf_path) as f:
                    buf_data = json.load(f)
                for d in buf_data:
                    self._trajectory_buffer.append(TrajectoryBuffer(
                        task_goal=d["task_goal"],
                        goal_type=d["goal_type"],
                        actions=d["actions"],
                        observations=[],
                        success=d["success"],
                        total_steps=d["total_steps"],
                        timestamp=d.get("timestamp", time.time()),
                    ))
            except Exception as e:
                print(f"[SkillMemory] load buffer error: {e}")
