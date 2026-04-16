"""
Skill-Conditioned Reinforcement Learning Framework for G-Memory++

Core idea: Q(goal_type, skill_id) — learn which skill works for which goal type.
Instead of evaluating individual actions, the RL evaluates whether to follow a
particular skill or act freely ("no_skill").

This is fundamentally different from the original Goal RL:
- Original: Q(s, a, g) over individual action patterns → weak signal
- New: Q(goal_type, skill_id) over skills → strong, interpretable signal
- At task start, RL selects the best skill to follow
- At task end, batch update based on success/failure

Also retains lightweight ExpRAG for experience storage (in integration layer).
"""

import os
import json
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import defaultdict
import pickle

from mas.llm import LLMCallable, Message
from .goal_module import StructuredGoal, GoalParser

# ================================ Constants ================================

NO_SKILL = "__no_skill__"

from enum import Enum

class ReplayStrategy(Enum):
    FUTURE = "future"
    FINAL = "final"
    EPISODE = "episode"
    RANDOM = "random"


# ================================ Data Classes ================================

@dataclass
class Experience:
    """A single experience tuple."""
    state: str
    action: str
    next_state: str
    reward: float
    done: bool
    goal: StructuredGoal

    achieved_goal: Optional[StructuredGoal] = None
    hindsight_reward: float = 0.0
    state_embedding: Optional[np.ndarray] = None
    goal_embedding: Optional[np.ndarray] = None
    step_idx: int = 0
    episode_id: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "done": self.done,
            "goal": self.goal.to_dict() if self.goal else None,
            "step_idx": self.step_idx,
            "episode_id": self.episode_id,
        }


@dataclass
class Episode:
    """A complete episode of experiences."""
    episode_id: str
    goal: StructuredGoal
    experiences: List[Experience] = field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False
    skill_id_used: str = NO_SKILL

    achieved_goals: List[StructuredGoal] = field(default_factory=list)
    total_steps: int = 0
    total_reward: float = 0.0

    def add_experience(self, exp: Experience):
        exp.episode_id = self.episode_id
        exp.step_idx = len(self.experiences)
        self.experiences.append(exp)
        self.total_steps += 1
        self.total_reward += exp.reward

    def get_trajectory_text(self) -> str:
        parts = []
        for exp in self.experiences:
            parts.append(f"Action: {exp.action}")
            parts.append(f"Obs: {exp.next_state[:150]}")
        return "\n".join(parts)


@dataclass
class SkillValueEntry:
    """Entry in the skill-conditioned value table."""
    goal_type: str
    skill_id: str
    q_value: float = 0.0
    total_uses: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.5
        return self.success_count / self.total_uses

    @property
    def confidence(self) -> float:
        """UCB-style confidence — higher when more data."""
        if self.total_uses == 0:
            return 0.0
        return min(self.total_uses / 10.0, 1.0)


# Keep old name for backward compat in __init__.py
GoalValueEntry = SkillValueEntry


# ================================ Skill-Conditioned Value Function ================================

class SkillConditionedValueFunction:
    """
    Q(goal_type, skill_id) — learns which skill works for which goal type.

    Much simpler and stronger signal than Q(s, a, g):
    - Key: (goal_type, skill_id)  where skill_id can be NO_SKILL
    - Updated at episode end with success=1.0 / failure=0.0
    - Uses exponential moving average for Q-values
    - UCB-style exploration bonus for under-explored skills
    """

    def __init__(
        self,
        learning_rate: float = 0.3,
        exploration_bonus: float = 0.2,
        working_dir: str = None,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.exploration_bonus = exploration_bonus
        self.working_dir = working_dir

        self.q_table: Dict[Tuple[str, str], SkillValueEntry] = {}
        self.update_count = 0

        self._load()

    def get_skill_value(self, goal_type: str, skill_id: str) -> float:
        key = (goal_type, skill_id)
        if key in self.q_table:
            return self.q_table[key].q_value
        return 0.0

    def get_best_skill(
        self,
        goal_type: str,
        available_skill_ids: List[str],
    ) -> Tuple[Optional[str], float]:
        """
        Select the best skill for a goal type, with exploration bonus.

        Returns:
            (skill_id or None, q_value)
            None means use no skill (free action).
        """
        candidates = available_skill_ids + [NO_SKILL]

        best_id = NO_SKILL
        best_score = -1.0

        for sid in candidates:
            key = (goal_type, sid)
            if key in self.q_table:
                entry = self.q_table[key]
                score = entry.q_value
                if entry.total_uses >= 1: # 3:
                    score += self.exploration_bonus * np.sqrt(
                        np.log(self.update_count + 1) / entry.total_uses
                    )
            else:
                score = 0.1 if sid == NO_SKILL else 0.0

            if score > best_score:
                best_score = score
                best_id = sid

        if best_id == NO_SKILL:
            return None, best_score
        return best_id, best_score

    def update(
        self,
        goal_type: str,
        skill_id: str,
        success: bool,
    ):
        """Update Q-value after episode completion."""
        key = (goal_type, skill_id)
        reward = 1.0 if success else 0.0

        if key not in self.q_table:
            self.q_table[key] = SkillValueEntry(
                goal_type=goal_type,
                skill_id=skill_id,
            )

        entry = self.q_table[key]
        entry.total_uses += 1
        if success:
            entry.success_count += 1

        # EMA update
        entry.q_value += self.learning_rate * (reward - entry.q_value)

        self.update_count += 1
        if self.update_count % 20 == 0:
            self._save()

    def get_statistics(self) -> Dict[str, Any]:
        goal_types = defaultdict(list)
        for (gt, sid), entry in self.q_table.items():
            goal_types[gt].append({
                "skill": sid,
                "q": round(entry.q_value, 3),
                "uses": entry.total_uses,
                "sr": round(entry.success_rate, 2),
            })
        return {
            "num_entries": len(self.q_table),
            "update_count": self.update_count,
            "per_goal_type": dict(goal_types),
        }

    def _save(self):
        if not self.working_dir:
            return
        os.makedirs(self.working_dir, exist_ok=True)
        path = os.path.join(self.working_dir, "skill_q_table.pkl")
        data = {}
        for key, entry in self.q_table.items():
            data[key] = {
                "goal_type": entry.goal_type,
                "skill_id": entry.skill_id,
                "q_value": entry.q_value,
                "total_uses": entry.total_uses,
                "success_count": entry.success_count,
            }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _load(self):
        if not self.working_dir:
            return
        path = os.path.join(self.working_dir, "skill_q_table.pkl")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            for key, d in data.items():
                self.q_table[key] = SkillValueEntry(**d)
            print(f"Loaded {len(self.q_table)} skill Q-table entries")
        except Exception as e:
            print(f"Failed to load skill Q-table: {e}")


# Keep old class name as alias for backward compat
GoalConditionedValueFunction = SkillConditionedValueFunction


# ================================ Main Framework ================================

@dataclass
class SkillRLConfig:
    """Configuration for Skill-Conditioned RL."""
    learning_rate: float = 0.3
    exploration_bonus: float = 0.2

GoalRLConfig = SkillRLConfig  # backward compat


class SkillRL:
    """
    Skill-Conditioned RL Framework.

    Flow:
    1. start_episode(goal, skill_id) — record which skill is being used
    2. step() — record experiences (no per-step update)
    3. end_episode(success) — update Q(goal_type, skill_id) with outcome
    """

    def __init__(
        self,
        llm_model: LLMCallable = None,
        embedding_func: Callable = None,
        goal_parser: GoalParser = None,
        working_dir: str = ".",
        config: SkillRLConfig = None,
    ):
        self.llm_model = llm_model
        self.embedding_func = embedding_func
        self.goal_parser = goal_parser
        self.working_dir = working_dir
        self.config = config or SkillRLConfig()

        os.makedirs(working_dir, exist_ok=True)

        self.value_function = SkillConditionedValueFunction(
            learning_rate=self.config.learning_rate,
            exploration_bonus=self.config.exploration_bonus,
            working_dir=os.path.join(working_dir, "skill_value"),
        )

        self.goal_manager = None

        self.current_episode: Optional[Episode] = None
        self.step_count = 0
        self.total_episodes = 0
        self.successful_episodes = 0

    def start_episode(
        self,
        task_id: str,
        goal: StructuredGoal,
        initial_state: str,
        skill_id: str = NO_SKILL,
    ):
        self.current_episode = Episode(
            episode_id=task_id,
            goal=goal,
            skill_id_used=skill_id,
        )
        self.step_count = 0

    def select_skill(
        self,
        goal: StructuredGoal,
        available_skill_ids: List[str],
    ) -> Tuple[Optional[str], float]:
        """Let RL pick the best skill for the current goal."""
        return self.value_function.get_best_skill(goal.verb, available_skill_ids)

    def step(
        self,
        state: str,
        action: str,
        next_state: str,
        reward: float,
        done: bool,
    ):
        if not self.current_episode:
            return

        exp = Experience(
            state=state, action=action, next_state=next_state,
            reward=reward, done=done, goal=self.current_episode.goal,
        )
        self.current_episode.add_experience(exp)
        self.step_count += 1

    def end_episode(self, success: bool) -> Dict[str, Any]:
        if not self.current_episode:
            return {}

        self.current_episode.success = success
        self.current_episode.final_reward = 1.0 if success else 0.0

        # Core: update Q(goal_type, skill_id)
        self.value_function.update(
            goal_type=self.current_episode.goal.verb,
            skill_id=self.current_episode.skill_id_used,
            success=success,
        )

        self.total_episodes += 1
        if success:
            self.successful_episodes += 1

        stats = {
            "episode_id": self.current_episode.episode_id,
            "total_steps": self.current_episode.total_steps,
            "total_reward": self.current_episode.total_reward,
            "success": success,
            "skill_used": self.current_episode.skill_id_used,
            "completion_ratio": 0.0,
        }

        self.current_episode = None
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": self.successful_episodes / max(self.total_episodes, 1),
            "value_function_stats": self.value_function.get_statistics(),
        }

    def save(self):
        self.value_function._save()
        stats_path = os.path.join(self.working_dir, "skill_rl_stats.json")
        os.makedirs(self.working_dir, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({
                "total_episodes": self.total_episodes,
                "successful_episodes": self.successful_episodes,
            }, f)

    def load(self):
        self.value_function._load()
        stats_path = os.path.join(self.working_dir, "skill_rl_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.total_episodes = stats.get("total_episodes", 0)
                self.successful_episodes = stats.get("successful_episodes", 0)


GoalRLFramework = SkillRL  # backward compat
