"""
G-Memory++ Main Module
Integrates enhancement modules:
- Goal Module (goal parsing and matching)
- Prompt Evolution (adaptive prompt improvement)
- Skill Miner (experience collection)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .memory_base import MASMemoryBase
from .GMemory import GMemory, TaskLayer, InsightsManager
from ..common import MASMessage, AgentMessage, StateChain

from .goal_module import GoalParser, GoalMatcher, StructuredGoal
# from .prompt_evolution import PromptEvolutionManager, PromptEvolutionConfig, PromptVariant
from .skill_miner import SkillMiner, Skill

from mas.llm import LLMCallable, Message
from mas.utils import EmbeddingFunc


@dataclass
class GMemoryPlusConfig:
    """Configuration for G-Memory++"""
    
    enable_goal_module: bool = True
    enable_skill_miner: bool = True
    
    use_llm_for_goal_parsing: bool = False
    
    min_cluster_size: int = 3
    skill_similarity_threshold: float = 0.65


@dataclass
class GMemoryPlus(GMemory):
    """
    G-Memory++ extends G-Memory with:
    1. Goal Module - parses tasks into structured goals for better matching
    2. Prompt Evolution - evolves agent prompts based on success/failure (DISABLED)
    3. Skill Miner - extracts reusable skills from successful trajectories
    """
    
    config: GMemoryPlusConfig = field(default_factory=GMemoryPlusConfig)
    
    def __post_init__(self):
        super().__post_init__()
        
        if not hasattr(self, 'config') or self.config is None:
            self.config = GMemoryPlusConfig()
        
        self._init_modules()
        self._task_counter = 0
        self._current_task_id: Optional[str] = None
        self._current_goal: Optional[StructuredGoal] = None
        
        print(f"G-Memory++ initialized with config: {self._get_config_summary()}")
    
    def _init_modules(self):
        if self.config.enable_goal_module:
            self.goal_parser = GoalParser(
                llm_model=self.llm_model if self.config.use_llm_for_goal_parsing else None
            )
            self.goal_matcher = GoalMatcher(self.goal_parser)
        else:
            self.goal_parser = None
            self.goal_matcher = None
        
        # if self.config.enable_prompt_evolution:
        #     self.prompt_evolution = PromptEvolutionManager(
        #         llm_model=self.llm_model,
        #         working_dir=self.persist_dir,
        #         config=self.config.prompt_evolution_config
        #     )
        # else:
        self.prompt_evolution = None
        
        if self.config.enable_skill_miner:
            self.skill_miner = SkillMiner(
                llm_model=self.llm_model,
                embedding_func=self.embedding_func,
                working_dir=self.persist_dir,
                min_cluster_size=self.config.min_cluster_size,
                similarity_threshold=self.config.skill_similarity_threshold
            )
        else:
            self.skill_miner = None
    
    def _get_config_summary(self) -> str:
        enabled = []
        if self.config.enable_goal_module:
            enabled.append("Goal")
        # if self.config.enable_prompt_evolution:
        #     enabled.append("PromptEvo")
        if self.config.enable_skill_miner:
            enabled.append("Skills")
        return f"[{', '.join(enabled)}]"
    
    # ================================ Task Context ================================
    
    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
        screenshot: Any = None,
        domain: str = None
    ) -> MASMessage:
        mas_message = super().init_task_context(task_main, task_description)
        
        self._current_task_id = f"task_{self._task_counter}"
        self._task_counter += 1
        
        if self.goal_parser:
            self._current_goal = self.goal_parser.parse(
                task_main=task_main,
                task_description=task_description or "",
                use_llm=self.config.use_llm_for_goal_parsing
            )
            mas_message.add_extra_field('structured_goal', self._current_goal.to_dict())
        
        return mas_message
    
    # ================================ Retrieval ================================
    
    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 2,
        failed_topk: int = 1,
        insight_topk: int = 10,
        skill_topk: int = 3,
        threshold: float = 0.3,
        screenshot: Any = None,
        **kwargs
    ) -> Tuple[List[MASMessage], List[MASMessage], List[str], List[Skill]]:
        query_goal = None
        if self.goal_parser:
            query_goal = self.goal_parser.parse(query_task, "")
        
        successful_trajs, failed_trajs, insights = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk * 2,
            failed_topk=failed_topk * 2,
            insight_topk=insight_topk,
            threshold=threshold
        )
        
        if query_goal and successful_trajs:
            successful_trajs = self._filter_by_goal_similarity(
                query_goal, successful_trajs, successful_topk
            )
        
        skills = []
        if self.skill_miner and query_goal:
            skill_results = self.skill_miner.retrieve_skills(query_goal, top_k=skill_topk)
            skills = [skill for skill, score in skill_results]
        
        return successful_trajs[:successful_topk], failed_trajs[:failed_topk], insights, skills
    
    def _filter_by_goal_similarity(
        self,
        query_goal: StructuredGoal,
        trajectories: List[MASMessage],
        top_k: int
    ) -> List[MASMessage]:
        scored = []
        for traj in trajectories:
            goal_dict = traj.get_extra_field('structured_goal')
            if goal_dict:
                traj_goal = StructuredGoal.from_dict(goal_dict)
                sim = self.goal_parser.compute_similarity(query_goal, traj_goal)
            else:
                sim = 0.5 if query_goal.verb in traj.task_main.lower() else 0.3
            scored.append((traj, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [traj for traj, score in scored[:top_k]]
    
    # ================================ Prompt Management (DISABLED) ================================
    
    # def get_evolved_prompt(
    #     self,
    #     role: str,
    #     default_prompt: str,
    #     domain: str = "general",
    #     insights: List[str] = None
    # ) -> str:
    #     if not self.prompt_evolution:
    #         return default_prompt
    #     
    #     self.prompt_evolution.register_default_prompt(role, default_prompt, domain)
    #     variant = self.prompt_evolution.select_prompt(role, domain)
    #     prompt = variant.content
    #     
    #     if insights:
    #         prompt = self.prompt_evolution.inject_insights(prompt, insights)
    #     
    #     return prompt
    
    # def update_prompt_stats(
    #     self,
    #     role: str,
    #     variant: "PromptVariant",
    #     success: bool,
    #     tokens_used: int = 0,
    #     failure_reason: str = None
    # ):
    #     if self.prompt_evolution:
    #         self.prompt_evolution.update_stats(
    #             role=role,
    #             variant=variant,
    #             success=success,
    #             tokens_used=tokens_used,
    #             failure_reason=failure_reason
    #         )
    
    # ================================ Memory Management ================================
    
    def add_memory(self, mas_message: MASMessage) -> None:
        mas_message = self.refine_memory(mas_message)
        
        if mas_message.get_extra_field('deduplicated'):
            return
        
        super().add_memory(mas_message)
        
        if self.skill_miner and mas_message.label == True:
            goal = None
            goal_dict = mas_message.get_extra_field('structured_goal')
            if goal_dict:
                goal = StructuredGoal.from_dict(goal_dict)
            else:
                goal = self.goal_parser.parse(mas_message.task_main, "") if self.goal_parser else None
            
            if goal:
                key_steps = mas_message.get_extra_field('key_steps') or []
                if isinstance(key_steps, str):
                    key_steps = [s.strip() for s in key_steps.split('\n') if s.strip()]
                
                self.skill_miner.add_trajectory(
                    task_id=mas_message.task_main,
                    goal=goal,
                    trajectory=mas_message.task_trajectory,
                    key_steps=key_steps,
                    success=True
                )
    
    # ================================ Skill Access ================================
    
    def get_relevant_skills(self, top_k: int = 3) -> List[Tuple[Skill, float]]:
        if not self.skill_miner or not self._current_goal:
            return []
        return self.skill_miner.retrieve_skills(self._current_goal, top_k=top_k)
    
    def format_skills_for_prompt(self, skills: List[Skill], max_skills: int = 2) -> str:
        if not skills:
            return ""
        
        active_skills = [s for s in skills if s.active][:max_skills]
        if not active_skills:
            return ""
        
        parts = ["## Relevant Skills from Past Experience"]
        parts.append("These are proven procedures for similar tasks. Follow them unless the situation requires deviation.\n")
        for skill in active_skills:
            parts.append(f"### {skill.name} (success rate: {skill.success_rate:.0%})")
            parts.append(f"When to use: {skill.description}")
            if skill.goal_pattern:
                parts.append(f"Applies to: {skill.goal_pattern}")
            if skill.preconditions:
                parts.append(f"Preconditions: {', '.join(skill.preconditions)}")
            parts.append("Steps:")
            for i, step in enumerate(skill.steps[:8], 1):
                parts.append(f"  {i}. {step}")
            if skill.postconditions:
                parts.append(f"Expected result: {', '.join(skill.postconditions)}")
            parts.append("")
        
        return "\n".join(parts)
    
    # ================================ Memory Refinement (Evo-Memory) ================================
    
    def refine_memory(self, mas_message: MASMessage) -> MASMessage:
        """
        Refine memory after task completion (Evo-Memory Refine step).
        
        1. Trajectory compression — keep only key decision points
        2. Experience deduplication — merge similar trajectories (>0.9 sim)
        3. Low-value experience deprioritization
        
        Called automatically from add_memory.
        """
        # 1. Trajectory compression
        mas_message = self._compress_trajectory(mas_message)
        
        # 2. Deduplication check
        self._deduplicate_experience(mas_message)
        
        return mas_message
    
    def _compress_trajectory(self, mas_message: MASMessage) -> MASMessage:
        """
        Compress trajectory to keep only key actions (skip thinks and no-ops).
        For successful trajectories, the key_steps already provide a compressed view.
        For failed ones, keep last 10 steps to understand failure mode.
        """
        if not mas_message.task_trajectory:
            return mas_message
        
        lines = mas_message.task_trajectory.split('\n')
        compressed = []
        step_count = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Skip think actions from trajectory storage
            if stripped.lower().startswith(('think:', 'thought:', '>think', '>thought')):
                continue
            
            compressed.append(line)
            if stripped.startswith('>'):
                step_count += 1
        
        # For failed trajectories, keep last N steps
        if mas_message.label == False and step_count > 10:
            action_indices = [i for i, l in enumerate(compressed) if l.strip().startswith('>')]
            if len(action_indices) > 10:
                cut_idx = action_indices[-10]
                compressed = ['(... earlier steps omitted ...)'] + compressed[cut_idx:]
        
        mas_message.task_trajectory = '\n'.join(compressed)
        return mas_message
    
    def _deduplicate_experience(self, mas_message: MASMessage):
        """
        Check if a highly similar experience already exists.
        If so, update the existing one's metadata instead of adding a new entry.
        Uses embedding similarity > 0.9 threshold.
        """
        try:
            query_embedding = self.embedding_func.embed_query(mas_message.task_main)
            existing_docs = self.main_memory.similarity_search_with_score(
                query=mas_message.task_main,
                k=3,
                filter={'label': mas_message.label}
            )
            
            for doc, distance in existing_docs:
                similarity = 1 - distance
                if similarity > 0.9:
                    existing_traj = doc.metadata.get('task_trajectory', '')
                    if len(existing_traj) > len(mas_message.task_trajectory):
                        mas_message.add_extra_field('deduplicated', True)
                        return
        except Exception:
            pass
    
    # ================================ Statistics ================================
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "memory_size": self.memory_size,
            "task_counter": self._task_counter,
            "modules_enabled": self._get_config_summary(),
        }
        
        if self.skill_miner:
            stats["num_skills"] = len(self.skill_miner.skills)
            stats["active_skills"] = len(self.skill_miner.get_active_skills())
        
        return stats
