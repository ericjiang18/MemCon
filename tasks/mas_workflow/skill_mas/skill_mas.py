from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

from mas.agents import Agent, Env
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, Message

from mas.memory.mas_memory.skill_memory import SkillMemory
from mas.memory.mas_memory.goal_module import GoalParser, StructuredGoal

from ..format import (
    format_task_context,
    build_system_context,
    build_step_prompt,
)


EXECUTOR_PROMPT = """You are a smart agent designed to solve problems. You MUST strictly follow the output format of other agents' output."""

CRITIC_PROMPT = """You are a judge. Given a task and an agent's output for that task, your job is to evaluate the agent's output and give your suggestion.
NOTE:
- If you believe the agent's answer is correct, simply output `Support`.
- If you believe the agent's answer is incorrect, provide a concise and strong suggestion."""

WINDOW_SIZE = 7
MAX_CONSECUTIVE_THINKS = 2

def _extract_action(raw_response: str) -> str:
    """
    Extract a single executable action from a potentially verbose LLM response.
    Strips common prefixes and returns the first non-trivial line.
    The environment's process_action handles validation.
    For math/QA tasks, returns the full response so the env can extract the answer.
    """
    if not raw_response or not raw_response.strip():
        return "look"

    import re

    if re.search(r'\\boxed\{|Answer:\s*[A-J]|the answer is', raw_response, re.IGNORECASE):
        return raw_response.strip()

    def _clean_line(line: str) -> str:
        line = line.strip()
        line = re.sub(r'^[>\-•*`]+\s*', '', line)
        line = re.sub(r'^\d+[.)]\s*', '', line)
        return line.strip()

    lines = raw_response.strip().split('\n')

    action_match = re.search(r'(?:Action|OUTPUT|COMMAND)\s*:\s*(.+)', raw_response, re.IGNORECASE)
    if action_match:
        candidate = _clean_line(action_match.group(1).split('\n')[0])
        if candidate:
            return candidate

    for line in lines:
        cleaned = _clean_line(line)
        if cleaned and cleaned.upper() not in ('OK.', 'OK', ''):
            return cleaned

    return "look"


class AgentWrapper:
    def __init__(self, agent_id: str, role: str, agent: Agent):
        self.agent_id = agent_id
        self.role = role
        self.agent = agent


@dataclass
class SkillMAS(MetaMAS):
    """
    Skill-conditioned MAS with Skill RL, Skill Discovery, and Evolving Experience.

    Architecture: Think-Act-Refine loop with:
    - System prompt set once (few_shots, insights, skills)
    - Sliding window step prompts (last N action/observation pairs)
    - Think isolation (think content visible only for current step)
    - Proactive loop detection
    """

    def __post_init__(self):
        self.observers = []
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=None)

        self._executors: List[AgentWrapper] = []
        self._critic: AgentWrapper = None

        self._current_goal: StructuredGoal = None
        self._goal_parser: GoalParser = None

    def build_system(
        self,
        reasoning: ReasoningBase,
        mas_memory: MASMemoryBase,
        env: Env,
        config: dict,
    ):
        num_executors = config.get('num_executors', 1)
        use_critic = config.get('use_critic', False)

        self._successful_topk = config.get('successful_topk', 2)
        self._failed_topk = config.get('failed_topk', 1)
        self._insights_topk = config.get('insights_topk', 5)
        self._threshold = config.get('threshold', 0.3)

        self.notify_observers("========================================")
        self.notify_observers("Skill MAS Configuration")
        self.notify_observers("========================================")
        self.notify_observers(f"Num Executors     : {num_executors}")
        self.notify_observers(f"Use Critic        : {use_critic}")
        self.notify_observers(f"Successful TopK   : {self._successful_topk}")
        self.notify_observers(f"Insights TopK     : {self._insights_topk}")

        self._create_agents(reasoning, num_executors, use_critic)

        self.set_env(env)
        self.meta_memory = mas_memory

        if hasattr(mas_memory, 'goal_parser'):
            self._goal_parser = mas_memory.goal_parser
        else:
            self._goal_parser = GoalParser()

        self.notify_observers("Skill MAS initialized successfully")

    def _create_agents(
        self,
        reasoning: ReasoningBase,
        num_executors: int,
        use_critic: bool,
    ):
        for i in range(num_executors):
            executor_agent = Agent(
                name=f'executor_{i}',
                role='executor',
                system_instruction=EXECUTOR_PROMPT,
                reasoning_module=reasoning,
            )
            self._executors.append(AgentWrapper(f'executor_{i}', 'executor', executor_agent))
            self.hire([executor_agent])

        if use_critic:
            critic_agent = Agent(
                name='critic',
                role='critic',
                system_instruction=CRITIC_PROMPT,
                reasoning_module=reasoning,
            )
            self._critic = AgentWrapper('critic', 'critic', critic_agent)
            self.hire([critic_agent])
    
    def schedule(self, task_config: dict) -> Tuple[float, bool]:
        task_main = task_config.get('task_main')
        task_description = task_config.get('task_description')
        few_shots = task_config.get('few_shots', [])

        if not task_main or not task_description:
            raise ValueError("task_main and task_description required")

        env = self.env

        self.meta_memory.init_task_context(task_main, task_description)
        self._current_goal = self._goal_parser.parse(task_main, task_description)

        # ---- Retrieve memory (once) ----
        retrieval_result = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
        )

        if len(retrieval_result) == 4:
            successful_trajs, _, insights, skills = retrieval_result
        else:
            successful_trajs, _, insights = retrieval_result
            skills = []

        memory_shots = [
            format_task_context(t.task_description, t.task_trajectory, t.get_extra_field('key_steps'))
            for t in successful_trajs
        ]

        # ---- Skill-Conditioned RL: let RL pick the best skill ----
        self._active_skill = None
        self._active_skill_step_idx = 0

        if skills and isinstance(self.meta_memory, SkillMemory) and hasattr(self.meta_memory, 'select_skill_for_task'):
            selected_skill, skill_id, q_val = self.meta_memory.select_skill_for_task(skills)
            if selected_skill:
                self._active_skill = selected_skill
                self.notify_observers(f"[Skill RL] Selected \"{selected_skill.name}\" (Q={q_val:.2f}, SR={selected_skill.success_rate:.0%})")
            else:
                self.notify_observers(f"[Skill RL] No skill selected — free action mode")
        elif skills:
            active_skills = [s for s in skills if s.active]
            if active_skills:
                self._active_skill = active_skills[0]

        skills_text = ""
        if skills and hasattr(self.meta_memory, 'format_skills_for_prompt'):
            display_skills = [self._active_skill] if self._active_skill else skills[:1]
            skills_text = self.meta_memory.format_skills_for_prompt(display_skills, max_skills=1)

        if skills_text:
            self.notify_observers(f"[Skills] Injecting skill into system prompt")

        # ---- Build system context ONCE ----
        system_context = build_system_context(
            few_shots=few_shots,
            memory_few_shots=memory_shots,
            insights=insights[:5] if insights else [],
            skills_text=skills_text,
        )
        # APPEND to existing instruction (which already contains task-specific
        # rules like ALFWorld syntax set by run.py), NOT overwrite.
        for executor in self._executors:
            executor.agent.total_system_instruction += '\n' + system_context

        # ---- Main execution loop (Think-Act-Refine) ----
        action_obs_window: List[Tuple[str, str]] = []
        recent_actions: List[str] = []
        think_buffer = ""
        initial_observation = task_description
        current_observation = task_description
        real_steps = 0
        consecutive_thinks = 0
        step = 0

        while real_steps < env.max_trials and step < env.max_trials * 2:
            policy_hint = self._get_skill_hint()

            loop_warning = self._detect_loop(recent_actions, consecutive_thinks)

            step_prompt = build_step_prompt(
                task_goal=task_main,
                action_obs_window=action_obs_window[-WINDOW_SIZE:],
                current_observation=current_observation,
                initial_observation=initial_observation,
                loop_warning=loop_warning,
                policy_hint=policy_hint,
                think_buffer=think_buffer,
            )

            # Force physical action after too many thinks
            if consecutive_thinks >= MAX_CONSECUTIVE_THINKS:
                step_prompt += "\n\n[SYSTEM] You have used all your think budget. You MUST output a physical action NOW."

            # Collect executor proposals
            proposed_actions = []
            for executor in self._executors:
                response = executor.agent.response(step_prompt, self.reasoning_config)
                action = _extract_action(response or "")
                proposed_actions.append((executor.agent_id, action))

            best_action = proposed_actions[0][1] if proposed_actions else "look"

            if self._critic and len(proposed_actions) >= 2:
                unique_actions = list({a for _, a in proposed_actions if a})
                if len(unique_actions) > 1:
                    best_action = self._critic_select(proposed_actions)

            if not best_action:
                best_action = "look"

            action = env.process_action(best_action)
            is_think = action.lower().startswith(('think', 'thought'))

            # ---- Think isolation ----
            if is_think:
                consecutive_thinks += 1
                # Think content only visible via think_buffer, not in sliding window
                think_buffer += f"\n- {action}"
                if consecutive_thinks > MAX_CONSECUTIVE_THINKS:
                    real_steps += 1
            else:
                consecutive_thinks = 0
                think_buffer = ""
                real_steps += 1

            # ---- Environment step ----
            observation, reward, done = env.step(action)

            step += 1
            self.notify_observers(f"Step {step}: {action}\nObs: {observation}")

            # Update sliding window (physical actions only)
            if not is_think:
                action_obs_window.append((action, observation))

            # Update loop tracking
            recent_actions.append(action)
            if len(recent_actions) > 10:
                recent_actions.pop(0)

            # ---- Loop breaker ----
            should_break = self._should_break_loop(recent_actions, consecutive_thinks)
            if should_break:
                self.notify_observers(f"[WARNING] Agent stuck in loop. Breaking out.")
                break

            # Update memory state (for persistence, not for prompt)
            self.meta_memory.move_memory_state(action, observation, reward=reward, done=done)

            current_observation = observation

            # Track skill execution progress
            if not is_think and self._active_skill:
                self._active_skill_step_idx += 1

            if done:
                break

        # ---- Refine phase (post-task) ----
        final_reward, final_done, final_feedback = env.feedback()
        self.notify_observers(final_feedback)

        # Reset executor system instructions for next task
        for executor in self._executors:
            executor.agent.total_system_instruction = EXECUTOR_PROMPT

        # Update skill stats if we used a skill
        if self._active_skill and hasattr(self.meta_memory, 'skill_miner'):
            if hasattr(self.meta_memory.skill_miner, 'update_skill_stats'):
                self.meta_memory.skill_miner.update_skill_stats(
                    self._active_skill.skill_id, final_done
                )

        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)

        return final_reward, final_done

    # ======================== Loop Detection (Phase 5) ========================

    def _detect_loop(self, recent_actions: List[str], consecutive_thinks: int) -> str:
        """Proactive loop detection — returns warning string if loop detected."""
        if not recent_actions:
            return ""

        n = len(recent_actions)

        # 1. Identical action repeated 2+ times
        if n >= 2 and recent_actions[-1] == recent_actions[-2]:
            return (
                "\n[SYSTEM WARNING] You just repeated the same action twice. "
                "You MUST try a completely different strategy. Repeating the same "
                "action will not change the environment state."
            )

        # 2. Alternating pattern (A-B-A-B)
        if n >= 4:
            a, b, c, d = recent_actions[-4], recent_actions[-3], recent_actions[-2], recent_actions[-1]
            if a == c and b == d and a != b:
                return (
                    "\n[SYSTEM WARNING] You are alternating between two actions without progress. "
                    "Both strategies have failed. Try a fundamentally different approach."
                )

        # 3. Same action type dominating recent history
        if n >= 3:
            action_types = [a.split()[0].lower() if a.split() else '' for a in recent_actions[-3:]]
            if len(set(action_types)) == 1 and action_types[0] not in ('think', 'thought'):
                return (
                    f"\n[SYSTEM HINT] Your last 3 actions all started with \"{action_types[0]}\". "
                    f"Consider trying a different action type to make progress."
                )

        # 4. Excessive thinking
        if consecutive_thinks >= MAX_CONSECUTIVE_THINKS:
            return (
                "\n[SYSTEM WARNING] You have been thinking without acting. "
                "You MUST execute a physical action immediately."
            )

        return ""

    def _should_break_loop(self, recent_actions: List[str], consecutive_thinks: int) -> bool:
        """Hard loop-break conditions — returns True to terminate the episode."""
        n = len(recent_actions)

        # 3+ identical actions in a row
        if n >= 3 and len(set(recent_actions[-3:])) == 1:
            return True

        # 2 full alternating cycles (A-B-A-B-A-B)
        if n >= 6:
            last6 = recent_actions[-6:]
            if (last6[0] == last6[2] == last6[4] and
                last6[1] == last6[3] == last6[5] and
                last6[0] != last6[1]):
                return True

        # 5+ consecutive thinks
        if consecutive_thinks >= MAX_CONSECUTIVE_THINKS + 3:
            return True

        return False

    # ======================== Skill-Guided Execution ========================

    def _get_skill_hint(self) -> str:
        """Generate skill-guided action hint if a high-confidence skill is active."""
        if not self._active_skill:
            return ""

        skill = self._active_skill
        if not hasattr(skill, 'steps') or not skill.steps:
            return ""

        sr = skill.success_rate if hasattr(skill, 'success_rate') else 0.5
        if sr < 0.4:
            return ""

        idx = self._active_skill_step_idx
        if idx >= len(skill.steps):
            return ""

        remaining = skill.steps[idx:idx + 3]
        steps_text = '\n'.join(f"  {i+1}. {s}" for i, s in enumerate(remaining))

        confidence = "high" if sr >= 0.7 else "moderate"
        return (
            f"[Skill Guidance — {confidence} confidence, success rate {sr:.0%}]\n"
            f"Skill: {skill.name}\n"
            f"Next recommended steps:\n{steps_text}\n"
            f"Follow these steps unless the situation requires deviation."
        )

    # ======================== Critic ========================

    def _critic_select(self, actions: List[Tuple[str, str]]) -> str:
        if not self._critic:
            return actions[0][1] if actions else ""

        unique_actions = list({a for _, a in actions if a})
        if len(unique_actions) <= 1:
            return unique_actions[0] if unique_actions else ""

        candidates = "\n".join(
            f"  Option {i+1}: {a}" for i, a in enumerate(unique_actions)
        )
        prompt = (
            f"You are a critic. Given the candidate actions below, pick the single "
            f"best action to execute. Reply with ONLY the chosen action text, nothing else.\n"
            f"{candidates}"
        )
        response = self._critic.agent.response(prompt, self.reasoning_config)
        chosen = response.strip() if response else ""

        for a in unique_actions:
            if a.lower() in chosen.lower() or chosen.lower() in a.lower():
                return a

        return unique_actions[0]

    # ======================== Observer ========================

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)


GoalGCNMAS = SkillMAS  # backward compat
