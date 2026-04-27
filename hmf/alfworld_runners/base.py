"""
Generic ALFWorld execution loop — mirrors SkillMAS's Think-Act loop exactly,
but uses pluggable framework backends for the LLM call.

Uses the REAL ALFWorld prompts (system prompt + task-type few-shots) from
tasks/prompts/alfworld_prompt.py, ensuring identical conditions to SkillMAS.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TASKS_DIR = os.path.join(_REPO_ROOT, "tasks")
for _p in (_REPO_ROOT, _TASKS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_STEPS = 30
WINDOW_SIZE = 7
MAX_CONSECUTIVE_THINKS = 2


def _extract_action(raw: str) -> str:
    """Same extraction logic as SkillMAS."""
    if not raw or not raw.strip():
        return "look"
    if re.search(r"\\boxed\{|Answer:\s*[A-J]|the answer is", raw, re.IGNORECASE):
        return raw.strip()

    def _clean(line):
        line = line.strip()
        line = re.sub(r"^[>\-•*`]+\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        return line.strip()

    lines = raw.strip().split("\n")
    m = re.search(r"(?:Action|OUTPUT|COMMAND)\s*:\s*(.+)", raw, re.IGNORECASE)
    if m:
        c = _clean(m.group(1).split("\n")[0])
        if c:
            return c
    for line in lines:
        c = _clean(line)
        if c and c.upper() not in ("OK.", "OK", ""):
            return c
    return "look"


@dataclass
class TokenTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0

    def add(self, pt: int, ct: int):
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_calls += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
            "llm_calls": self.total_calls,
        }


@dataclass
class TaskResult:
    task_id: int
    goal: str
    goal_type: str
    success: bool
    steps: int
    tokens: Dict[str, int] = field(default_factory=dict)
    time_sec: float = 0.0


class ALFWorldRunner(ABC):

    def __init__(self, model: str, api_key: str, api_base: str):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.token_tracker = TokenTracker()

    @property
    @abstractmethod
    def framework_name(self) -> str: ...

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        """Return (text, prompt_tokens, completion_tokens)."""
        ...

    def generate_with_retry(self, system_prompt: str, user_prompt: str,
                            max_retries: int = 5, base_delay: float = 5.0) -> Tuple[str, int, int]:
        """Wrapper around generate() with retry + exponential backoff.

        Handles transient API errors (403 expired tokens, 500 server errors,
        rate limits) so a single bad AWS account doesn't kill the whole run.
        """
        import time as _time
        last_err = None
        for attempt in range(max_retries):
            try:
                return self.generate(system_prompt, user_prompt)
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # Retry on auth/server/rate errors; raise immediately on others
                is_retryable = any(k in err_str for k in (
                    "security token", "expired", "400", "403", "404", "500", "502", "503",
                    "rate limit", "throttl", "timeout", "connection", "legacy", "invalid model",
                ))
                if not is_retryable:
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"    [RETRY {attempt+1}/{max_retries}] {type(e).__name__}: "
                      f"{str(e)[:100]}... waiting {delay:.0f}s")
                _time.sleep(delay)
        raise last_err

    def run_alfworld(
        self,
        memory_backend,
        output_path: str,
        max_tasks: int = 134,
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.chdir(_REPO_ROOT)

        from envs import get_env, get_task
        from prompts import get_dataset_system_prompt, get_task_few_shots
        from mas_workflow.format import build_system_context, build_step_prompt, format_task_context
        import yaml

        cfg_path = os.path.join(_TASKS_DIR, "env_configs", "alfworld_config.yaml")
        with open(cfg_path) as f:
            env_config = yaml.safe_load(os.path.expandvars(f.read()))

        env = get_env("alfworld", env_config, MAX_STEPS)
        tasks = get_task("alfworld")[:max_tasks]

        results: List[TaskResult] = []
        rs = [0] * 6
        cnts = [0] * 6
        _type_order = ["put", "clean", "heat", "cool", "examine", "puttwo"]

        for task_id, task_config in enumerate(tasks):
            t0 = time.time()
            task_tokens = TokenTracker()

            task_main, task_desc = env.set_env(task_config)
            task_type = task_config.get("task_type", "other")

            # --- Get real ALFWorld prompts (identical to SkillMAS) ---
            base_system = get_dataset_system_prompt("alfworld", task_config)
            few_shots = get_task_few_shots("alfworld", task_config, few_shots_num=1)

            # --- Memory retrieval ---
            memory_shots = []
            insights = []
            if memory_backend:
                memory_backend.init_task_context(task_main, task_desc)
                mem_result = memory_backend.retrieve_memory(
                    query_task=task_main, successful_topk=1,
                    failed_topk=0, insight_topk=5, threshold=0.3,
                )
                successful_trajs = mem_result[0] if len(mem_result) > 0 else []
                insights = mem_result[2] if len(mem_result) > 2 else []
                for t in successful_trajs[:1]:
                    memory_shots.append(format_task_context(
                        t.task_description, t.task_trajectory,
                        t.get_extra_field("key_steps") if hasattr(t, "get_extra_field") else None,
                    ))

            # --- Build system context (identical to SkillMAS) ---
            system_context = build_system_context(
                few_shots=few_shots,
                memory_few_shots=memory_shots,
                insights=insights[:5],
                skills_text="",
            )
            full_system = base_system + "\n" + system_context

            # --- Think-Act loop ---
            action_obs_window: List[Tuple[str, str]] = []
            recent_actions: List[str] = []
            think_buffer = ""
            observation = task_desc
            initial_obs = task_desc
            consecutive_thinks = 0
            real_steps = 0
            step = 0

            while real_steps < MAX_STEPS and step < MAX_STEPS * 2:
                # Loop warning
                loop_warning = self._detect_loop(recent_actions, consecutive_thinks)

                step_prompt = build_step_prompt(
                    task_goal=task_main,
                    action_obs_window=action_obs_window[-WINDOW_SIZE:],
                    current_observation=observation,
                    initial_observation=initial_obs,
                    loop_warning=loop_warning,
                    policy_hint="",
                    think_buffer=think_buffer,
                )

                if consecutive_thinks >= MAX_CONSECUTIVE_THINKS:
                    step_prompt += "\n\n[SYSTEM] You MUST output a physical action NOW."

                response, pt, ct = self.generate_with_retry(full_system, step_prompt)
                task_tokens.add(pt, ct)
                self.token_tracker.add(pt, ct)

                action = _extract_action(response)
                action = env.process_action(action)
                is_think = action.lower().startswith(("think", "thought"))

                if is_think:
                    consecutive_thinks += 1
                    think_buffer += f"\n- {action}"
                    if consecutive_thinks > MAX_CONSECUTIVE_THINKS:
                        real_steps += 1
                else:
                    consecutive_thinks = 0
                    think_buffer = ""
                    real_steps += 1

                obs_new, reward, done = env.step(action)
                step += 1

                if not is_think:
                    action_obs_window.append((action, obs_new))

                recent_actions.append(action)
                if len(recent_actions) > 10:
                    recent_actions.pop(0)

                # Hard loop break
                if self._should_break(recent_actions, consecutive_thinks):
                    break

                if memory_backend and not is_think:
                    memory_backend.move_memory_state(action, obs_new, reward=reward, done=done)

                observation = obs_new
                if done:
                    break

            # --- Task end ---
            final_reward, final_done, final_feedback = env.feedback()

            if memory_backend:
                memory_backend.save_task_context(label=final_done, feedback=final_feedback)
                memory_backend.backward(final_done)

            tidx = _type_order.index(task_type) if task_type in _type_order else 0
            rs[tidx] += int(final_done)
            cnts[tidx] += 1

            elapsed = time.time() - t0
            result = TaskResult(
                task_id=task_id, goal=task_main, goal_type=task_type,
                success=final_done, steps=real_steps,
                tokens=task_tokens.to_dict(), time_sec=round(elapsed, 2),
            )
            results.append(result)

            acc = sum(r.success for r in results) / len(results) * 100
            print(
                f"  [{task_id+1}/{len(tasks)}] "
                f"{'✓' if final_done else '✗'} "
                f"acc={acc:.1f}% "
                f"tok={task_tokens.total} "
                f"steps={real_steps} "
                f"{elapsed:.1f}s"
            )

        # --- Summary ---
        n = len(results)
        successes = sum(r.success for r in results)
        summary = {
            "framework": self.framework_name,
            "memory": memory_backend.__class__.__name__ if memory_backend else "none",
            "model": self.model,
            "total_tasks": n,
            "successes": successes,
            "success_rate": round(successes / max(n, 1), 4),
            "rs": rs,
            "cnts": cnts,
            "tokens": self.token_tracker.to_dict(),
            "avg_tokens_per_task": self.token_tracker.total // max(n, 1),
            "avg_time_per_task": round(sum(r.time_sec for r in results) / max(n, 1), 2),
            "tasks": [
                {"id": r.task_id, "type": r.goal_type, "success": r.success,
                 "steps": r.steps, "tokens": r.tokens, "time": r.time_sec}
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  {self.framework_name} + {summary['memory']}")
        print(f"  Success: {successes}/{n} = {summary['success_rate']:.1%}")
        print(f"  Tokens:  {self.token_tracker.total:,} total, "
              f"{summary['avg_tokens_per_task']:,}/task avg")
        print(f"  rs={rs}  cnts={cnts}")
        print(f"{'='*60}\n")

        return summary

    @staticmethod
    def _detect_loop(recent: List[str], consec_thinks: int) -> str:
        n = len(recent)
        if n >= 2 and recent[-1] == recent[-2]:
            return ("\n[WARNING] Same action repeated. Try a completely different strategy.")
        if n >= 4:
            a, b, c, d = recent[-4], recent[-3], recent[-2], recent[-1]
            if a == c and b == d and a != b:
                return ("\n[WARNING] Alternating between two failing actions. "
                        "Try a fundamentally different approach.")
        if consec_thinks >= MAX_CONSECUTIVE_THINKS:
            return "\n[WARNING] You must execute a physical action immediately."
        return ""

    @staticmethod
    def _should_break(recent: List[str], consec_thinks: int) -> bool:
        n = len(recent)
        if n >= 3 and len(set(recent[-3:])) == 1:
            return True
        if n >= 6:
            last6 = recent[-6:]
            if (last6[0] == last6[2] == last6[4] and
                    last6[1] == last6[3] == last6[5] and last6[0] != last6[1]):
                return True
        if consec_thinks >= MAX_CONSECUTIVE_THINKS + 3:
            return True
        return False
