import json
import os
import random
import re
import math
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

from .base_env import BaseEnv, BaseRecorder

@dataclass
class MathEnvConfig:
    data_dir: str
    split: str = "train"  # 'train' or 'test'
    max_steps: int = 10
    
class MathEnv(BaseEnv):
    """
    Environment for Math problems.
    Loads problems from the `data/math_{split}` directory.
    """
    def __init__(self, env_config: Dict[str, Any], max_trials: int = 10):
        self.config = env_config
        self.max_trials = max_trials
        
        # Parse nested config if present
        if 'env' in env_config:
            env_props = env_config['env']
            self.data_dir = env_props.get("data_dir", "data/math_train")
            self.split = env_props.get("split", "train")
        else:
            self.data_dir = env_config.get("data_dir", "data/math_train")
            self.split = env_config.get("split", "train")
        
        self.problems: List[Dict[str, str]] = []
        self._load_data()
        
        self.current_problem: Optional[Dict[str, str]] = None
        self.steps_taken = 0
        self.done = False
        
    def _load_data(self):
        """Load data from a specific file or all .jsonl files in a directory."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data path {self.data_dir} not found.")
            
        print(f"Loading math data from {self.data_dir}...")
        
        if os.path.isfile(self.data_dir) and self.data_dir.endswith(".jsonl"):
            with open(self.data_dir, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.problems.append(json.loads(line))
        else:
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".jsonl"):
                    path = os.path.join(self.data_dir, filename)
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                self.problems.append(json.loads(line))
        print(f"Loaded {len(self.problems)} problems.")
    
    def set_env(self, configs: dict) -> Tuple[str, str]:
        """Set a specific problem for the environment."""
        if 'env_kwargs' in configs and 'problem' in configs['env_kwargs']:
            self.current_problem = configs['env_kwargs']['problem']
        else:
            self.current_problem = random.choice(self.problems)
            
        self.steps_taken = 0
        self.done = False
        self.total_reward = 0.0

        choices = self.current_problem.get('choices', [])
        if choices:
            option_str = "\n".join(
                f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices)
            )
            task_main = (
                f"Answer the following question by selecting the correct option.\n\n"
                f"Question: {self.current_problem['problem']}\n\n"
                f"Options:\n{option_str}"
            )
        else:
            task_main = f"Solve the following math problem:\n{self.current_problem['problem']}"

        task_description = f"Source: {self.current_problem.get('source', 'unknown')}"
        return task_main, task_description

    def reset(self) -> str:
        """Reset the environment."""
        self.done = False
        self.steps_taken = 0
        self.total_reward = 0.0
        
        if self.current_problem is None:
             self.current_problem = random.choice(self.problems)
        
        # Return the problem text
        return f"Solve the following math problem:\n{self.current_problem['problem']}"

    def step(self, action: str) -> Tuple[str, float, bool]:
        """
        Execute a step.
        The action is the agent's reasoning or answer.
        """
        self.steps_taken += 1

        if action.startswith('think:'):
            self.total_reward -= 0.5
            return 'Feedback registered. Please continue with the amended plan.', -0.5, False
        
        # Check for max steps
        if self.steps_taken >= self.max_trials:
            self.done = True
            return "Max steps reached.", -1.0, True

        # Extract answer
        predicted_answer = self._extract_answer(action)
        
        if predicted_answer:
            is_correct = self._check_answer(predicted_answer, self.current_problem['answer'])
            
            if is_correct:
                reward = 1.0
                self.total_reward += reward
                self.done = True
                feedback = "Correct! Well done."
                return feedback, reward, True
            else:
                # INCORRECT ANSWER logic:
                # 4-Agent Architecture: If it submits a final incorrect answer, terminate with -1.0 penalty.
                reward = -1.0
                self.total_reward += reward
                self.done = True
                feedback = "Incorrect final answer submitted. Episode terminated."
                return feedback, reward, True

        # Intermediate step without think (like a partial formulation without a conclusion)
        # Should be rare given the prompts, but we can just treat it as a small step
        return "Continue.", 0.0, False
            
    @staticmethod
    def process_action(action: str) -> str:
        """Process the action string from the agent."""
        return action.strip()

    def feedback(self) -> Tuple[float, bool, str]:
        """Return final feedback."""
        return self.total_reward, self.done, ""

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from text. Handles both MC letters and open-ended math."""
        is_mc = bool(self.current_problem and self.current_problem.get('choices'))

        if is_mc:
            match = re.search(r'(?:answer is|Answer:?)\s*\(?([A-J])\)?', text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match = re.search(r'\b([A-J])\b(?!.*\b[A-J]\b)', text)
            if match:
                return match.group(1).upper()
            return None

        match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
        if match:
            res = match.group(1).strip()
            if res.count('{') > res.count('}'):
                greedy_match = re.search(r'\\boxed\{(.*)\}', text, re.DOTALL)
                if greedy_match:
                    greedy_res = greedy_match.group(1).strip()
                    while greedy_res and greedy_res.count('{') < greedy_res.count('}'):
                        greedy_res = greedy_res[:-1]
                    return greedy_res.strip()
            return res

        match = re.search(r'(?:answer is|The answer is|Answer:?)\s*([^\n.]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None
        
    def _check_answer(self, prediction: str, ground_truth: str) -> bool:
        """Robustly check if prediction matches ground truth."""
        is_mc = bool(self.current_problem and self.current_problem.get('choices'))
        if is_mc:
            return prediction.strip().upper() == str(ground_truth).strip().upper()

        def normalize(s):
            s = str(s).lower().strip()
            s = s.replace(' ', '').replace('$', '').replace(',', '')
            return s
            
        norm_pred = normalize(prediction)
        norm_gt = normalize(ground_truth)
        
        if norm_pred == norm_gt:
            return True
            
        try:
            if abs(float(norm_pred) - float(norm_gt)) < 1e-6:
                return True
        except ValueError:
            pass
            
        try:
            import sympy
            expr_pred = sympy.sympify(norm_pred)
            expr_gt = sympy.sympify(norm_gt)
            if sympy.simplify(expr_pred - expr_gt) == 0:
                return True
        except Exception:
            pass
            
        return False


@dataclass
class MathRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'math'
        self.total_correct = 0
        self.total_attempted = 0
        
    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        self.log(f"--- Math Task: {task_id} ---")
        
    def task_end(self, reward: float, done: bool):
        self.total_attempted += 1
        if reward > 0:
            self.total_correct += 1
        
        acc = self.total_correct / self.total_attempted
        self.log(f"Done: {done}, Reward: {reward}, Accuracy: {acc:.2f} ({self.total_correct}/{self.total_attempted})")
