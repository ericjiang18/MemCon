"""Base runner with shared benchmark execution logic."""

import asyncio
import json
import os
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

NUM_ROUNDS = 2

DOMAIN_AGENT_ROLES: Dict[str, List[str]] = {
    "humaneval":     ["Algorithm Designer", "Programming Expert", "Test Analyst"],
    "livecodebench": ["Algorithm Designer", "Programming Expert", "Test Analyst"],
    "mmlu_pro":      ["Knowlegable Expert", "Mathematician", "Critic"],
    "gaia":          ["Knowlegable Expert", "Mathematician", "Critic"],
    "aime":          ["Math Solver", "Mathematical Analyst", "Inspector"],
    "hmmt":          ["Math Solver", "Mathematical Analyst", "Inspector"],
    "hle":           ["Knowlegable Expert", "Mathematician", "Critic"],
}


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BaseRunner(ABC):
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 5

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.prompt_set = None
        self.domain: str = ""

    @property
    @abstractmethod
    def framework_name(self) -> str:
        ...

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        """Return the LLM answer + token usage via the framework's agent abstraction."""
        ...

    def get_agent_roles(self) -> List[str]:
        return DOMAIN_AGENT_ROLES.get(self.domain, ["Knowlegable Expert", "Mathematician", "Critic"])

    def get_role_prompts(self) -> List[Tuple[str, str]]:
        """Return [(role_name, role_system_prompt), ...] for the current domain."""
        roles = self.get_agent_roles()
        result = []
        for role in roles:
            desc = self.prompt_set.get_description(role) if self.prompt_set else role
            result.append((role, desc))
        return result

    def get_decision_system_prompt(self) -> str:
        if not self.prompt_set:
            return "You are a decision maker."
        return (
            self.prompt_set.get_decision_role().strip()
            + "\n\n"
            + self.prompt_set.get_decision_constraint().strip()
        )

    async def _generate_with_retry(self, system_prompt: str, user_prompt: str) -> GenerateResult:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await self.generate(system_prompt, user_prompt)
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * attempt
                    print(f"    [retry {attempt}/{self.MAX_RETRIES}] {exc!r} – waiting {delay}s")
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def run_benchmark(
        self,
        dataset,
        prompt_set,
        output_dir: str,
        benchmark_name: str,
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        self.prompt_set = prompt_set

        from run_benchmark import DOMAIN_MAP
        self.domain = DOMAIN_MAP.get(benchmark_name, benchmark_name)

        system_prompt = self.get_decision_system_prompt()
        few_shot = prompt_set.get_decision_few_shot()
        if few_shot:
            system_prompt += "\n\n" + few_shot.strip()

        results = []
        correct = 0.0
        total = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        wall_start = time.time()

        samples = list(dataset)

        for idx, sample in enumerate(samples):
            t0 = time.time()
            user_prompt = prompt_set.get_answer_prompt(sample.task)

            try:
                gen = await self._generate_with_retry(system_prompt, user_prompt)
                prediction = gen.text
            except Exception as exc:
                prediction = f"ERROR: {exc}"
                gen = GenerateResult(text=prediction)
                traceback.print_exc()

            processed = prompt_set.postprocess_answer(prediction)

            try:
                score = dataset.evaluate(processed, sample.ground_truth)
            except Exception:
                score = 0.0
                traceback.print_exc()

            correct += score
            total += 1
            total_prompt_tokens += gen.prompt_tokens
            total_completion_tokens += gen.completion_tokens
            elapsed = time.time() - t0
            acc_pct = correct / total * 100

            results.append(
                {
                    "task_id": sample.task_id,
                    "prediction_raw": prediction[:2000],
                    "prediction_processed": processed[:500],
                    "ground_truth": sample.ground_truth[:500],
                    "score": score,
                    "time_sec": round(elapsed, 2),
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "total_tokens": gen.total_tokens,
                }
            )

            tok_str = ""
            if gen.total_tokens:
                tok_str = f"  tok={gen.total_tokens}"

            print(
                f"  [{idx + 1}/{len(samples)}] {sample.task_id:<30} "
                f"score={score:.1f}  acc={acc_pct:5.1f}%  {elapsed:.1f}s{tok_str}"
            )

        wall_total = time.time() - wall_start
        accuracy = correct / total if total > 0 else 0.0
        total_all_tokens = total_prompt_tokens + total_completion_tokens

        summary = {
            "framework": self.framework_name,
            "benchmark": benchmark_name,
            "model": self.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_samples": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "total_time_sec": round(wall_total, 2),
            "avg_time_per_sample": round(wall_total / total, 2) if total else 0,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_all_tokens,
            "results": results,
        }

        out_path = os.path.join(output_dir, f"{self.framework_name}_{benchmark_name}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 64}")
        print(f"  Framework : {self.framework_name}")
        print(f"  Benchmark : {benchmark_name}")
        print(f"  Model     : {self.model}")
        print(f"  Accuracy  : {accuracy * 100:.2f}%  ({correct:.0f} / {total})")
        if total:
            print(f"  Wall time : {wall_total:.1f}s   avg {wall_total / total:.1f}s/sample")
        print(f"  Tokens    : {total_all_tokens:,}  (prompt {total_prompt_tokens:,} + completion {total_completion_tokens:,})")
        if total:
            print(f"  Tok/sample: {total_all_tokens / total:,.0f} avg")
        print(f"  Saved     : {out_path}")
        print(f"{'=' * 64}\n")

        return summary
