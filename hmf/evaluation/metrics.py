"""
HMF Evaluation Metrics — extends standard accuracy with memory-specific measures.

Metrics tracked:
  1. Task success rate      (standard)
  2. Token efficiency       (tokens per successful task)
  3. Latency               (wall-clock time per task and per step)
  4. Memory growth          (entries over time in each substrate)
  5. Cache hit rate         (fraction of queries served by cache)
  6. Skill reuse rate       (fraction of tasks using a stored skill)
  7. MPC action distribution (what actions the controller chooses)
  8. Robustness score       (performance under noisy/adversarial memory)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskMetric:
    task_id: str
    success: bool
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    steps: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retrievals: int = 0
    skill_invocations: int = 0
    skill_consolidations: int = 0
    evictions: int = 0
    mpc_actions: Dict[str, int] = field(default_factory=dict)
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)


class HMFMetrics:
    """Aggregates per-task metrics into summary statistics."""

    def __init__(self):
        self.tasks: List[TaskMetric] = []
        self._memory_growth: List[Dict[str, int]] = []
        self._wall_start: Optional[float] = None

    def start_run(self):
        self._wall_start = time.time()

    def record_task(self, metric: TaskMetric):
        self.tasks.append(metric)
        if metric.memory_snapshot:
            self._memory_growth.append(metric.memory_snapshot)

    def summary(self) -> Dict[str, Any]:
        n = len(self.tasks)
        if n == 0:
            return {"error": "no tasks recorded"}

        successes = sum(1 for t in self.tasks if t.success)
        total_tokens = sum(t.total_tokens for t in self.tasks)
        success_tokens = sum(t.total_tokens for t in self.tasks if t.success) or 1
        total_latency = sum(t.latency_sec for t in self.tasks)
        total_cache_hits = sum(t.cache_hits for t in self.tasks)
        total_cache_misses = sum(t.cache_misses for t in self.tasks)
        total_retrievals = sum(t.retrievals for t in self.tasks)
        total_skill_invocations = sum(t.skill_invocations for t in self.tasks)

        mpc_dist: Dict[str, int] = {}
        for t in self.tasks:
            for action, count in t.mpc_actions.items():
                mpc_dist[action] = mpc_dist.get(action, 0) + count

        wall_total = time.time() - self._wall_start if self._wall_start else total_latency

        return {
            "total_tasks": n,
            "successes": successes,
            "success_rate": successes / n,
            "total_tokens": total_tokens,
            "avg_tokens_per_task": total_tokens / n,
            "tokens_per_success": success_tokens / max(successes, 1),
            "total_latency_sec": round(total_latency, 2),
            "avg_latency_per_task": round(total_latency / n, 2),
            "wall_time_sec": round(wall_total, 2),
            "cache_hit_rate": total_cache_hits / max(total_cache_hits + total_cache_misses, 1),
            "total_retrievals": total_retrievals,
            "skill_reuse_rate": total_skill_invocations / n,
            "mpc_action_distribution": mpc_dist,
            "memory_growth_snapshots": len(self._memory_growth),
        }

    def robustness_summary(
        self,
        clean_metrics: "HMFMetrics",
        noisy_metrics: "HMFMetrics",
    ) -> Dict[str, Any]:
        """Compare clean vs. noisy run to measure robustness."""
        clean = clean_metrics.summary()
        noisy = noisy_metrics.summary()

        sr_drop = clean["success_rate"] - noisy["success_rate"]
        token_ratio = noisy["avg_tokens_per_task"] / max(clean["avg_tokens_per_task"], 1)
        latency_ratio = noisy["avg_latency_per_task"] / max(clean["avg_latency_per_task"], 0.01)

        robustness_score = max(0.0, 1.0 - sr_drop) * (1.0 / max(token_ratio, 0.5))

        return {
            "clean_success_rate": clean["success_rate"],
            "noisy_success_rate": noisy["success_rate"],
            "success_rate_drop": sr_drop,
            "token_overhead_ratio": token_ratio,
            "latency_overhead_ratio": latency_ratio,
            "robustness_score": round(robustness_score, 4),
        }

    def save(self, output_path: str):
        data = {
            "summary": self.summary(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "success": t.success,
                    "total_tokens": t.total_tokens,
                    "latency_sec": t.latency_sec,
                    "steps": t.steps,
                    "cache_hits": t.cache_hits,
                    "skill_invocations": t.skill_invocations,
                    "mpc_actions": t.mpc_actions,
                }
                for t in self.tasks
            ],
            "memory_growth": self._memory_growth,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
