#!/usr/bin/env python3
"""
Hyperparameter sweep for MemCon policy.

Runs random search over MemCon's policy parameters on a small subset
(ALFWorld + Lobster, 30 tasks) and saves the best configuration.

Usage:
    python scripts/sweep_hyperparams.py
    python scripts/sweep_hyperparams.py --n-samples 10 --max-tasks 20
    python scripts/sweep_hyperparams.py --benchmark pddl --framework langgraph
"""

import argparse
import itertools
import json
import os
import random
import shutil
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tasks"))
sys.path.insert(0, os.path.join(_REPO, "agent_baseline"))
os.chdir(_REPO)

from dotenv import load_dotenv
load_dotenv()

# ── Search space ──

SEARCH_SPACE = {
    "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3],
    "ucb_c": [0.5, 1.0, 1.4, 2.0],
    "discount": [0.8, 0.9, 0.95],
    "warm_start": [True, False],
}


def sample_configs(n: int) -> list[dict]:
    """Random sample from the grid."""
    all_combos = list(itertools.product(*SEARCH_SPACE.values()))
    keys = list(SEARCH_SPACE.keys())
    if n >= len(all_combos):
        samples = all_combos
    else:
        samples = random.sample(all_combos, n)
    return [dict(zip(keys, combo)) for combo in samples]


def run_one_config(config: dict, benchmark: str, framework: str,
                   max_tasks: int, model: str) -> dict:
    """Run one experiment with the given policy config and return summary."""
    from hmf.alfworld_runners.run_full_experiment import get_runner, get_memory

    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    # Create a unique working directory for this sweep trial
    trial_id = f"sweep_{hash(json.dumps(config, sort_keys=True)) % 100000:05d}"
    working_dir = os.path.join(".db", "_sweep", trial_id)

    # Build memory with custom policy config
    from mas.module_map import module_map
    from mas.llm import GPTChat
    from mas.utils import EmbeddingFunc

    _, mem_cls = module_map("io", "memcon")
    llm = GPTChat(model_name=model)
    embed = EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2")

    mem_backend = mem_cls(
        namespace="memcon",
        global_config={
            "working_dir": working_dir,
            "hop": 1,
            "policy_config": config,
        },
        llm_model=llm,
        embedding_func=embed,
    )

    # Run the benchmark
    runner = get_runner(framework)
    out_path = os.path.join(working_dir, f"result_{benchmark}.json")

    try:
        summary = runner.run_alfworld(
            memory_backend=mem_backend,
            output_path=out_path,
            max_tasks=max_tasks,
        )
    except Exception as e:
        print(f"  [ERROR] config={config}: {e}")
        summary = {"success_rate": 0.0, "avg_tokens_per_task": 999999}

    # Clean up sweep trial DB to save disk
    try:
        shutil.rmtree(working_dir, ignore_errors=True)
    except Exception:
        pass

    return summary


def main():
    parser = argparse.ArgumentParser(description="MemCon hyperparameter sweep")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of random configs to try (default: 20)")
    parser.add_argument("--max-tasks", type=int, default=30,
                        help="Max tasks per trial (default: 30)")
    parser.add_argument("--benchmark", default="alfworld",
                        help="Benchmark to sweep on (default: alfworld)")
    parser.add_argument("--framework", default="lobster",
                        help="Framework to use (default: lobster)")
    parser.add_argument("--model", default=None,
                        help="LLM model (default: from LLM_MODEL env)")
    parser.add_argument("--output", default="configs/best_memcon_policy.json",
                        help="Output path for best config")
    args = parser.parse_args()

    model = args.model or os.environ.get("LLM_MODEL", "gpt-4.1-mini")

    configs = sample_configs(args.n_samples)
    print(f"=== MemCon Hyperparameter Sweep ===")
    print(f"  Benchmark:  {args.benchmark}")
    print(f"  Framework:  {args.framework}")
    print(f"  Model:      {model}")
    print(f"  Max tasks:  {args.max_tasks}")
    print(f"  Configs:    {len(configs)}")
    print()

    results = []
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Testing: {config}")
        t0 = time.time()

        summary = run_one_config(
            config, args.benchmark, args.framework,
            args.max_tasks, model
        )

        sr = summary.get("success_rate", 0.0)
        tok = summary.get("avg_tokens_per_task", 0)
        elapsed = time.time() - t0

        results.append({
            "config": config,
            "success_rate": sr,
            "avg_tokens_per_task": tok,
            "time_sec": round(elapsed, 1),
        })
        print(f"  → success_rate={sr:.3f}, avg_tok={tok}, time={elapsed:.0f}s")
        print()

    # Sort by success rate (primary), then by token efficiency (secondary)
    results.sort(key=lambda r: (-r["success_rate"], r["avg_tokens_per_task"]))

    print("=" * 60)
    print("RESULTS (sorted by success rate):")
    print("=" * 60)
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1} SR={r['success_rate']:.3f} Tok={r['avg_tokens_per_task']} "
              f"Config={r['config']}")

    best = results[0]
    print(f"\n  BEST: {best['config']}")
    print(f"        SR={best['success_rate']:.3f}, Tok={best['avg_tokens_per_task']}")

    # Save best config
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "policy_config": best["config"],
            "sweep_results": results,
            "sweep_params": {
                "benchmark": args.benchmark,
                "framework": args.framework,
                "model": model,
                "max_tasks": args.max_tasks,
                "n_samples": len(configs),
            },
        }, f, indent=2)
    print(f"\n  Saved to: {args.output}")

    # Also save all results
    sweep_log = args.output.replace(".json", "_full.json")
    with open(sweep_log, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results: {sweep_log}")


if __name__ == "__main__":
    main()
