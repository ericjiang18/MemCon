#!/usr/bin/env python3
"""
Unified benchmark runner for agent frameworks.

Usage
-----
    python run_benchmark.py -f autogen -b humaneval
    python run_benchmark.py -f langgraph -b aime_2025 --limit 10
    python run_benchmark.py -f agent-framework -b mmlu --limit 50 -o results/
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_ALIASES = {
    "humaneval": "humaneval",
    "livecodebench": "livecodebench",
    "mmlu": "mmlu_pro",
    "mmlu_pro": "mmlu_pro",
    "aime_2025": "aime_2025",
    "aime_2026": "aime_2026",
    "beyond_aime": "beyond_aime",
    "hmmt_feb_2025": "hmmt_2025",
    "hmmt_2025": "hmmt_2025",
    "hle": "hle",
}

DOMAIN_MAP = {
    "humaneval": "humaneval",
    "livecodebench": "livecodebench",
    "mmlu_pro": "mmlu_pro",
    "aime_2025": "aime",
    "aime_2026": "aime",
    "beyond_aime": "aime",
    "hmmt_2025": "hmmt",
    "hle": "hle",
}

FRAMEWORKS = [
    "agent-framework", "autogen", "langgraph", "lobster",
    "langgraph_hmf", "agent_framework_hmf", "lobster_hmf",
    "langgraph_gmemory",
]


def load_env():
    """Load .env and return config dict."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

    base_url = os.environ.get("BASE_URL")
    api_key = os.environ.get("API_KEY")
    model = os.environ.get("LLM_MODEL")

    missing = []
    if not base_url:
        missing.append("BASE_URL")
    if not api_key:
        missing.append("API_KEY")
    if not model:
        missing.append("LLM_MODEL")
    if missing:
        sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}  (check .env)")

    return {"base_url": base_url, "api_key": api_key, "model": model}


def resolve_benchmark(name: str) -> str:
    key = BENCHMARK_ALIASES.get(name)
    if key is None:
        avail = ", ".join(sorted(BENCHMARK_ALIASES.keys()))
        sys.exit(f"Unknown benchmark '{name}'.  Available: {avail}")
    return key


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks across agent frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Benchmarks: humaneval  livecodebench  mmlu  aime_2025  aime_2026\n"
            "            beyond_aime  hmmt_feb_2025\n\n"
            "Frameworks: agent-framework  autogen  langgraph  lobster"
        ),
    )
    parser.add_argument("-f", "--framework", required=True, choices=FRAMEWORKS)
    parser.add_argument("-b", "--benchmark", required=True)
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Max samples (useful for quick tests)")
    parser.add_argument("-o", "--output-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    config = load_env()

    bench_key = resolve_benchmark(args.benchmark)
    domain = DOMAIN_MAP[bench_key]

    print(f"[config]  model     = {config['model']}")
    print(f"[config]  base_url  = {config['base_url']}")
    print(f"[config]  framework = {args.framework}")
    print(f"[config]  benchmark = {args.benchmark} -> {bench_key}")
    if args.limit:
        print(f"[config]  limit     = {args.limit}")
    print()

    # ── dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset '{bench_key}' …")
    from dataset.datasets import get_dataset
    ds = get_dataset(bench_key, limit=args.limit)
    n = len(ds)
    if n == 0:
        sys.exit("ERROR: Dataset loaded 0 samples.")
    print(f"  {n} samples loaded.\n")

    # ── prompt set ───────────────────────────────────────────────────────
    from dataset.prompt import PromptSetRegistry
    prompt_set = PromptSetRegistry[domain]

    # ── runner ───────────────────────────────────────────────────────────
    print(f"Initialising {args.framework} runner …")
    from runners import get_runner
    runner = get_runner(args.framework, **config)

    # ── run ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  START  {args.framework}  ×  {bench_key}  ({n} samples)")
    print(f"{'=' * 60}\n")

    summary = asyncio.run(
        runner.run_benchmark(
            dataset=ds,
            prompt_set=prompt_set,
            output_dir=args.output_dir,
            benchmark_name=bench_key,
        )
    )
    return summary


if __name__ == "__main__":
    main()
