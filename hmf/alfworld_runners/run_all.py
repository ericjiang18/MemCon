#!/usr/bin/env python3
"""
Run ALFWorld across 3 frameworks × 2 memory backends = 6 combinations.

Usage:
    python hmf/alfworld_runners/run_all.py [--limit N] [--combo FRAMEWORK:MEMORY]

Examples:
    python hmf/alfworld_runners/run_all.py --limit 20
    python hmf/alfworld_runners/run_all.py --combo lobster:memcon
    python hmf/alfworld_runners/run_all.py --combo langgraph:g-memory
"""

import argparse
import json
import os
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))


def get_runner(framework: str):
    kwargs = dict(model=MODEL, api_key=API_KEY, api_base=API_BASE)
    if framework == "langgraph":
        from hmf.alfworld_runners.langgraph_runner import LangGraphALFWorld
        return LangGraphALFWorld(**kwargs)
    elif framework == "lobster":
        from hmf.alfworld_runners.lobster_runner import LobsterALFWorld
        return LobsterALFWorld(**kwargs)
    elif framework == "agent_framework":
        from hmf.alfworld_runners.agent_framework_runner import AgentFrameworkALFWorld
        return AgentFrameworkALFWorld(**kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def get_memory(memory_name: str, framework: str = ""):
    from mas.module_map import module_map
    from mas.llm import GPTChat
    from mas.utils import EmbeddingFunc

    if memory_name == "none":
        return None

    _, mem_cls = module_map("io", memory_name)
    llm = GPTChat(model_name=MODEL)
    embed = EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2")

    working_dir = os.path.join(".db", MODEL.replace("/", "_"), f"fw_{framework}_{memory_name}")
    os.makedirs(working_dir, exist_ok=True)

    return mem_cls(
        namespace=memory_name,
        global_config={"working_dir": working_dir, "hop": 1},
        llm_model=llm,
        embedding_func=embed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=134)
    parser.add_argument("--combo", type=str, default=None,
                        help="Run single combo: framework:memory (e.g., lobster:memcon)")
    args = parser.parse_args()

    if args.combo:
        fw, mem = args.combo.split(":")
        combos = [(fw, mem)]
    else:
        combos = [
            ("lobster", "memcon"),
            ("lobster", "g-memory"),
            ("langgraph", "memcon"),
            ("langgraph", "g-memory"),
            ("agent_framework", "memcon"),
            ("agent_framework", "g-memory"),
        ]

    out_dir = "results/alfworld_frameworks"
    os.makedirs(out_dir, exist_ok=True)
    all_summaries = []

    for framework, memory in combos:
        print(f"\n{'='*60}")
        print(f"  {framework} + {memory}  ({MODEL}, {args.limit} tasks)")
        print(f"{'='*60}\n")

        try:
            runner = get_runner(framework)
            mem_backend = get_memory(memory, framework)
            out_path = os.path.join(out_dir, f"{framework}_{memory}.json")

            summary = runner.run_alfworld(
                memory_backend=mem_backend,
                output_path=out_path,
                max_tasks=args.limit,
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"  [ERROR] {framework}+{memory}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final comparison table
    if all_summaries:
        print(f"\n{'='*70}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Framework':<18} {'Memory':<15} {'Success':>8} {'Tokens':>10} {'Tok/Task':>10}")
        print("-" * 70)
        for s in sorted(all_summaries, key=lambda x: x["success_rate"], reverse=True):
            tok = s.get("tokens", s.get("total_tokens", {}))
            total_tok = tok.get("total_tokens", 0) if isinstance(tok, dict) else 0
            print(f"{s['framework']:<18} {s['memory']:<15} "
                  f"{s['success_rate']:>7.1%} "
                  f"{total_tok:>10,} "
                  f"{s['avg_tokens_per_task']:>10,}")
        print(f"{'='*70}")

        with open(os.path.join(out_dir, "comparison.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
