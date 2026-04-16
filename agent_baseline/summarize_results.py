#!/usr/bin/env python3
"""
Read all JSON results from the results/ directory and print a comparison table.

Usage:  python summarize_results.py [results_dir]
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRAMEWORKS = ["agent-framework", "autogen", "langgraph", "lobster"]
BENCHMARKS = ["humaneval", "livecodebench", "mmlu_pro", "aime_2025", "aime_2026", "beyond_aime", "hmmt_2025", "hle"]
BENCH_LABELS = {
    "humaneval": "HumanEval",
    "livecodebench": "LiveCodeBench",
    "mmlu_pro": "MMLU-Pro",
    "aime_2025": "AIME 2025",
    "aime_2026": "AIME 2026",
    "beyond_aime": "BeyondAIME",
    "hmmt_2025": "HMMT 2025",
    "hle": "HLE",
}


def load_results(results_dir: str) -> dict:
    data = {}
    for path in Path(results_dir).glob("*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            key = (d["framework"], d["benchmark"])
            data[key] = d
        except Exception:
            pass
    return data


def fmt_acc(val):
    if val is None:
        return "  ---  "
    return f"{val * 100:6.2f}%"


def fmt_tok(val):
    if val is None or val == 0:
        return "    ---   "
    if val >= 1_000_000:
        return f"{val / 1_000_000:8.2f}M"
    if val >= 1_000:
        return f"{val / 1_000:8.1f}K"
    return f"{val:9d}"


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "results")
    if not os.path.isdir(results_dir):
        print(f"No results directory: {results_dir}")
        return

    data = load_results(results_dir)
    if not data:
        print("No result files found.")
        return

    model = None
    for v in data.values():
        model = v.get("model", "?")
        break

    # ── Accuracy Table ───────────────────────────────────────────────
    fw_w = 18
    b_w = 14
    sep = "+" + "-" * (b_w + 2)
    for _ in FRAMEWORKS:
        sep += "+" + "-" * (fw_w - 2)
    sep += "+"

    print()
    print(f"  Model: {model}")
    print()
    print("=" * 80)
    print("  ACCURACY  (% correct)")
    print("=" * 80)

    header = f"{'Benchmark':<{b_w}}  "
    for fw in FRAMEWORKS:
        header += f"| {fw:^{fw_w - 4}} "
    header += "|"
    print(header)
    print(sep)

    fw_totals = {fw: {"correct": 0, "total": 0} for fw in FRAMEWORKS}

    for bench in BENCHMARKS:
        label = BENCH_LABELS.get(bench, bench)
        row = f"{label:<{b_w}}  "
        for fw in FRAMEWORKS:
            d = data.get((fw, bench))
            if d:
                acc = d.get("accuracy")
                n = d.get("total_samples", 0)
                row += f"| {fmt_acc(acc)} ({n:>3}) "
                fw_totals[fw]["correct"] += d.get("correct", 0)
                fw_totals[fw]["total"] += n
            else:
                row += f"|{'---':^{fw_w - 2}} "
        row += "|"
        print(row)

    print(sep)

    avg_row = f"{'AVG':<{b_w}}  "
    for fw in FRAMEWORKS:
        t = fw_totals[fw]
        if t["total"] > 0:
            avg = t["correct"] / t["total"]
            avg_row += f"| {fmt_acc(avg)} ({t['total']:>3}) "
        else:
            avg_row += f"|{'---':^{fw_w - 2}} "
    avg_row += "|"
    print(avg_row)
    print()

    # ── Token Usage Table ────────────────────────────────────────────
    print("=" * 80)
    print("  TOKEN USAGE")
    print("=" * 80)

    header2 = f"{'Benchmark':<{b_w}}  "
    for fw in FRAMEWORKS:
        header2 += f"| {fw:^{fw_w - 4}} "
    header2 += "|"
    print(header2)
    print(sep)

    fw_tok_totals = {fw: 0 for fw in FRAMEWORKS}

    for bench in BENCHMARKS:
        label = BENCH_LABELS.get(bench, bench)
        row = f"{label:<{b_w}}  "
        for fw in FRAMEWORKS:
            d = data.get((fw, bench))
            if d:
                tok = d.get("total_tokens", 0)
                row += f"|{fmt_tok(tok)} "
                fw_tok_totals[fw] += tok
            else:
                row += f"|{'---':^{fw_w - 2}} "
        row += "|"
        print(row)

    print(sep)

    tot_row = f"{'TOTAL':<{b_w}}  "
    for fw in FRAMEWORKS:
        tot_row += f"|{fmt_tok(fw_tok_totals[fw])} "
    tot_row += "|"
    print(tot_row)
    print()

    # ── Per-Framework Summary ────────────────────────────────────────
    print("=" * 80)
    print("  PER-FRAMEWORK SUMMARY")
    print("=" * 80)
    print(f"{'Framework':<20} {'Accuracy':>10} {'Samples':>8} {'Tokens':>12} {'Tok/Sample':>11} {'Time(s)':>9}")
    print("-" * 72)

    for fw in FRAMEWORKS:
        t = fw_totals[fw]
        tok = fw_tok_totals[fw]
        total_time = sum(
            data[(fw, b)].get("total_time_sec", 0)
            for b in BENCHMARKS
            if (fw, b) in data
        )
        if t["total"] > 0:
            acc_pct = t["correct"] / t["total"] * 100
            tok_per = tok / t["total"]
            print(f"{fw:<20} {acc_pct:9.2f}% {t['total']:>8} {tok:>12,} {tok_per:>10,.0f} {total_time:>9.1f}")
        else:
            print(f"{fw:<20} {'---':>10} {'---':>8}")

    print()


if __name__ == "__main__":
    main()
