#!/usr/bin/env bash
#
# Run ALL HMF framework variants across ALL benchmarks.
# Compares baseline (langgraph, agent-framework, lobster) vs. +ours variants.
#
# Usage:
#   bash scripts/run_hmf_all.sh [--limit N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

LIMIT_ARGS=""
if [ "${1:-}" = "--limit" ] && [ -n "${2:-}" ]; then
    LIMIT_ARGS="--limit $2"
fi

FRAMEWORKS=(
    "langgraph"
    "langgraph_hmf"
    "lobster"
    "lobster_hmf"
)

BENCHMARKS=(
    "aime_2025"
    "mmlu"
    "humaneval"
)

echo "=============================================="
echo "  HMF Full Comparison Benchmark"
echo "  Frameworks: ${FRAMEWORKS[*]}"
echo "  Benchmarks: ${BENCHMARKS[*]}"
echo "=============================================="

for BENCH in "${BENCHMARKS[@]}"; do
    for FW in "${FRAMEWORKS[@]}"; do
        echo ""
        echo ">>> Running $FW x $BENCH ..."
        bash scripts/run_hmf_benchmark.sh "$FW" "$BENCH" $LIMIT_ARGS || {
            echo "  [WARN] $FW x $BENCH failed, continuing..."
        }
    done
done

echo ""
echo "All benchmarks complete. Results in results/hmf/"

# Summarize if available
if [ -f agent_baseline/summarize_results.py ]; then
    echo "Generating summary..."
    python3 agent_baseline/summarize_results.py results/hmf/ || true
fi
