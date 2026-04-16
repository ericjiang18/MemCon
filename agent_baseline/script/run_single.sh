#!/usr/bin/env bash
# Run a SINGLE benchmark on a SINGLE framework.
# Usage:  bash script/run_single.sh <framework> <benchmark> [--limit N]
#
# Examples:
#   bash script/run_single.sh autogen humaneval
#   bash script/run_single.sh langgraph aime_2025 --limit 10
#   bash script/run_single.sh lobster mmlu --limit 50
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <framework> <benchmark> [extra-args...]"
    echo ""
    echo "Frameworks: agent-framework  autogen  langgraph  lobster"
    echo "Benchmarks: humaneval  livecodebench  mmlu  aime_2025  aime_2026  beyond_aime  hmmt_feb_2025"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRAMEWORK="$1"
BENCHMARK="$2"
shift 2

python3 "$PROJECT_DIR/run_benchmark.py" \
    --framework "$FRAMEWORK" \
    --benchmark "$BENCHMARK" \
    --output-dir "$PROJECT_DIR/results" \
    "$@"
