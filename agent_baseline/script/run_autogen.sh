#!/usr/bin/env bash
# Run all benchmarks on AutoGen.
# Usage:  bash script/run_autogen.sh [--limit N]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXTRA_ARGS="${*}"

BENCHMARKS=(humaneval livecodebench mmlu aime_2025 aime_2026 beyond_aime hmmt_feb_2025)

for bench in "${BENCHMARKS[@]}"; do
    echo ""
    echo "############################################################"
    echo "  autogen  ×  ${bench}"
    echo "############################################################"
    python3 "$PROJECT_DIR/run_benchmark.py" \
        --framework autogen \
        --benchmark "$bench" \
        --output-dir "$PROJECT_DIR/results" \
        ${EXTRA_ARGS} \
    || echo "  [ERROR] ${bench} failed – continuing …"
done

echo ""
echo "All autogen benchmarks finished.  Results in $PROJECT_DIR/results/"
