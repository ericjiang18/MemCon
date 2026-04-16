#!/usr/bin/env bash
# Run ALL benchmarks on ALL frameworks.
# Usage:
#   bash script/run_all.sh                  # full run
#   bash script/run_all.sh --limit 5        # quick sanity check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="${*}"

echo "================================================================"
echo "  Running all benchmarks on all 4 frameworks"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "================================================================"

for script in run_agent_framework.sh run_autogen.sh run_langgraph.sh run_lobster.sh; do
    echo ""
    echo ">>> Starting $script …"
    bash "$SCRIPT_DIR/$script" ${EXTRA_ARGS}
done

echo ""
echo "================================================================"
echo "  All runs complete.  Generating summary …"
echo "================================================================"
echo ""
python3 "$(dirname "$SCRIPT_DIR")/summarize_results.py" "$(dirname "$SCRIPT_DIR")/results"
