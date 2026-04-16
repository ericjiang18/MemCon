#!/usr/bin/env bash
# Run Humanity's Last Exam (first 250 questions) on all 4 frameworks
# using Gemini 3.1 Flash Lite.
#
# Usage:
#   bash script/run_hle_gemini_pro.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results_hle_gemini_pro"

mkdir -p "$RESULTS_DIR"

export BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
export API_KEY="${GEMINI_API_KEY:-AIzaSyBKclGf3tRvHEC9z2PueILdoqx4eOeAo1Y}"
export LLM_MODEL="gemini-3.1-flash-lite-preview"

LIMIT=250

echo "================================================================"
echo "  HLE Benchmark — Gemini 3.1 Flash Lite"
echo "  Model     : $LLM_MODEL"
echo "  Samples   : first $LIMIT"
echo "  Results   : $RESULTS_DIR"
echo "================================================================"

FRAMEWORKS=(agent-framework autogen langgraph lobster)

for fw in "${FRAMEWORKS[@]}"; do
    echo ""
    echo "############################################################"
    echo "  ${fw}  ×  HLE (${LIMIT} samples)"
    echo "############################################################"
    python3 "$PROJECT_DIR/run_benchmark.py" \
        --framework "$fw" \
        --benchmark hle \
        --limit "$LIMIT" \
        --output-dir "$RESULTS_DIR" \
    || echo "  [ERROR] ${fw} × HLE failed – continuing …"
done

echo ""
echo "================================================================"
echo "  All HLE runs complete.  Generating summary …"
echo "================================================================"
echo ""
python3 "$PROJECT_DIR/summarize_results.py" "$RESULTS_DIR"
