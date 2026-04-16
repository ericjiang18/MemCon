#!/usr/bin/env bash
# Run Humanity's Last Exam (first 250 MCQ questions) on all 4 frameworks
# using GPT-OSS:120B via Groq.
#
# Usage:
#   bash script/run_hle_gpt_oss.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results_hle_gpt_oss"

mkdir -p "$RESULTS_DIR"

export BASE_URL="https://api.groq.com/openai/v1"
export API_KEY="${GROQ_API_KEY:?Set GROQ_API_KEY}"
export LLM_MODEL="openai/gpt-oss-120b"

LIMIT=250

echo "================================================================"
echo "  HLE Benchmark — GPT-OSS:120B"
echo "  Model     : $LLM_MODEL"
echo "  Samples   : first $LIMIT MCQ"
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
