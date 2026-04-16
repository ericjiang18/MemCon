#!/usr/bin/env bash
# Run ALL benchmarks on ALL frameworks using the Gemini model.
# Results go to results_gemini/ to avoid overwriting the original runs.
#
# Usage:
#   bash script/run_all_gemini.sh                  # full run
#   bash script/run_all_gemini.sh --limit 5        # quick sanity check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXTRA_ARGS="${*}"
RESULTS_DIR="$PROJECT_DIR/results_gemini"

mkdir -p "$RESULTS_DIR"

# ── Override env vars to use Gemini ──────────────────────────────────
export BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"
export API_KEY="${GEMINI_API_KEY:-AIzaSyBKclGf3tRvHEC9z2PueILdoqx4eOeAo1Y}"
export LLM_MODEL="${GEMINI_LLM_MODEL:-gemini-3.1-flash-lite-preview}"

echo "================================================================"
echo "  Gemini benchmark run"
echo "  Model     : $LLM_MODEL"
echo "  Base URL  : $BASE_URL"
echo "  Results   : $RESULTS_DIR"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "================================================================"

FRAMEWORKS=(agent-framework autogen langgraph lobster)
BENCHMARKS=(humaneval livecodebench mmlu aime_2025 aime_2026 beyond_aime hmmt_feb_2025)

for fw in "${FRAMEWORKS[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
        echo ""
        echo "############################################################"
        echo "  ${fw}  ×  ${bench}"
        echo "############################################################"
        python3 "$PROJECT_DIR/run_benchmark.py" \
            --framework "$fw" \
            --benchmark "$bench" \
            --output-dir "$RESULTS_DIR" \
            ${EXTRA_ARGS} \
        || echo "  [ERROR] ${fw} × ${bench} failed – continuing …"
    done
    echo ""
    echo "All ${fw} benchmarks finished."
done

echo ""
echo "================================================================"
echo "  All Gemini runs complete.  Generating summary …"
echo "================================================================"
echo ""
python3 "$PROJECT_DIR/summarize_results.py" "$RESULTS_DIR"
