#!/usr/bin/env bash
# Run ALL benchmarks on ALL frameworks using the Mercury-2 model.
# Results go to results_mercury/ to avoid overwriting the original runs.
#
# Usage:
#   bash script/run_all_mercury.sh                  # full run
#   bash script/run_all_mercury.sh --limit 5        # quick sanity check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXTRA_ARGS="${*}"
RESULTS_DIR="$PROJECT_DIR/results_mercury"

mkdir -p "$RESULTS_DIR"

# ── Override env vars to use Mercury-2 ───────────────────────────────
export BASE_URL="https://api.inceptionlabs.ai/v1"
export API_KEY="sk_6d95e8b8934a0e5fd29d38a71eb52f7d"
export LLM_MODEL="mercury-2"

echo "================================================================"
echo "  Mercury-2 benchmark run"
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
echo "  All Mercury-2 runs complete.  Generating summary …"
echo "================================================================"
echo ""
python3 "$PROJECT_DIR/summarize_results.py" "$RESULTS_DIR"
