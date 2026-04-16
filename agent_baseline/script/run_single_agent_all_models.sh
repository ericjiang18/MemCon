#!/usr/bin/env bash
# ============================================================
#  Single-agent evaluation across 5 frontier models × 8 benchmarks
#
#  Models:
#    1. Gemini 3.1 Pro
#    2. Claude Opus 4
#    3. Claude Sonnet 4
#    4. Kimi K2 Instruct (Groq)
#    5. GPT 4.1 (OpenAI)
#
#  Usage:
#    bash script/run_single_agent_all_models.sh
#    bash script/run_single_agent_all_models.sh gemini       # one model
#    bash script/run_single_agent_all_models.sh claude_opus gpt  # specific models
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SELECTED="${@:-gemini claude_opus claude_sonnet kimi gpt}"

BENCHMARKS=(humaneval livecodebench mmlu aime_2025 aime_2026 beyond_aime hmmt_feb_2025 hle)

run_model() {
    local NAME="$1"
    local URL="$2"
    local KEY="$3"
    local MODEL="$4"
    local OUTDIR="$PROJECT_DIR/results_single_${NAME}"

    mkdir -p "$OUTDIR"

    echo ""
    echo "################################################################"
    echo "  Model: $NAME ($MODEL)"
    echo "  URL:   $URL"
    echo "  Output: $OUTDIR"
    echo "################################################################"

    export BASE_URL="$URL"
    export API_KEY="$KEY"
    export LLM_MODEL="$MODEL"

    for bench in "${BENCHMARKS[@]}"; do
        echo ""
        echo "  >>> ${NAME} × ${bench}"
        python3 "$PROJECT_DIR/run_benchmark.py" \
            --framework lobster \
            --benchmark "$bench" \
            --output-dir "$OUTDIR" \
        || echo "  [ERROR] ${NAME} × ${bench} failed – continuing"
    done

    echo ""
    echo "  Generating summary for $NAME ..."
    python3 "$PROJECT_DIR/summarize_results.py" "$OUTDIR"
}

# ── 1. Gemini 3.1 Pro ───────────────────────────────────────
if [[ "$SELECTED" == *"gemini"* ]]; then
    GEMINI_KEY="${GEMINI_API_KEY:?Set GEMINI_API_KEY}"
    run_model "gemini_pro" \
        "https://generativelanguage.googleapis.com/v1beta/openai" \
        "$GEMINI_KEY" \
        "gemini-3.1-pro-preview"
fi

# ── 2. Claude Opus 4 ────────────────────────────────────────
if [[ "$SELECTED" == *"claude_opus"* ]]; then
    ANTHROPIC_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
    run_model "claude_opus" \
        "https://api.anthropic.com/v1/" \
        "$ANTHROPIC_KEY" \
        "claude-opus-4-6"
fi

# ── 3. Claude Sonnet 4 ──────────────────────────────────────
if [[ "$SELECTED" == *"claude_sonnet"* ]]; then
    ANTHROPIC_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
    run_model "claude_sonnet" \
        "https://api.anthropic.com/v1/" \
        "$ANTHROPIC_KEY" \
        "claude-sonnet-4-6"
fi

# ── 4. Kimi K2 Instruct (Groq) ──────────────────────────────
if [[ "$SELECTED" == *"kimi"* ]]; then
    GROQ_KEY="${GROQ_API_KEY:?Set GROQ_API_KEY}"
    run_model "kimi_k2" \
        "https://api.groq.com/openai/v1" \
        "$GROQ_KEY" \
        "moonshotai/kimi-k2-instruct"
fi

# ── 5. GPT 4.1 (OpenAI) ─────────────────────────────────────
if [[ "$SELECTED" == *"gpt"* ]]; then
    OPENAI_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
    run_model "gpt" \
        "https://api.openai.com/v1" \
        "$OPENAI_KEY" \
        "gpt-5.4-2026-03-05"
fi

echo ""
echo "================================================================"
echo "  ALL SINGLE-AGENT RUNS COMPLETE"
echo "================================================================"
