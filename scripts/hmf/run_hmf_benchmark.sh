#!/usr/bin/env bash
#
# Run HMF (Hierarchical Memory Framework) benchmarks.
#
# Usage:
#   bash scripts/run_hmf_benchmark.sh <framework> <benchmark> [limit]
#
# Examples:
#   bash scripts/run_hmf_benchmark.sh langgraph_hmf aime_2025
#   bash scripts/run_hmf_benchmark.sh lobster_hmf mmlu --limit 50
#   bash scripts/run_hmf_benchmark.sh agent_framework_hmf humaneval
#
# Frameworks:  langgraph_hmf | agent_framework_hmf | lobster_hmf
# Benchmarks:  humaneval | livecodebench | mmlu | aime_2025 | aime_2026 |
#              beyond_aime | hmmt_feb_2025 | hle

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# Activate conda environment
export PATH="/local3/ericjiang/miniconda3/bin:$PATH"
source activate hmf

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

FRAMEWORK="${1:?Usage: $0 <framework> <benchmark> [--limit N]}"
BENCHMARK="${2:?Usage: $0 <framework> <benchmark> [--limit N]}"
shift 2

# Map OPENAI env vars to what agent_baseline expects
export BASE_URL="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"

OUTPUT_DIR="results/hmf"
mkdir -p "$OUTPUT_DIR"

# Ensure hmf package is importable from repo root
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/agent_baseline:${PYTHONPATH:-}"

echo "================================================"
echo "  HMF Benchmark Runner"
echo "  Framework : $FRAMEWORK"
echo "  Benchmark : $BENCHMARK"
echo "  Model     : $LLM_MODEL"
echo "  Output    : $OUTPUT_DIR"
echo "================================================"

cd agent_baseline

python3 run_benchmark.py \
    -f "$FRAMEWORK" \
    -b "$BENCHMARK" \
    -o "../$OUTPUT_DIR" \
    "$@"

echo ""
echo "Results saved to $OUTPUT_DIR/"
