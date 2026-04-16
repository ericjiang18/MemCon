#!/usr/bin/env bash
#
# ALFWorld three-way comparison:
#   1. empty       (no memory baseline)
#   2. g-memory    (G-Memory — existing SOTA baseline)
#   3. hmf         (HMF + MPC — ours)
#
# All use: skill-mas execution loop, gpt-4o-mini, 30 max steps
#
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/local3/ericjiang/miniconda3/bin:$PATH"
source activate hmf

source .env 2>/dev/null || true
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export ALFWORLD_DATA="${HOME}/.cache/alfworld"

MODEL="${MODEL:-gpt-5.4-nano}"
TASK="alfworld"
MAS="skill-mas"
TRIALS=30

MEMORY="${1:-all}"

run_one() {
    local mem="$1"
    echo ""
    echo "======================================================"
    echo "  ALFWorld  |  memory=$mem  |  model=$MODEL"
    echo "======================================================"

    # Clean memory DB for fair comparison
    local db_dir="./.db/${MODEL}/${TASK}/${MAS}"
    rm -rf "$db_dir/$mem" 2>/dev/null || true

    python3 tasks/run.py \
        --task "$TASK" \
        --mas_type "$MAS" \
        --mas_memory "$mem" \
        --reasoning io \
        --model "$MODEL" \
        --max_trials "$TRIALS" \
        --successful_topk 1 \
        --insights_topk 3
}

if [ "$MEMORY" = "all" ]; then
    run_one "empty"
    run_one "g-memory"
    run_one "hmf"
elif [ "$MEMORY" = "empty" ] || [ "$MEMORY" = "g-memory" ] || [ "$MEMORY" = "hmf" ]; then
    run_one "$MEMORY"
else
    echo "Usage: $0 [all|empty|g-memory|hmf]"
    exit 1
fi

echo ""
echo "======================================================"
echo "  Comparison complete. Check logs in .db/"
echo "======================================================"
