#!/usr/bin/env bash
# HMF on PDDL (gpt-4o-mini)
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/local3/ericjiang/miniconda3/bin:$PATH"
source activate hmf

source .env 2>/dev/null || true
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"

python3 tasks/run.py \
    --task pddl \
    --mas_type skill-mas \
    --mas_memory hmf \
    --reasoning io \
    --model gpt-4o-mini \
    --max_trials 30 \
    --successful_topk 1 \
    --insights_topk 3
