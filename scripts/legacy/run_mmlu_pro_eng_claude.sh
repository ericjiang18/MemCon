#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate GMemory

cd "$(dirname "$0")/.." || exit 1

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-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}"

python3 tasks/run.py \
    --task mmlu_pro_eng \
    --reasoning io \
    --mas_memory skill-rl \
    --mas_type skill-mas \
    --model claude-haiku-4-5-20251001 \
    --max_trials 3
