#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate GMemory

cd "$(dirname "$0")/.." || exit 1

export GOOGLE_API_KEY="${GOOGLE_API_KEY:-AIzaSyBAu0RE6fFvXgllNUvU-MMhAHDLNJCpf_k}"

python3 tasks/run.py \
    --task sciworld \
    --reasoning io \
    --mas_memory skill-rl \
    --mas_type skill-mas \
    --model gemini-2.5-flash \
    --max_trials 40
