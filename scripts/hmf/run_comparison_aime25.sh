#!/usr/bin/env bash
#
# Three-way comparison on AIME 2025:
#   1. langgraph          (no memory baseline)
#   2. langgraph_gmemory  (G-Memory)
#   3. langgraph_hmf      (HMF — ours)
#
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/local3/ericjiang/miniconda3/bin:$PATH"
source activate hmf

source .env 2>/dev/null || true
export BASE_URL="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
export PYTHONPATH="$(pwd):$(pwd)/agent_baseline:${PYTHONPATH:-}"

LIMIT="${1:-10}"
OUTPUT="results/comparison"
mkdir -p "$OUTPUT"

echo "=============================================="
echo "  LangGraph Comparison: AIME 2025"
echo "  Model : $LLM_MODEL"
echo "  Limit : $LIMIT samples"
echo "=============================================="

for FW in langgraph langgraph_gmemory langgraph_hmf; do
    echo ""
    echo ">>> Running $FW ..."
    cd agent_baseline
    python3 run_benchmark.py -f "$FW" -b aime_2025 --limit "$LIMIT" -o "../$OUTPUT" 2>&1 || {
        echo "  [WARN] $FW failed, continuing..."
    }
    cd ..
done

echo ""
echo "=============================================="
echo "  Comparison complete. Results in $OUTPUT/"
echo "=============================================="

python3 -u -c "
import json, os, glob
files = sorted(glob.glob('$OUTPUT/langgraph*_aime_2025.json'))
print()
print(f'{'Framework':<25} {'Accuracy':>10} {'Tokens':>10} {'Tok/sample':>12} {'Time/sample':>12}')
print('-' * 72)
for f in files:
    d = json.load(open(f))
    name = d['framework']
    acc = d['accuracy'] * 100
    tok = d['total_tokens']
    n = d['total_samples']
    avg_tok = tok / n if n else 0
    avg_t = d.get('avg_time_per_sample', 0)
    print(f'{name:<25} {acc:>9.1f}% {tok:>10,} {avg_tok:>11,.0f} {avg_t:>11.1f}s')
print()
" 2>&1 || true
