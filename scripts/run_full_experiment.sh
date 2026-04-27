#!/usr/bin/env bash
#
# Full experiment suite via LiteLLM proxy (multi-AWS Bedrock).
# 3 frameworks x 5 memories x N benchmarks
#
# Prerequisites:
#   1. source /home/ubuntu/workplace/0SysExp/source_all_bedrock_accounts.sh
#   2. Start LiteLLM proxy:
#      cd /home/ubuntu/workplace/AI-Scientist-v2
#      nohup .venv/bin/litellm --config /home/ubuntu/workplace/MemCon/litellm_config.yaml \
#            --port 4000 --num_workers 16 > /home/ubuntu/workplace/MemCon/litellm_proxy.log 2>&1 &
#
# Or simply: ./scripts/run.sh setup
#
set -euo pipefail
cd "$(dirname "$0")/.."

# ── LiteLLM proxy configuration ─────────────────────────────────────────────
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:4000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-placeholder}"
export LLM_MODEL="${LLM_MODEL:-gpt-4.1-mini}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
export PYTHONPATH="$(pwd):$(pwd)/agent_baseline:$(pwd)/tasks:${PYTHONPATH:-}"

OUTDIR="results/exp_${LLM_MODEL//\//_}_$(date +%Y%m%d)"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

FRAMEWORKS=(lobster langgraph agent_framework)
MEMORIES=(memcon g-memory latentmem empty)

# Interactive benchmarks
INTERACTIVE=(alfworld pddl sciworld)
# QA benchmarks
QA_BENCHMARKS=(humaneval aime_2025 beyond_aime hmmt_feb_2025 hle)

echo "=============================================="
echo "  MemCon Full Experiment Suite"
echo "  Model:      $LLM_MODEL"
echo "  API Base:   $OPENAI_API_BASE"
echo "  Frameworks: ${FRAMEWORKS[*]}"
echo "  Memories:   ${MEMORIES[*]}"
echo "  Benchmarks: ${INTERACTIVE[*]} ${QA_BENCHMARKS[*]}"
echo "  Output:     $OUTDIR"
echo "=============================================="

# Verify proxy is running
if ! curl -s "http://localhost:4000/health" >/dev/null 2>&1; then
    echo "[ERROR] LiteLLM proxy not running on port 4000."
    echo "Run: ./scripts/run.sh setup"
    exit 1
fi
echo ">>> LiteLLM proxy: OK"

# ── Launch QA benchmarks (all in parallel) ──
echo ""
echo ">>> Launching QA benchmarks..."
for BENCH in "${QA_BENCHMARKS[@]}"; do
    for FW in "${FRAMEWORKS[@]}"; do
        for MEM in "${MEMORIES[@]}"; do
            LOG="$LOGDIR/${FW}_${MEM}_${BENCH}.log"
            echo "  $FW + $MEM + $BENCH → $LOG"
            nohup python3 hmf/alfworld_runners/run_full_experiment.py \
                --benchmark "$BENCH" --framework "$FW" --memory "$MEM" \
                > "$LOG" 2>&1 &
        done
    done
done

# ── Launch interactive benchmarks (parallel per framework x memory) ──
echo ""
echo ">>> Launching interactive benchmarks..."
for BENCH in "${INTERACTIVE[@]}"; do
    for FW in "${FRAMEWORKS[@]}"; do
        for MEM in "${MEMORIES[@]}"; do
            LOG="$LOGDIR/${FW}_${MEM}_${BENCH}.log"
            echo "  $FW + $MEM + $BENCH → $LOG"
            nohup python3 hmf/alfworld_runners/run_full_experiment.py \
                --benchmark "$BENCH" --framework "$FW" --memory "$MEM" \
                > "$LOG" 2>&1 &
        done
    done
done

TOTAL=$(jobs -p | wc -l)
echo ""
echo ">>> $TOTAL experiments launched!"
echo ">>> Monitor: tail -f $LOGDIR/*.log"
echo ">>> Errors:  grep -l 'ERROR\|Traceback' $LOGDIR/*.log"
echo ">>> Status:  ./scripts/run.sh status"
