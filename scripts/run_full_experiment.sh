#!/usr/bin/env bash
#
# Full experiment: 3 frameworks × 3 memories × 9 benchmarks = 81 runs
# Uses gpt-5-mini. Launches in parallel batches.
#
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/local3/ericjiang/miniconda3/bin:$PATH"
source activate hmf

source .env 2>/dev/null || true
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export LLM_MODEL="gpt-5-mini"
export ALFWORLD_DATA="${HOME}/.cache/alfworld"
export PYTHONPATH="$(pwd):$(pwd)/agent_baseline:$(pwd)/tasks:${PYTHONPATH:-}"

OUTDIR="results/full_experiment_gpt-5-mini"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

FRAMEWORKS=(lobster langgraph agent_framework)
MEMORIES=(memcon g-memory empty)

# Interactive benchmarks (sequential per framework to avoid env conflicts)
INTERACTIVE=(alfworld pddl sciworld)

# QA benchmarks (can run in parallel freely)
QA_BENCHMARKS=(humaneval aime_2025 beyond_aime hmmt_feb_2025 hle)

echo "=============================================="
echo "  Full Experiment Suite — gpt-5-mini"
echo "  Frameworks: ${FRAMEWORKS[*]}"
echo "  Memories:   ${MEMORIES[*]}"
echo "  Benchmarks: ${INTERACTIVE[*]} ${QA_BENCHMARKS[*]}"
echo "  Output:     $OUTDIR"
echo "=============================================="

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

# ── Launch interactive benchmarks (parallel per framework×memory, sequential tasks) ──
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
echo ">>> Check:   grep -l 'ERROR' $LOGDIR/*.log"
