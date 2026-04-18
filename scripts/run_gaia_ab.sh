#!/bin/bash
set -e
source /local3/ericjiang/miniconda3/etc/profile.d/conda.sh
conda activate hmf
cd /local3/ericjiang/agent_layered_memory
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'")
export OPENAI_API_BASE=$(grep OPENAI_API_BASE .env | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'")

LOGDIR="results/full_experiment_gpt-4.1-mini/logs"
mkdir -p "$LOGDIR"

echo "===== GAIA/AssistantBench (fresh per-benchmark memory DBs) ====="
echo "Time: $(date)"

for BENCH in gaia assistantbench; do
  for fw in lobster langgraph agent_framework; do
    for mem in memcon g-memory; do
      logf="$LOGDIR/${fw}_${mem}_${BENCH}.log"
      if [ -f "$logf" ] && grep -q "on ${BENCH}:" "$logf" 2>/dev/null; then
        echo "[SKIP] $fw + $mem + $BENCH — already done"
        continue
      fi
      echo ""
      echo ">>> $fw + $mem + $BENCH  [$(date)]"
      timeout 1800 python hmf/alfworld_runners/run_full_experiment.py \
        --benchmark "$BENCH" --framework "$fw" --memory "$mem" 2>&1 | tee "$logf" | tail -5
      echo "<<< done $fw + $mem + $BENCH  [$(date)]"
      sleep 3
    done
  done
done

echo ""
echo "===== ALL DONE ====="
echo "Time: $(date)"
