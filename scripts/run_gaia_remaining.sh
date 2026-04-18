#!/bin/bash
source /local3/ericjiang/miniconda3/etc/profile.d/conda.sh
conda activate hmf
cd /local3/ericjiang/agent_layered_memory
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'")
export OPENAI_API_BASE=$(grep OPENAI_API_BASE .env | head -1 | cut -d= -f2 | tr -d '"' | tr -d "'")

LOGDIR="results/full_experiment_gpt-4.1-mini/logs"
mkdir -p "$LOGDIR"

echo "===== Restarting incomplete GAIA memory runs ====="
echo "Time: $(date)"

for fw in lobster langgraph agent_framework; do
  for mem in memcon g-memory; do
    logf="$LOGDIR/${fw}_${mem}_gaia.log"
    echo ""
    echo ">>> ${fw} + ${mem} + gaia  [$(date)]"
    timeout 1800 python hmf/alfworld_runners/run_full_experiment.py \
      --benchmark gaia --framework "${fw}" --memory "${mem}" 2>&1 | tee "${logf}" | tail -5
    rc=$?
    echo "<<< done ${fw} + ${mem} + gaia exit=${rc}  [$(date)]"
    sleep 3
  done
done

echo ""
echo "===== ALL GAIA RUNS DONE ====="
echo "Time: $(date)"
