#!/usr/bin/env bash
#
# Launch each baseline as a separate background job with its own log.
# Results all go to results/exp_{model}/
#
# Usage:
#   ./scripts/run_all_baselines.sh                    # run all 11 methods
#   ./scripts/run_all_baselines.sh metagpt voyager    # run specific ones
#   MODEL=haiku-3 ./scripts/run_all_baselines.sh      # different model
#
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-sonnet-4}"  # Change this for different models (e.g., haiku-3)
BENCHMARKS="alfworld,pddl,sciworld,triviaqa,webwalkerqa,gaia"
FRAMEWORKS="lobster,langgraph,agent_framework"
MAX_PARALLEL="${MAX_PARALLEL:-72}"

ALL_METHODS=(empty g-memory latentmem memcon metagpt voyager generative chatdev memorybank oagent experiencebank)

# If args given, use those; otherwise run all
if [[ $# -gt 0 ]]; then
    METHODS=("$@")
else
    METHODS=("${ALL_METHODS[@]}")
fi

echo "=============================================="
echo "  Launching baselines individually"
echo "  Model:      $MODEL"
echo "  Methods:    ${METHODS[*]}"
echo "  Benchmarks: $BENCHMARKS"
echo "  Frameworks: $FRAMEWORKS"
echo "=============================================="
echo ""

for mem in "${METHODS[@]}"; do
    echo ">>> Launching: $mem"
    ./scripts/run.sh run \
        --benchmark "$BENCHMARKS" \
        --framework "$FRAMEWORKS" \
        --memory "$mem" \
        --model "$MODEL" \
        --max-parallel "$MAX_PARALLEL" \
        --exp-name "$mem" \
        -b
    echo ""
done

echo "=============================================="
echo "  All ${#METHODS[@]} baselines launched!"
echo "  Logs: run_<baseline>.log"
echo "  Monitor: tail -f run_metagpt.log"
echo "  Check all: tail -1 run_*.log"
echo "=============================================="
