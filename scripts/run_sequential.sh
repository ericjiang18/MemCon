#!/usr/bin/env bash
#
# Run experiments sequentially — one at a time.
# No resource contention, full 72 AWS accounts per experiment.
#
# Usage:
#   ./scripts/run_sequential.sh                           # run ALL (11 methods × 3 fw × 6 bench)
#   ./scripts/run_sequential.sh lobster                   # one framework, all methods
#   ./scripts/run_sequential.sh lobster oagent            # one framework, one method
#   ./scripts/run_sequential.sh lobster oagent alfworld   # one specific experiment
#
#   MODEL=haiku-3 ./scripts/run_sequential.sh             # different model
#   API=openai OPENAI_API_KEY=sk-... ./scripts/run_sequential.sh lobster  # use OpenAI
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

MODEL="${MODEL:-sonnet-4}"
API="${API:-bedrock}"
LITELLM_PORT="${LITELLM_PORT:-4001}"

if [[ "$API" == "openai" ]]; then
    export OPENAI_API_BASE="https://api.openai.com/v1"
else
    export OPENAI_API_BASE="http://localhost:${LITELLM_PORT}/v1"
    export OPENAI_API_KEY="sk-placeholder"
fi

export LLM_MODEL="$MODEL"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/agent_baseline:$PROJECT_DIR/tasks:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"

RESULT_DIR="results/exp_${MODEL//\//_}"
LOGDIR="$RESULT_DIR/logs"
mkdir -p "$LOGDIR"

ALL_FRAMEWORKS=(lobster langgraph agent_framework)
ALL_MEMORIES=(metagpt voyager generative chatdev memorybank oagent experiencebank) # empty g-memory memcon latentmem 
ALL_BENCHMARKS=(triviaqa webwalkerqa gaia pddl sciworld alfworld)

# Parse args
background=false
FW_FILTER=""
MEM_FILTER=""
BENCH_FILTER=""
args_for_relaunch=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--background) background=true; shift ;;
        --framework)  FW_FILTER="$2"; args_for_relaunch+=(--framework "$2"); shift 2 ;;
        --memory)     MEM_FILTER="$2"; args_for_relaunch+=(--memory "$2"); shift 2 ;;
        --benchmark)  BENCH_FILTER="$2"; args_for_relaunch+=(--benchmark "$2"); shift 2 ;;
        *)
            # Positional args: framework [method [benchmark]]
            args_for_relaunch+=("$1")
            if [[ -z "$FW_FILTER" ]]; then FW_FILTER="$1"
            elif [[ -z "$MEM_FILTER" ]]; then MEM_FILTER="$1"
            elif [[ -z "$BENCH_FILTER" ]]; then BENCH_FILTER="$1"
            fi
            shift ;;
    esac
done

# If -b, relaunch in background
if $background; then
    bg_log="$PROJECT_DIR/run_sequential_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> Launching in background. Log: $bg_log"
    MODEL="$MODEL" API="$API" nohup "$0" "${args_for_relaunch[@]}" > "$bg_log" 2>&1 &
    echo ">>> PID: $!"
    echo ">>> Monitor: tail -f $bg_log"
    exit 0
fi

if [[ -n "$FW_FILTER" ]]; then
    FRAMEWORKS=("$FW_FILTER")
else
    FRAMEWORKS=("${ALL_FRAMEWORKS[@]}")
fi

if [[ -n "$MEM_FILTER" ]]; then
    MEMORIES=("$MEM_FILTER")
else
    MEMORIES=("${ALL_MEMORIES[@]}")
fi

if [[ -n "$BENCH_FILTER" ]]; then
    BENCHMARKS=("$BENCH_FILTER")
else
    BENCHMARKS=("${ALL_BENCHMARKS[@]}")
fi

# Count total and skip completed
total=0
skip=0
run_list=()
for fw in "${FRAMEWORKS[@]}"; do
    for mem in "${MEMORIES[@]}"; do
        for bench in "${BENCHMARKS[@]}"; do
            json="$RESULT_DIR/${fw}_${mem}_${bench}.json"
            if [[ -f "$json" ]]; then
                # Check if JSON is valid (non-zero tokens)
                if python3 -c "
import json,sys
with open('$json') as f: d=json.load(f)
sys.exit(0 if d.get('tokens',{}).get('total',d.get('avg_tokens_per_task',0))==0 else 1)
" 2>/dev/null; then
                    rm -f "$json"
                    run_list+=("${fw}|${mem}|${bench}")
                    total=$((total + 1))
                else
                    skip=$((skip + 1))
                fi
            else
                run_list+=("${fw}|${mem}|${bench}")
                total=$((total + 1))
            fi
        done
    done
done

echo "=============================================="
echo "  Sequential Experiment Runner"
echo "  Model:      $MODEL"
echo "  API:        $API ($OPENAI_API_BASE)"
echo "  Frameworks: ${FRAMEWORKS[*]}"
echo "  Memories:   ${MEMORIES[*]}"
echo "  Benchmarks: ${BENCHMARKS[*]}"
echo "  To run:     $total"
echo "  Skipped:    $skip (already completed)"
echo "  Results:    $RESULT_DIR/"
echo "=============================================="

if [[ $total -eq 0 ]]; then
    echo ">>> Nothing to run — all completed!"
    exit 0
fi

# Run sequentially
done_count=0
failed=()
for combo in "${run_list[@]}"; do
    IFS='|' read -r fw mem bench <<< "$combo"
    done_count=$((done_count + 1))
    log="$LOGDIR/${fw}_${mem}_${bench}.log"

    echo ""
    echo ">>> [$done_count/$total] $fw + $mem + $bench"
    echo "    Log: $log"

    python3 hmf/alfworld_runners/run_full_experiment.py \
        --benchmark "$bench" --framework "$fw" --memory "$mem" --model "$MODEL" \
        > "$log" 2>&1

    # Check result
    json="$RESULT_DIR/${fw}_${mem}_${bench}.json"
    if [[ -f "$json" ]]; then
        acc=$(python3 -c "import json; d=json.load(open('$json')); print(f'{d.get(\"success_rate\",d.get(\"accuracy\",0))*100:.1f}%')" 2>/dev/null)
        echo "    Done: $acc"
    else
        echo "    FAILED — no JSON produced"
        failed+=("$fw + $mem + $bench")
    fi
done

echo ""
echo "=============================================="
echo "  Finished: $done_count/$total"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "  Failed: ${#failed[@]}"
    for f in "${failed[@]}"; do
        echo "    - $f"
    done
else
    echo "  All succeeded!"
fi
echo "=============================================="
