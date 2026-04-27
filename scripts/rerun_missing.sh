#!/usr/bin/env bash
#
# Re-run missing experiments.
#
# Three modes:
#   1. Auto-detect: scans results/ for missing JSONs and runs them
#   2. Manual list: edit the MANUAL_RUNS array below to run specific experiments
#   3. Auto-fix:   scans log dirs for failed runs (403/expired token errors) and reruns them
#
# Usage:
#   ./scripts/rerun_missing.sh                     # auto-detect missing JSONs
#   ./scripts/rerun_missing.sh -b                  # auto-detect, background
#   ./scripts/rerun_missing.sh --manual             # run MANUAL_RUNS list
#   ./scripts/rerun_missing.sh --manual -b          # manual list, background
#   ./scripts/rerun_missing.sh --auto-fix           # scan logs for errors, rerun failed
#   ./scripts/rerun_missing.sh --auto-fix -b        # auto-fix, background
#   ./scripts/rerun_missing.sh --max-parallel 12   # limit parallelism
#

# ══════════════════════════════════════════════════════════════════════════════
# MANUAL RUNS — edit this list to run specific experiments.
# Format: "framework|memory|benchmark"
# Uncomment/add lines as needed, then run with --manual flag.
# ══════════════════════════════════════════════════════════════════════════════
MANUAL_RUNS=(
    "lobster|latentmem|sciworld"
    "langgraph|latentmem|pddl"
    "agent_framework|latentmem|alfworld"
    "agent_framework|latentmem|pddl"
    "agent_framework|latentmem|sciworld"
)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# Load .env if exists (contains OPENAI_API_KEY)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

export LLM_MODEL="${LLM_MODEL:-gpt-4.1-mini}"
# If model starts with "gpt-" or "o3" or "o4", use direct OpenAI API
if [[ "${LLM_MODEL}" == gpt-* ]] || [[ "${LLM_MODEL}" == o3* ]] || [[ "${LLM_MODEL}" == o4* ]]; then
    export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
else
    export OPENAI_API_BASE="http://localhost:4001/v1"
    export OPENAI_API_KEY="sk-placeholder"
fi
export PYTHONPATH="$(pwd):$(pwd)/agent_baseline:$(pwd)/tasks:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"

# ── Parse arguments ──
max_parallel=36
background=false
use_manual=false
use_autofix=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-parallel) max_parallel="$2"; shift 2 ;;
        -b|--background) background=true; shift ;;
        --manual) use_manual=true; shift ;;
        --auto-fix) use_autofix=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If -b, re-launch in background
if $background; then
    bg_log="$PROJECT_DIR/rerun_background.log"
    local_args=(--max-parallel "$max_parallel")
    $use_manual && local_args+=(--manual)
    $use_autofix && local_args+=(--auto-fix)
    echo ">>> Launching in background. Log: $bg_log"
    nohup "$0" "${local_args[@]}" > "$bg_log" 2>&1 &
    echo ">>> PID: $!"
    echo ">>> Monitor: tail -f $bg_log"
    exit 0
fi

RESULT_DIR="results/exp_${LLM_MODEL//\//_}"
DEFAULT_LOGDIR="${RESULT_DIR}/logs"
mkdir -p "$DEFAULT_LOGDIR"

# ── Build experiment list ──
experiments=()
declare -A log_paths  # combo -> original log path (to overwrite on rerun)

if $use_autofix; then
    # Scan all log directories for failed runs (403, expired token, Traceback)
    echo ">>> Scanning logs for failed experiments..."
    for logfile in $(find "$RESULT_DIR/logs" -name "*.log" 2>/dev/null | sort -u); do
        # Check last 5 lines for error patterns
        if tail -5 "$logfile" 2>/dev/null | grep -qE 'ERROR|error code: 403|security token.*invalid|security token.*expired|Traceback'; then
            # Extract fw_mem_bench from filename
            base=$(basename "$logfile" .log)
            # Parse: framework_memory_benchmark (memory can have hyphens)
            for fw in lobster langgraph agent_framework; do
                if [[ "$base" == "${fw}_"* ]]; then
                    rest="${base#${fw}_}"
                    for bench in alfworld pddl sciworld triviaqa webwalkerqa gaia humaneval aime_2025 beyond_aime hmmt_feb_2025 hle; do
                        if [[ "$rest" == *"_${bench}" ]]; then
                            mem="${rest%_${bench}}"
                            # Skip inactive methods (g-memory-orig was dropped)
                            case "$mem" in
                                empty|g-memory|latentmem|memcon|metagpt|voyager|generative|chatdev|memorybank|oagent|experiencebank) ;;
                                *) break ;;
                            esac
                            combo="${fw}|${mem}|${bench}"
                            json="$RESULT_DIR/${fw}_${mem}_${bench}.json"
                            # Skip if JSON exists AND has valid results
                            need_rerun=false
                            if [[ ! -f "$json" ]]; then
                                need_rerun=true
                            elif python3 -c "
import json,sys
with open('$json') as f: d=json.load(f)
tok=d.get('tokens',{}).get('total',d.get('avg_tokens_per_task',0))
sys.exit(0 if tok==0 else 1)
" 2>/dev/null; then
                                echo "  [BAD JSON] $fw + $mem + $bench — removing"
                                rm -f "$json"
                                need_rerun=true
                            fi
                            if $need_rerun; then
                                experiments+=("$combo")
                                log_paths["$combo"]="$logfile"
                            fi
                            break
                        fi
                    done
                    break
                fi
            done
        fi
    done
    # Deduplicate
    if [[ ${#experiments[@]} -gt 0 ]]; then
        readarray -t experiments < <(printf '%s\n' "${experiments[@]}" | sort -u)
    fi
    if [[ ${#experiments[@]} -eq 0 ]]; then
        echo ">>> No failed experiments found (or all already have result JSONs)."
        exit 0
    fi
    echo ">>> Found ${#experiments[@]} failed experiments to re-run"
elif $use_manual; then
    # Use the manually specified list
    for entry in "${MANUAL_RUNS[@]}"; do
        [[ "$entry" =~ ^#.*$ ]] && continue  # skip commented
        [[ -z "$entry" ]] && continue
        experiments+=("$entry")
    done
    if [[ ${#experiments[@]} -eq 0 ]]; then
        echo ">>> MANUAL_RUNS is empty! Edit scripts/rerun_missing.sh and uncomment the runs you need."
        exit 1
    fi
else
    # Auto-detect missing from results
    FRAMEWORKS=(lobster langgraph agent_framework)
    MEMORIES=(empty g-memory latentmem memcon metagpt voyager generative chatdev memorybank oagent experiencebank)
    BENCHMARKS=(alfworld pddl sciworld triviaqa webwalkerqa gaia)

    for fw in "${FRAMEWORKS[@]}"; do
        for mem in "${MEMORIES[@]}"; do
            for bench in "${BENCHMARKS[@]}"; do
                json="$RESULT_DIR/${fw}_${mem}_${bench}.json"
                if [[ ! -f "$json" ]]; then
                    experiments+=("${fw}|${mem}|${bench}")
                elif python3 -c "
import json,sys
with open('$json') as f: d=json.load(f)
tok=d.get('tokens',{}).get('total',d.get('avg_tokens_per_task',0))
sys.exit(0 if tok==0 else 1)
" 2>/dev/null; then
                    echo "  [BAD JSON] $fw + $mem + $bench (0 tokens, 0 accuracy) — will rerun"
                    rm -f "$json"
                    experiments+=("${fw}|${mem}|${bench}")
                fi
            done
        done
    done

    if [[ ${#experiments[@]} -eq 0 ]]; then
        echo ">>> All experiments already completed! Nothing to re-run."
        exit 0
    fi
fi

total=${#experiments[@]}
echo "=============================================="
echo "  Re-run Experiments"
echo "  Mode:       $(if $use_autofix; then echo 'AUTO-FIX (scan logs)'; elif $use_manual; then echo 'MANUAL'; else echo 'AUTO-DETECT (missing JSONs)'; fi)"
echo "  Model:      $LLM_MODEL"
echo "  Runs:       $total"
echo "  Parallel:   $max_parallel"
echo "  Logs:       $DEFAULT_LOGDIR/"
echo "=============================================="

# ── Clean old logs for experiments we're about to rerun ──
echo ">>> Cleaning old logs..."
for combo in "${experiments[@]}"; do
    IFS='|' read -r fw mem bench <<< "$combo"
    old_log="$DEFAULT_LOGDIR/${fw}_${mem}_${bench}.log"
    [[ -f "$old_log" ]] && rm -f "$old_log"
done

# ── Split into QA (high parallelism) and Interactive (low parallelism) ──
INTERACTIVE_BENCHES="alfworld pddl sciworld"
interactive_exps=()
qa_exps=()
for combo in "${experiments[@]}"; do
    IFS='|' read -r fw mem bench <<< "$combo"
    is_interactive=false
    for ib in $INTERACTIVE_BENCHES; do
        [[ "$bench" == "$ib" ]] && is_interactive=true && break
    done
    if $is_interactive; then
        interactive_exps+=("$combo")
    else
        qa_exps+=("$combo")
    fi
done

echo ">>> QA benchmarks: ${#qa_exps[@]} (parallel: $max_parallel)"
echo ">>> Interactive benchmarks: ${#interactive_exps[@]} (parallel: max 6)"
echo ""

# ── Launch QA first (high parallelism) ──
running=0
launched=0
if [[ ${#qa_exps[@]} -gt 0 ]]; then
    echo ">>> Phase 1: QA benchmarks..."
    for combo in "${qa_exps[@]}"; do
        IFS='|' read -r fw mem bench <<< "$combo"
        while [[ $running -ge $max_parallel ]]; do
            wait -n 2>/dev/null || true
            running=$(jobs -rp | wc -l)
        done
        log="$DEFAULT_LOGDIR/${fw}_${mem}_${bench}.log"
        launched=$((launched + 1))
        echo "  [$launched/$total] $fw + $mem + $bench → $log"
        (
            python3 hmf/alfworld_runners/run_full_experiment.py \
                --benchmark "$bench" --framework "$fw" --memory "$mem" \
                > "$log" 2>&1
        ) &
        running=$((running + 1))
    done
    echo ">>> Waiting for QA benchmarks..."
    wait
    echo ">>> QA benchmarks done!"
    echo ""
fi

# ── Then Interactive (low parallelism to avoid memory issues) ──
interactive_parallel=6
if [[ $max_parallel -lt $interactive_parallel ]]; then
    interactive_parallel=$max_parallel
fi
running=0
if [[ ${#interactive_exps[@]} -gt 0 ]]; then
    echo ">>> Phase 2: Interactive benchmarks (max $interactive_parallel parallel)..."
    for combo in "${interactive_exps[@]}"; do
        IFS='|' read -r fw mem bench <<< "$combo"
        while [[ $running -ge $interactive_parallel ]]; do
            wait -n 2>/dev/null || true
            running=$(jobs -rp | wc -l)
        done
        log="$DEFAULT_LOGDIR/${fw}_${mem}_${bench}.log"
        launched=$((launched + 1))
        echo "  [$launched/$total] $fw + $mem + $bench → $log"

        (
            python3 hmf/alfworld_runners/run_full_experiment.py \
                --benchmark "$bench" --framework "$fw" --memory "$mem" \
                > "$log" 2>&1
        ) &
        running=$((running + 1))
    done
    echo ">>> Waiting for interactive benchmarks..."
    wait
    echo ">>> Interactive benchmarks done!"
fi

echo ""
echo ">>> All $total experiments launched and completed."
echo ">>> Checking for errors..."

# No monitor loop needed — we already waited above
sleep 1
done
wait

echo ""
echo ">>> All $total experiments finished!"
errors=$(grep -rl 'ERROR\|Traceback' "$RESULT_DIR"/logs/*.log results/exp_${LLM_MODEL//\//_}_*/logs/*.log 2>/dev/null | sort -u | wc -l)
if [[ "$errors" -gt 0 ]]; then
    echo ">>> WARNING: $errors logs with errors:"
    grep -rl 'ERROR\|Traceback' "$RESULT_DIR"/logs/*.log results/exp_${LLM_MODEL//\//_}_*/logs/*.log 2>/dev/null | sort -u
else
    echo ">>> All clean — no errors detected."
fi

# Auto-generate table
model_slug="${LLM_MODEL//\//_}"
echo ""
echo ">>> Generating LaTeX table..."
python3 scripts/generate_table.py results/exp_${model_slug}*/ \
    --output "$RESULT_DIR/table.tex" --model "$LLM_MODEL" 2>&1
echo ">>> Table saved to: $RESULT_DIR/table.tex"
