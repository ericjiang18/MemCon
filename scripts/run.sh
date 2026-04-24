#!/usr/bin/env bash
# MemCon Experiment Runner
# Refreshes AWS credentials, starts LiteLLM proxy, runs experiments.
#
# Usage:
#   ./scripts/run.sh                                          # show help
#   ./scripts/run.sh setup                                    # refresh creds + start proxy only
#   ./scripts/run.sh run --benchmark alfworld --framework lobster --memory memcon
#   ./scripts/run.sh run --all                                # full experiment matrix
#   ./scripts/run.sh run --benchmark alfworld --memory memcon,g-memory,empty --framework lobster
#   ./scripts/run.sh status                                   # check proxy + running jobs

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ─── Activate virtual environment ───────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# ─── Configuration ──────────────────────────────────────────────────────────
LITELLM_PORT="${LITELLM_PORT:-4001}"
LITELLM_WORKERS="${LITELLM_WORKERS:-16}"
LITELLM_CONFIG="$PROJECT_DIR/litellm_config.yaml"
LITELLM_LOG="$PROJECT_DIR/litellm_proxy.log"
LITELLM_BIN="${LITELLM_BIN:-/home/ubuntu/workplace/AI-Scientist-v2/.venv/bin/litellm}"
CRED_SCRIPT="/home/ubuntu/workplace/0SysExp/source_all_bedrock_accounts.sh"

# Load .env if exists (contains OPENAI_API_KEY)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

export LLM_MODEL="${LLM_MODEL:-gpt-4.1-mini}"
# API setup is deferred to run_experiments() after --api flag is parsed
export AWS_DEFAULT_REGION=us-east-1
export AWS_REGION=us-east-1
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/agent_baseline:$PROJECT_DIR/tasks:${PYTHONPATH:-}"

OUTDIR="results/exp_${LLM_MODEL//\//_}_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs"

# ─── Helper functions ───────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
MemCon Experiment Runner

Commands:
  setup                         Refresh AWS credentials and start LiteLLM proxy
  run [OPTIONS]                 Run experiments
  status                        Check proxy and running experiment status
  stop                          Stop LiteLLM proxy
  logs                          Tail experiment logs

Run options:
  --benchmark BENCH[,BENCH..]   Benchmarks: alfworld, pddl, sciworld, humaneval,
                                aime_2025, beyond_aime, hmmt_feb_2025, hle,
                                triviaqa, webwalkerqa, gaia
  --framework FW[,FW..]        Frameworks: lobster, langgraph, agent_framework
  --memory MEM[,MEM..]         Memory types: memcon, g-memory, g-memory-orig,
                                latentmem, empty, skill-memory, hmf-v2
  --model MODEL                LLM model name (default: gpt-4.1-mini)
  --api MODE                   API mode: "bedrock" (default, via LiteLLM proxy) or "openai" (direct OpenAI)
  --name NAME                  Custom run name (default: exp_{model}). Results go to results/{NAME}/
  --exp-name NAME              Experiment name for background log (default: datetime). Log: run_{NAME}.log
  --all                        Run full experiment matrix
  --max-parallel N             Max parallel experiment jobs (default: 8)
  -b, --background             Run in background (nohup), safe to close terminal
  --dry-run                    Print what would run without executing

Environment:
  LLM_MODEL                    Override default model
  LITELLM_PORT                 Proxy port (default: 4000)
  LITELLM_WORKERS              Proxy workers (default: 16)

Examples:
  ./scripts/run.sh setup
  ./scripts/run.sh run --benchmark alfworld --framework lobster --memory memcon
  ./scripts/run.sh run --benchmark alfworld,pddl --framework lobster,langgraph --memory memcon,g-memory,empty
  ./scripts/run.sh run --all --model sonnet-4.6
EOF
}

refresh_credentials() {
    echo ">>> Refreshing AWS Bedrock credentials..."
    if [[ -f "$CRED_SCRIPT" ]]; then
        source "$CRED_SCRIPT"
        echo ">>> Credentials refreshed for all accounts."
    else
        echo ">>> [WARN] Credential script not found: $CRED_SCRIPT"
        echo ">>>        Set AWS credentials manually or fix the path."
    fi
}

start_proxy() {
    # Check if proxy is already running
    if curl -s "http://localhost:${LITELLM_PORT}/health" >/dev/null 2>&1; then
        echo ">>> LiteLLM proxy already running on port $LITELLM_PORT"
        return 0
    fi

    echo ">>> Starting LiteLLM proxy on port $LITELLM_PORT ($LITELLM_WORKERS workers)..."
    nohup "$LITELLM_BIN" \
        --config "$LITELLM_CONFIG" \
        --port "$LITELLM_PORT" \
        --num_workers "$LITELLM_WORKERS" \
        > "$LITELLM_LOG" 2>&1 &
    PROXY_PID=$!
    echo ">>> LiteLLM proxy PID: $PROXY_PID (log: $LITELLM_LOG)"

    # Wait for proxy to become healthy
    echo -n ">>> Waiting for proxy"
    for i in $(seq 1 30); do
        if curl -s "http://localhost:${LITELLM_PORT}/health" >/dev/null 2>&1; then
            echo " ready!"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo " TIMEOUT"
    echo ">>> [ERROR] LiteLLM proxy failed to start. Check $LITELLM_LOG"
    return 1
}

stop_proxy() {
    echo ">>> Stopping LiteLLM proxy..."
    pkill -f "litellm.*--port.*${LITELLM_PORT}" 2>/dev/null && echo ">>> Stopped." || echo ">>> No proxy found."
}

check_status() {
    echo "=== MemCon Experiment Status ==="
    echo ""

    # Proxy status
    if curl -s "http://localhost:${LITELLM_PORT}/health" >/dev/null 2>&1; then
        echo "LiteLLM Proxy: RUNNING (port $LITELLM_PORT)"
    else
        echo "LiteLLM Proxy: NOT RUNNING"
    fi
    echo ""

    # Running experiments
    RUNNING=$(pgrep -af "run_full_experiment.py" 2>/dev/null | wc -l)
    echo "Running experiments: $RUNNING"
    if [[ "$RUNNING" -gt 0 ]]; then
        pgrep -af "run_full_experiment.py" 2>/dev/null || true
    fi
    echo ""

    # Results
    if [[ -d "$OUTDIR" ]]; then
        DONE=$(find "$OUTDIR" -name "*.json" -not -path "*/logs/*" 2>/dev/null | wc -l)
        echo "Results in $OUTDIR: $DONE completed"
        if [[ -d "$LOGDIR" ]]; then
            ERRORS=$(grep -l "ERROR\|Traceback" "$LOGDIR"/*.log 2>/dev/null | wc -l)
            echo "Logs with errors: $ERRORS"
        fi
    fi
}

run_experiments() {
    local benchmarks=()
    local frameworks=()
    local memories=()
    local run_all=false
    local max_parallel=8
    local dry_run=false
    local background=false
    local model="$LLM_MODEL"
    local run_name=""
    local exp_name=""
    local api_mode="bedrock"  # default: use LiteLLM proxy for Bedrock

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --benchmark)  IFS=',' read -ra benchmarks <<< "$2"; shift 2 ;;
            --framework)  IFS=',' read -ra frameworks <<< "$2"; shift 2 ;;
            --memory)     IFS=',' read -ra memories <<< "$2"; shift 2 ;;
            --model)      model="$2"; shift 2 ;;
            --api)        api_mode="$2"; shift 2 ;;
            --name)       run_name="$2"; shift 2 ;;
            --exp-name)   exp_name="$2"; shift 2 ;;
            --all)        run_all=true; shift ;;
            --max-parallel) max_parallel="$2"; shift 2 ;;
            --dry-run)    dry_run=true; shift ;;
            -b|--background) background=true; shift ;;
            *)            echo "Unknown option: $1"; usage; exit 1 ;;
        esac
    done

    # Set API base/key based on --api flag
    if [[ "$api_mode" == "openai" ]]; then
        export OPENAI_API_BASE="https://api.openai.com/v1"
        # Key from .env or environment
    else
        export OPENAI_API_BASE="http://localhost:${LITELLM_PORT}/v1"
        export OPENAI_API_KEY="sk-placeholder"
    fi

    # Default exp_name to datetime
    if [[ -z "$exp_name" ]]; then
        exp_name="$(date +%Y%m%d_%H%M%S)"
    fi

    # If -b flag, re-launch ourselves in background with nohup
    if $background; then
        local bg_log="$PROJECT_DIR/run_${exp_name}.log"
        # Rebuild args without -b, joining arrays with commas
        local args=()
        local IFS_OLD="$IFS"
        IFS=','; args+=(--benchmark "${benchmarks[*]}" --framework "${frameworks[*]}" --memory "${memories[*]}"); IFS="$IFS_OLD"
        args+=(--model "$model" --max-parallel "$max_parallel" --exp-name "$exp_name" --api "$api_mode")
        [[ -n "$run_name" ]] && args+=(--name "$run_name")
        $run_all && args+=(--all)
        $dry_run && args+=(--dry-run)

        echo ">>> Launching in background. Log: $bg_log"
        nohup "$0" run "${args[@]}" > "$bg_log" 2>&1 &
        echo ">>> PID: $!"
        echo ">>> Monitor: tail -f $bg_log"
        echo ">>> Exp name: $exp_name"
        return 0
    fi

    export LLM_MODEL="$model"
    # Use --name if given, otherwise default model name (no timestamp)
    if [[ -n "$run_name" ]]; then
        OUTDIR="results/${run_name}"
    else
        OUTDIR="results/exp_${model//\//_}"
    fi
    LOGDIR="$OUTDIR/logs"

    if $run_all; then
        benchmarks=(alfworld pddl sciworld humaneval aime_2025 beyond_aime hmmt_feb_2025 hle)
        frameworks=(lobster langgraph agent_framework)
        memories=(memcon g-memory g-memory-orig latentmem empty)
    fi

    if [[ ${#benchmarks[@]} -eq 0 ]] || [[ ${#frameworks[@]} -eq 0 ]] || [[ ${#memories[@]} -eq 0 ]]; then
        echo "[ERROR] Must specify --benchmark, --framework, and --memory (or --all)"
        usage
        exit 1
    fi

    # Ensure proxy is running
    if ! curl -s "http://localhost:${LITELLM_PORT}/health" >/dev/null 2>&1; then
        echo ">>> LiteLLM proxy not running. Starting it..."
        refresh_credentials
        start_proxy
    fi

    mkdir -p "$LOGDIR"

    # Build experiment list
    local combos=()
    for bench in "${benchmarks[@]}"; do
        for fw in "${frameworks[@]}"; do
            for mem in "${memories[@]}"; do
                combos+=("${bench}|${fw}|${mem}")
            done
        done
    done

    local total=${#combos[@]}
    echo ""
    echo "=============================================="
    echo "  MemCon Experiment Suite"
    echo "  Model:      $model"
    echo "  Proxy:      http://localhost:${LITELLM_PORT}/v1"
    echo "  Frameworks: ${frameworks[*]}"
    echo "  Memories:   ${memories[*]}"
    echo "  Benchmarks: ${benchmarks[*]}"
    echo "  Total runs: $total"
    echo "  Parallel:   $max_parallel"
    echo "  Output:     $OUTDIR"
    echo "=============================================="

    if $dry_run; then
        echo ""
        echo ">>> DRY RUN — would launch:"
        for combo in "${combos[@]}"; do
            IFS='|' read -r bench fw mem <<< "$combo"
            echo "  python3 hmf/alfworld_runners/run_full_experiment.py --benchmark $bench --framework $fw --memory $mem"
        done
        return 0
    fi

    # Launch experiments with parallelism control
    local running=0
    local launched=0
    for combo in "${combos[@]}"; do
        IFS='|' read -r bench fw mem <<< "$combo"

        # Wait if at max parallel
        while [[ $running -ge $max_parallel ]]; do
            wait -n 2>/dev/null || true
            running=$(jobs -rp | wc -l)
        done

        local log="$LOGDIR/${fw}_${mem}_${bench}.log"
        launched=$((launched + 1))
        echo "  [$launched/$total] $fw + $mem + $bench → $log"

        (
            python3 hmf/alfworld_runners/run_full_experiment.py \
                --benchmark "$bench" --framework "$fw" --memory "$mem" \
                > "$log" 2>&1
        ) &
        running=$((running + 1))
    done

    echo ""
    echo ">>> $total experiments launched! (max $max_parallel concurrent)"
    echo ">>> Monitoring for errors..."

    # Monitor loop: surface errors while waiting
    while true; do
        running_jobs=$(jobs -rp | wc -l)
        if [[ "$running_jobs" -eq 0 ]]; then
            break
        fi
        for logfile in "$LOGDIR"/*.log; do
            [[ -f "$logfile" ]] || continue
            if tail -3 "$logfile" 2>/dev/null | grep -qE '\[ERROR\]|Traceback'; then
                base=$(basename "$logfile" .log)
                echo "  [FAILED] $base — $(tail -1 "$logfile" | head -c 120)"
            fi
        done | sort -u
        sleep 30
    done
    wait
    echo ""
    echo ">>> All $total experiments finished!"

    local errors=$(grep -rl 'ERROR\|Traceback' "$LOGDIR"/*.log 2>/dev/null | wc -l)
    if [[ "$errors" -gt 0 ]]; then
        echo ">>> WARNING: $errors logs contain errors. Check:"
        grep -l 'ERROR\|Traceback' "$LOGDIR"/*.log 2>/dev/null
    fi

    # Auto-generate LaTeX table from all results for this model
    local model_slug="${model//\//_}"
    echo ""
    echo ">>> Generating LaTeX table from results/exp_${model_slug}_*/ ..."
    python3 scripts/generate_table.py results/exp_${model_slug}_*/ \
        --output "$OUTDIR/table.tex" --model "$model" 2>&1
    echo ">>> Table saved to: $OUTDIR/table.tex"
}

# ─── Main ───────────────────────────────────────────────────────────────────

CMD="${1:-help}"
shift || true

case "$CMD" in
    setup)
        refresh_credentials
        start_proxy
        echo ""
        echo ">>> Setup complete. API endpoint: $OPENAI_API_BASE"
        echo ">>> Test: curl http://localhost:${LITELLM_PORT}/health"
        ;;
    run)
        run_experiments "$@"
        ;;
    status)
        check_status
        ;;
    stop)
        stop_proxy
        ;;
    logs)
        if [[ -d "$LOGDIR" ]]; then
            tail -f "$LOGDIR"/*.log
        else
            echo "No logs found in $LOGDIR"
        fi
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $CMD"
        usage
        exit 1
        ;;
esac
