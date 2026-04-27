# MemCon Experiment Scripts

## Quick Reference

### 1. Setup (every session)

```bash
# Refresh AWS credentials
source /home/ubuntu/workplace/0SysExp/source_all_bedrock_accounts.sh

# Regenerate LiteLLM config (only includes live AWS profiles)
cd /home/ubuntu/workplace/MemCon
python3 scripts/gen_litellm_config.py

# Kill old proxy (must kill workers on the port, not just the parent)
fuser -k 4001/tcp 2>/dev/null; sleep 3

# Start proxy
nohup .venv/bin/litellm --config litellm_config.yaml --port 4001 --num_workers 16 > litellm_proxy.log 2>&1 &
sleep 15

# Verify
curl -s http://localhost:4001/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d.get('data',[])])"
```

### 2. Change model

Edit `scripts/gen_litellm_config.py` — uncomment/comment the MODELS list:
```python
MODELS = [
    {"name": "sonnet-4", ...},    # uncomment to use
    # {"name": "haiku-3", ...},   # comment out to disable
]
```
Then rerun setup steps above (gen config + restart proxy).

### 3. Run experiments

**Run all 11 baselines (each with its own log):**
```bash
./scripts/run_all_baselines.sh
```

**Run specific baselines:**
```bash
./scripts/run_all_baselines.sh metagpt voyager generative
```

**Run with a different model (default is sonnet-4):**
```bash
MODEL=haiku-3 ./scripts/run_all_baselines.sh
```

**Run specific combinations:**
```bash
./scripts/run.sh run --benchmark alfworld,pddl --framework lobster --memory metagpt,voyager --model sonnet-4 --max-parallel 36 --exp-name my_test -b
```

### 4. Check progress

```bash
# Which experiments are running
pgrep -c -f "run_full_experiment"

# Quick status of all background logs
tail -1 run_*.log

# Check a specific baseline
tail -f run_metagpt.log

# Check for errors
grep -l "ERROR" run_*.log

# Count completed results
ls results/exp_sonnet-4/*.json | wc -l
```

### 5. Rerun failed experiments

**Auto-detect and rerun failures (scans logs for errors, skips completed JSONs):**
```bash
LLM_MODEL=sonnet-4 ./scripts/rerun_missing.sh --auto-fix -b --max-parallel 72
```

**Auto-detect missing JSONs and run them:**
```bash
LLM_MODEL=sonnet-4 ./scripts/rerun_missing.sh -b --max-parallel 72
```

**Manual rerun (edit the MANUAL_RUNS list in the script first):**
```bash
LLM_MODEL=sonnet-4 ./scripts/rerun_missing.sh --manual -b
```

### 6. Generate LaTeX table from results

```bash
python3 scripts/generate_table.py results/exp_sonnet-4/ --model sonnet-4
```

### 7. Kill everything

```bash
# Kill all experiments
pkill -f "run_full_experiment"

# Kill proxy (use fuser, not pkill — kills worker processes too)
fuser -k 4001/tcp
```

### 8. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Invalid model name` | Proxy has wrong config | Rerun setup (gen config + restart proxy) |
| `security token expired/invalid` | AWS creds expired | Refresh creds + restart proxy |
| `Legacy model` | Some AWS accounts block the model | Retry handles this; if persistent, reduce accounts |
| `CUDA out of memory` | Too many embeddings on GPU | Already fixed: uses CPU |
| `ModuleNotFoundError` | Missing pip package | `.venv/bin/pip install <package>` |

## Available memory methods

| Name | Description |
|------|-------------|
| `empty` | No memory (baseline) |
| `g-memory` | Graph-based hierarchical memory |
| `memcon` | MemCon — our method |
| `latentmem` | LatentMem variant (different prompts, no merge) |
| `metagpt` | Pure vector similarity search |
| `voyager` | LLM-summarized task storage |
| `generative` | LLM-scored retrieval ranking |
| `chatdev` | Periodic LLM summarization |
| `memorybank` | Temporal decay with forgetting |
| `oagent` | Insight learning (no task graph) |
| `experiencebank` | LLM-scored retrieval + re-ranking |

## File structure

```
scripts/
  run.sh                  # Main experiment launcher
  run_all_baselines.sh    # Launch each baseline individually
  rerun_missing.sh        # Rerun failed/missing experiments
  gen_litellm_config.py   # Generate LiteLLM proxy config
  generate_table.py       # Generate LaTeX table from results
  sweep_hyperparams.py    # Hyperparameter sweep for MemCon
```
