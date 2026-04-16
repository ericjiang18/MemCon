# Hierarchical Memory Framework with Model Predictive Control for Agentic Systems

This repository implements a **hierarchical memory framework (HMF)** that unifies three complementary memory substrates — **cache**, **retrieval**, and **skill** — orchestrated by a **Model Predictive Control (MPC)** layer. It also includes the baseline **Skill-Conditioned RL** and **G-Memory** systems for comparison.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MPC Controller (Predictive Control)                  │
│                                                                         │
│  At each step, optimizes over horizon H to decide:                      │
│  • Reuse cached results     • Retrieve past experiences                 │
│  • Invoke stored skill      • Fresh LLM generation                     │
│  • Consolidate into skills  • Evict stale memory                       │
│  Under constraints: token cost ≤ B_token, latency ≤ B_latency          │
│                                                                         │
│          ┌──────────────────┼──────────────────┐                       │
│          ▼                  ▼                  ▼                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │ Cache Memory  │  │  Retrieval   │  │ Skill Memory │                 │
│  │ LRU+TTL, O(1)│  │  Semantic    │  │  Procedural  │                 │
│  │ hash + embed  │  │  vectors +   │  │  distillation│                 │
│  │ soft match    │  │  importance  │  │  + evolution  │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── hmf/                        # Hierarchical Memory Framework (main contribution)
│   ├── config.py               #   Central configuration
│   ├── memory/                 #   Three memory substrates
│   │   ├── cache_memory.py     #     LRU+TTL cache with semantic soft-match
│   │   ├── retrieval_memory.py #     Importance-weighted vector store
│   │   └── skill_memory.py     #     Procedural knowledge + trajectory distillation
│   ├── mpc/                    #   Model Predictive Control layer
│   │   ├── controller.py       #     Rolling-horizon optimizer (heuristic + LLM scoring)
│   │   ├── cost_model.py       #     Multi-objective: α·token + β·latency + γ·uncertainty - δ·utility
│   │   └── state.py            #     Joint memory + task state tracking
│   ├── agent/                  #   Orchestration layer
│   │   ├── hmf_agent.py        #     Full task lifecycle management
│   │   └── trajectory.py       #     Step-by-step execution traces
│   ├── integrations/           #   Framework adapters
│   │   ├── mas_hmf_memory.py   #     MASMemoryBase adapter (for tasks/run.py)
│   │   ├── langgraph_hmf.py    #     LangGraph + HMF runner
│   │   ├── langgraph_gmemory.py#     LangGraph + G-Memory runner
│   │   ├── agent_framework_hmf.py #  Agent-Framework + HMF runner
│   │   └── lobster_hmf.py      #     Lobster + HMF runner
│   └── evaluation/             #   Extended metrics (token efficiency, cache hits, etc.)
│
├── mas/                        # Core multi-agent system framework
│   ├── llm.py                  #   LLM backends (OpenAI, Gemini, Qwen, Claude)
│   ├── module_map.py           #   Registry: memory name → class
│   ├── memory/mas_memory/      #   Baseline memory implementations
│   │   ├── GMemory.py          #     G-Memory (graph-based, Chroma + NetworkX)
│   │   ├── gmemory_plus.py     #     G-Memory++ (goal module + skill mining)
│   │   ├── skill_memory.py     #     Skill-Conditioned RL + ExpRAG + LLM Refine
│   │   ├── skill_rl.py         #     Q(goal_type, skill_id) with UCB
│   │   ├── skill_miner.py      #     FINCH clustering → LLM skill extraction
│   │   └── goal_module.py      #     Goal parsing
│   └── agents/, reasoning/, tools/
│
├── tasks/                      # Task runner and environments
│   ├── run.py                  #   Main entry point
│   ├── configs.yaml            #   Global configuration
│   ├── envs/                   #   ALFWorld, PDDL, SciWorld, Math/QA environments
│   ├── mas_workflow/           #   SkillMAS execution loop (Think-Act-Refine)
│   └── prompts/                #   Task-specific prompt templates
│
├── agent_baseline/             # Multi-framework benchmark harness
│   ├── run_benchmark.py        #   Unified benchmark entry
│   ├── runners/                #   LangGraph / AutoGen / Lobster + HMF variants
│   └── dataset/                #   Dataset loaders and prompt registries
│
├── scripts/
│   ├── hmf/                    #   HMF experiment scripts (comparison, per-benchmark)
│   ├── legacy/                 #   Original gemini/claude scripts (Skill-RL baseline)
│   ├── clean_memory.sh         #   Utility: wipe .db/
│   └── download_qa_data.py     #   Download AIME/GPQA/MMLU-Pro from HuggingFace
│
├── data/                       #   Benchmark data (qa_test, math_test, pddl, humaneval)
├── configs/                    #   LLM configuration
└── results/                    #   Experiment outputs
```

## Quick Start

### Setup

```bash
# Create and activate conda environment
conda create -n hmf python=3.12 -y
conda activate hmf

# Install dependencies
pip install -r hmf/requirements.txt

# For ALFWorld
pip install alfworld && alfworld-download

# For SciWorld
pip install scienceworld
```

### Configuration

```bash
# Set in .env or export
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

### Running Experiments

```bash
# ALFWorld: three-way comparison (empty vs g-memory vs hmf)
bash scripts/hmf/run_alfworld_comparison.sh

# Individual benchmarks with HMF memory
bash scripts/hmf/run_hmf_alfworld.sh
bash scripts/hmf/run_hmf_pddl.sh
bash scripts/hmf/run_hmf_aime25.sh
bash scripts/hmf/run_hmf_gpqa.sh

# Direct command
python3 tasks/run.py \
    --task alfworld \
    --mas_type skill-mas \
    --mas_memory hmf \          # hmf | g-memory | skill-rl | empty
    --model gpt-4.1-mini \
    --max_trials 30
```

### Available Memory Backends

| `--mas_memory` | System | Description |
|----------------|--------|-------------|
| `empty` | No memory | Baseline with no cross-task memory |
| `g-memory` | G-Memory | Graph-based hierarchical memory (Chroma + NetworkX) |
| `skill-rl` | Skill-Conditioned RL | Q-learning skill selection + ExpRAG + LLM Refine |
| `hmf` | **HMF (ours)** | Cache + Retrieval + Skill + MPC controller |

## Supported Benchmarks

| Benchmark | Type | Tasks | Via |
|-----------|------|-------|----|
| ALFWorld | Household interaction | 134 | `tasks/run.py` |
| PDDL | Classical planning | 120 | `tasks/run.py` |
| SciWorld | Science experiments | varies | `tasks/run.py` |
| AIME 2024/2025 | Math competition | 30 | `tasks/run.py` |
| GPQA | Graduate-level QA | varies | `tasks/run.py` |
| MMLU-Pro | Multi-domain knowledge | varies | `tasks/run.py` |
| HumanEval | Code generation | 164 | `agent_baseline/` |

## Related Work

- **LLMPC** — LLM as implicit cost-function optimizer for predictive control ([arXiv:2501.02486](https://arxiv.org/abs/2501.02486))
- **MemP** — Procedural memory with step-by-step abstractions ([arXiv:2508.06433](https://arxiv.org/abs/2508.06433))
- **ProcMEM** — Skill-MDP with semantic gradients and PPO Gate ([arXiv:2602.01869](https://arxiv.org/abs/2602.01869))
- **MemSkill** — Learnable memory operations with closed-loop skill evolution ([arXiv:2602.02474](https://arxiv.org/abs/2602.02474))
- **G-Memory** — Graph-based hierarchical memory for MAS ([arXiv:2506.07398](https://arxiv.org/abs/2506.07398))

## License

MIT
