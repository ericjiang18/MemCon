# MemCon: Memory as a Controlled Process for LLM Agents

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MemCon** models agent memory operations as a Markov Decision Process and learns an online policy to control *when*, *what*, and *how much* to retrieve, encode, consolidate, and forget. It is **backend-agnostic** — wrapping any existing memory system and learning to use it better through task feedback, with zero pretraining and zero additional LLM calls.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                   MemCon Controller                      │
│                                                          │
│  State: (goal_type, step_phase, is_stuck,                │
│          mem_size, plan_available, learning_phase)        │
│                                                          │
│  Policy π(s) → action via UCB bandit:                    │
│    a = argmax [ Q(s,a) + c·√(ln N / Nₐ) ]              │
│                                                          │
│  Actions: RETRIEVE(top_k, insight_k, hop)                │
│           PLAN_INJECT | RE_RETRIEVE | CONSOLIDATE        │
│           FORGET | NO_OP                                 │
│                                                          │
│  Reward: success(+1) + efficiency − failure(−0.5)        │
│  Update: Q ← Q + α[γᵗ·r − Q]   (online, per task)      │
├──────────────────────────────────────────────────────────┤

```

## Setup

```bash
# Create conda environment
conda create -n memcon python=3.12 -y && conda activate memcon

# Install core dependencies
pip install -r hmf/requirements.txt

# Install agent frameworks
pip install langchain-openai langgraph agent-framework-core agent-framework-openai

# Install environment dependencies
pip install alfworld && alfworld-download
pip install scienceworld
pip install gym==0.26.2 scikit-image

# Configuration
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://api.openai.com/v1"
export ALFWORLD_DATA="$HOME/.cache/alfworld"
```

## Running Experiments

### Quick: 9 Core Experiments (3 benchmarks × 3 memories)

Each command runs one benchmark with one memory backend using the Lobster framework:

```bash
# ── ALFWorld (134 household tasks) ──

# 1. ALFWorld + No Memory
python hmf/alfworld_runners/run_all.py --combo lobster:empty --limit 134

# 2. ALFWorld + G-Memory
python hmf/alfworld_runners/run_all.py --combo lobster:g-memory --limit 134

# 3. ALFWorld + MemCon (ours)
python hmf/alfworld_runners/run_all.py --combo lobster:memcon --limit 134

# ── PDDL Planning (100 tasks) ──

# 4. PDDL + No Memory
python hmf/alfworld_runners/run_all.py --combo lobster:empty --limit 100

# 5. PDDL + G-Memory
python hmf/alfworld_runners/run_all.py --combo lobster:g-memory --limit 100

# 6. PDDL + MemCon (ours)
python hmf/alfworld_runners/run_all.py --combo lobster:memcon --limit 100

# ── ScienceWorld (100 tasks) ──

# 7. SciWorld + No Memory
python hmf/alfworld_runners/run_all.py --combo lobster:empty --limit 100

# 8. SciWorld + G-Memory
python hmf/alfworld_runners/run_all.py --combo lobster:g-memory --limit 100

# 9. SciWorld + MemCon (ours)
python hmf/alfworld_runners/run_all.py --combo lobster:memcon --limit 100
```

### QA Benchmarks (TriviaQA, WebWalkerQA)

```bash
# ── TriviaQA (200 tasks) ──

# 10. TriviaQA + No Memory (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark triviaqa --framework lobster --memory empty

# 11. TriviaQA + G-Memory (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark triviaqa --framework lobster --memory g-memory

# 12. TriviaQA + MemCon (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark triviaqa --framework lobster --memory memcon

# ── WebWalkerQA (200 tasks) ──

# 13. WebWalkerQA + No Memory (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark webwalkerqa --framework lobster --memory empty

# 14. WebWalkerQA + G-Memory (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark webwalkerqa --framework lobster --memory g-memory

# 15. WebWalkerQA + MemCon (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark webwalkerqa --framework lobster --memory memcon
```

### GAIA Benchmark (165 tasks)

```bash
# 16. GAIA + No Memory (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark gaia --framework lobster --memory empty

# 17. GAIA + G-Memory (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark gaia --framework lobster --memory g-memory

# 18. GAIA + MemCon (Lobster)
python hmf/alfworld_runners/run_full_experiment.py --benchmark gaia --framework lobster --memory memcon
```

### Multi-Framework (any benchmark × any framework × any memory)

```bash
# General form:
python hmf/alfworld_runners/run_full_experiment.py \
  --benchmark <alfworld|pddl|sciworld|triviaqa|webwalkerqa|gaia|assistantbench> \
  --framework <lobster|langgraph|agent_framework> \
  --memory <empty|g-memory|memcon>

# Run ALL combinations:
python hmf/alfworld_runners/run_full_experiment.py --all
```

## Project Structure

```
├── hmf/                             # MemCon framework
│   ├── mcp_framework/               #   Core: Memory MDP + Policy + Wrapper
│   │   ├── memory_mdp.py            #     State/Action/Reward formulation
│   │   ├── policy.py                #     UCB contextual bandit
│   │   └── wrapper.py               #     Backend-agnostic wrapper
│   ├── integrations/                #   Memory backend implementations
│   │   ├── mas_memcon.py            #     MemCon wrapping G-Memory
│   │   └── mas_*.py                 #     Ablation variants
│   ├── alfworld_runners/            #   Multi-framework evaluation
│   │   ├── base.py                  #     Generic ALFWorld/PDDL/SciWorld loop
│   │   ├── lobster_runner.py        #     OpenAI direct
│   │   ├── langgraph_runner.py      #     LangGraph
│   │   ├── agent_framework_runner.py#     MS Agent Framework
│   │   ├── run_all.py               #     Single experiment launcher
│   │   └── run_full_experiment.py   #     Full 72-experiment suite
│   ├── memory/                      #   Memory substrates (cache/retrieval/skill)
│   └── mpc/                         #   Original MPC controller (ablation)
│
├── mas/                             # Core multi-agent framework
│   ├── memory/mas_memory/           #   G-Memory, SkillMemory, etc.
│   ├── module_map.py                #   Memory backend registry
│   └── llm.py                       #   LLM backends (OpenAI, Gemini, Claude)
│
├── tasks/                           # Task runner and environments
│   ├── run.py                       #   Main entry (SkillMAS backbone)
│   ├── envs/                        #   ALFWorld, PDDL, SciWorld, Math/QA
│   └── mas_workflow/                #   SkillMAS Think-Act-Refine loop
│
├── agent_baseline/                  # Multi-framework benchmark harness
├── langgraph/                       # LangGraph framework (vendored)
├── lobster/                         # Lobster framework (vendored)
├── agent-framework/                 # MS Agent Framework (vendored)
├── data/                            # Benchmark datasets
└── scripts/                         # Run scripts
```

## Related Work

- [G-Memory](https://arxiv.org/abs/2506.07398) — Graph-based hierarchical memory for MAS
- [LLMPC](https://arxiv.org/abs/2501.02486) — LLM as implicit cost-function optimizer
- [MemP](https://arxiv.org/abs/2508.06433) — Procedural memory with step-by-step abstractions
- [ProcMEM](https://arxiv.org/abs/2602.01869) — Skill-MDP with semantic gradients
- [MemSkill](https://arxiv.org/abs/2602.02474) — Learnable memory operations

## Citation

```bibtex
@article{memcon2026,
  title={Memory as a Controlled Process: Learned Adaptive Memory Management for LLM Agents},
  author={Anonymous},
  year={2026}
}
```

## License

MIT
