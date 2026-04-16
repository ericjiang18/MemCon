# MemCon: Memory as a Controlled Process for Agentic Systems

**MemCon** models agent memory operations as a Markov Decision Process and learns an online policy to control *when*, *what*, and *how much* to retrieve, encode, consolidate, and forget. It is **backend-agnostic** — wrapping any existing memory system (G-Memory, vector stores, etc.) and learning to use it better through task feedback.

## Key Results

### ALFWorld (134 tasks, gpt-4.1-mini)

**MemCon consistently outperforms G-Memory across all agent frameworks, while using fewer tokens.**

| Framework | + G-Memory | + MemCon (Ours) | Δ Success | Δ Tokens |
|-----------|-----------|-----------------|-----------|----------|
| SkillMAS | 67.2% | **73.9%** | **+6.7%** | — |
| LangGraph | 61.2% | **70.1%** | **+8.9%** | **-17.3%** |
| Lobster | 61.9% | **63.4%** | **+1.5%** | **-1.4%** |
| Agent-Framework | 70.1% | 68.7% | -1.4% | **-5.2%** |
| No memory | 40.3% | — | — | — |

### Per-Task-Type Breakdown (SkillMAS backbone)

| Task Type | G-Memory | MemCon | Improvement |
|-----------|----------|--------|-------------|
| put | 83.3% | **87.5%** | +4.2% |
| clean | 77.4% | **80.6%** | +3.2% |
| heat | 60.9% | **69.6%** | +8.7% |
| cool | 90.5% | 90.5% | — |
| examine | 61.1% | **83.3%** | **+22.2%** |
| puttwo | 11.8% | **17.6%** | +5.8% |

## Method Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MemCon Controller                            │
│                                                                  │
│   State s = (goal_type, step_phase, is_stuck, mem_size,          │
│              plan_available, learning_phase)                      │
│                                                                  │
│   Policy π(s) → action a via UCB bandit:                         │
│     a = argmax [ Q(s,a) + c·√(ln N / Nₐ) ]                     │
│                                                                  │
│   Actions: RETRIEVE(top_k, insight_k, hop)                       │
│            PLAN_INJECT | RE_RETRIEVE | CONSOLIDATE               │
│            FORGET | NO_OP                                        │
│                                                                  │
│   Reward: success(+1) + efficiency_bonus − failure(−0.5)         │
│   Update: Q(s,a) ← Q(s,a) + α[γ^t · r − Q(s,a)]               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────┐                   │
│   │  ANY Memory Backend (wrapped)            │                   │
│   │  • G-Memory (graph + Chroma + insights)  │                   │
│   │  • Vector store                          │                   │
│   │  • Skill memory                          │                   │
│   │  • ...                                   │                   │
│   └──────────────────────────────────────────┘                   │
│                                                                  │
│   + Generalized Success Plans (object IDs → placeholders)        │
│   + Goal Decomposition (puttwo → 2× single put)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why it works

Existing memory systems use **fixed retrieval strategies** (always top_k=2, always hop=1). MemCon learns that:
- **Early tasks** (cold start): minimal retrieval works best — memory is empty anyway
- **Familiar goal types** (warm phase): plan injection is better than full retrieval
- **When stuck**: re-retrieve with different parameters helps
- **examine tasks**: success plans are highly transferable (+22%)

## Project Structure

```
├── hmf/
│   ├── mcp_framework/              # Core contribution
│   │   ├── memory_mdp.py           #   Memory MDP formulation (S, A, T, R)
│   │   ├── policy.py               #   Online UCB contextual bandit
│   │   └── wrapper.py              #   Backend-agnostic wrapper
│   ├── integrations/               # Memory backend implementations
│   │   ├── mas_memcon.py           #   MemCon wrapping G-Memory
│   │   ├── mas_amc.py              #   Anticipatory Memory Control (ablation)
│   │   ├── mas_hmf_memory.py       #   HMF with heuristic MPC (ablation)
│   │   ├── mas_hmf_static.py       #   HMF without controller (ablation)
│   │   ├── mas_hmf_learned.py      #   HMF with learned bandit (ablation)
│   │   └── mas_gmemory_enhanced.py #   G-Memory + cache (ablation)
│   ├── alfworld_runners/           # Multi-framework evaluation
│   │   ├── base.py                 #   Generic ALFWorld loop (mirrors SkillMAS)
│   │   ├── langgraph_runner.py     #   LangGraph backend
│   │   ├── lobster_runner.py       #   Lobster (OpenAI direct) backend
│   │   ├── agent_framework_runner.py #  MS Agent Framework backend
│   │   └── run_all.py              #   Run all framework×memory combos
│   ├── memory/                     # Memory substrates (cache/retrieval/skill)
│   ├── mpc/                        # Original MPC controller (ablation)
│   └── agent/                      # Orchestration layer
├── mas/                            # Core multi-agent framework
│   ├── memory/mas_memory/          #   G-Memory, SkillMemory, etc.
│   ├── module_map.py               #   Registry (--mas_memory memcon)
│   └── llm.py                      #   LLM backends
├── tasks/                          # Task runner and environments
│   ├── run.py                      #   Main entry point
│   ├── envs/                       #   ALFWorld, PDDL, SciWorld, Math/QA
│   └── mas_workflow/               #   SkillMAS execution loop
├── agent_baseline/                 # Multi-framework benchmark harness
├── method.tex                      # Methodology (LaTeX)
└── scripts/                        # Run scripts
```

## Quick Start

### Setup

```bash
conda create -n memcon python=3.12 -y && conda activate memcon

pip install -r hmf/requirements.txt
pip install alfworld && alfworld-download
pip install gym==0.26.2 scikit-image   # for pddlgym
```

### Configuration

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

### Run MemCon on ALFWorld (SkillMAS backbone)

```bash
python tasks/run.py \
    --task alfworld \
    --mas_type skill-mas \
    --mas_memory memcon \
    --model gpt-4.1-mini \
    --max_trials 30
```

### Run Multi-Framework Comparison

```bash
# All 6 combos: {lobster, langgraph, agent_framework} × {memcon, g-memory}
python hmf/alfworld_runners/run_all.py --limit 134

# Single combo
python hmf/alfworld_runners/run_all.py --combo lobster:memcon --limit 20
```

### Available Memory Backends

| `--mas_memory` | System | Description |
|----------------|--------|-------------|
| `memcon` | **MemCon (ours)** | Learned policy controlling G-Memory |
| `g-memory` | G-Memory | Graph + Chroma + insights (baseline) |
| `skill-rl` | Skill-Conditioned RL | Q-learning + ExpRAG + LLM Refine |
| `hmf` | HMF | Cache + Retrieval + Skill + heuristic MPC |
| `hmf-static` | HMF-Static | Same layers, fixed pipeline (ablation) |
| `hmf-learned` | HMF-Learned | Same layers, learned bandit (ablation) |
| `amc` | AMC | G-Memory + step-action memory (ablation) |
| `empty` | No memory | Baseline |

## Ablation Study (ALFWorld, SkillMAS, gpt-4.1-mini)

| Method | Success | Note |
|--------|---------|------|
| **MemCon** | **73.9%** | Learned policy over G-Memory |
| G-Memory | 67.2% | Graph + insight scoring |
| AMC | 66.4% | Step-level guidance, no policy learning |
| HMF-Static | 62.7% | Three layers, fixed pipeline |
| HMF-Learned | 59.0% | Three flat layers + bandit |
| HMF (heuristic MPC) | 58.2% | Three flat layers + hand-tuned costs |
| Empty | 40.3% | No memory |

## Related Work

- [G-Memory](https://arxiv.org/abs/2506.07398) — Graph-based hierarchical memory for MAS
- [LLMPC](https://arxiv.org/abs/2501.02486) — LLM as implicit cost-function optimizer
- [MemP](https://arxiv.org/abs/2508.06433) — Procedural memory with step-by-step abstractions
- [ProcMEM](https://arxiv.org/abs/2602.01869) — Skill-MDP with semantic gradients
- [MemSkill](https://arxiv.org/abs/2602.02474) — Learnable memory operations

## License

MIT
