# Agent Baseline: Multi-Agent Framework Benchmark

Benchmark suite that evaluates four industry-standard multi-agent frameworks on math, coding, and reasoning tasks. Each framework uses genuine multi-agent collaboration (not single-turn wrappers) with domain-specific agent roles.

## Frameworks

| Framework | Multi-Agent Pattern |
|---|---|---|
| **AutoGen** | `RoundRobinGroupChat` with domain-specific roles | 
| **LangGraph** | Multi-node `StateGraph` with conditional routing | 
| **Agent Framework** | Sequential pipeline with inter-agent communication | 
| **Lobster** | Single-agent baseline (direct API call) | 

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
cp .env.example .env
# Edit .env with your BASE_URL, API_KEY, LLM_MODEL
```

### Environment Variables

| Variable | Description | Example |
|---|---|---|
| `BASE_URL` | LLM API endpoint | `https://api.groq.com/openai/v1` |
| `API_KEY` | API key | your key |
| `LLM_MODEL` | Model name | `openai/gpt-oss-120b` |
| `GEMINI_API_KEY` | Google AI key (for Gemini scripts) | your key |

## Scripts

### Full Benchmark Runs

| Script | Model | Benchmarks | Output |
|---|---|---|---|
| `script/run_all.sh` | GPT-OSS:120B | 7 benchmarks × 4 frameworks | `results/` |
| `script/run_all_gemini.sh` | Gemini Flash Lite | 7 benchmarks × 4 frameworks | `results_gemini/` |
| `script/run_all_gemini_2.5.sh` | Gemini 2.5 Flash | 7 benchmarks × 4 frameworks | `results_gemini_2.5/` |
| `script/run_all_mercury.sh` | Mercury-2 | 7 benchmarks × 4 frameworks | `results_mercury/` |

### HLE (Humanity's Last Exam)

| Script | Model | Samples | Output |
|---|---|---|---|
| `script/run_hle_gemini_pro.sh` | Gemini Flash Lite | 250 MCQ | `results_hle_gemini_pro/` |
| `script/run_hle_gpt_oss.sh` | GPT-OSS:120B | 250 MCQ | `results_hle_gpt_oss/` |

### Single Framework / Benchmark

```bash
# Run one framework on one benchmark
python3 run_benchmark.py -f autogen -b mmlu --limit 10

# Quick sanity check (3 samples per benchmark)
bash script/run_quick_test.sh
```

## Benchmarks

| Benchmark | Domain | Samples | Type |
|---|---|---|---|
| HumanEval | Coding | 164 | Code generation |
| LiveCodeBench | Coding | 400 | Code generation |
| MMLU-Pro | Reasoning | 70 | Multiple choice (A-J) |
| AIME 2025 | Math | 30 | Numeric answer |
| AIME 2026 | Math | 30 | Numeric answer |
| Beyond-AIME | Math | 100 | Numeric answer |
| HMMT 2025 | Math | 30 | Numeric answer |
| HLE | Mixed | 250 (MCQ) | Multiple choice |

## Results

Each run produces JSON files in the output directory:

```
results_gemini/
├── agent-framework_humaneval.json
├── agent-framework_mmlu_pro.json
├── autogen_humaneval.json
├── autogen_mmlu_pro.json
├── langgraph_humaneval.json
├── lobster_humaneval.json
└── ...
```

Each JSON file contains:
```json
{
  "framework": "autogen",
  "benchmark": "mmlu_pro",
  "model": "gemini-3.1-flash-lite-preview",
  "accuracy": 0.8857,
  "total_samples": 70,
  "total_tokens": 286388,
  "results": [...]
}
```

View a summary table after any run:
```bash
python3 summarize_results.py results_gemini/
```

## Project Structure

```
agent_baseline/
├── run_benchmark.py          # Main entry point
├── summarize_results.py      # Generate comparison tables
├── runners/                  # Framework-specific runners
│   ├── base_runner.py        # Shared benchmark loop + multi-agent config
│   ├── autogen_runner.py     # AutoGen RoundRobinGroupChat
│   ├── langgraph_runner.py   # LangGraph StateGraph workflow
│   ├── agent_framework_runner.py  # Agent Framework pipeline
│   └── lobster_runner.py     # Single-agent baseline
├── dataset/
│   ├── datasets/             # Benchmark dataset loaders
│   └── prompt/               # Domain-specific prompt sets
├── script/                   # Shell scripts for batch runs
└── results*/                 # Output directories
```
