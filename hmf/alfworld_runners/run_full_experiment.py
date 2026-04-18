#!/usr/bin/env python3
"""
Full experiment suite: 3 frameworks × 3 memories × 9 benchmarks.

Interactive benchmarks (ALFWorld, SciWorld, PDDL) use the ALFWorldRunner
with framework-specific LLM backends.

QA/Code benchmarks (humaneval, livecodebench, aime_2025, beyond_aime,
hmmt_feb_2025, hle) use agent_baseline/run_benchmark.py.

Usage:
    python hmf/alfworld_runners/run_full_experiment.py --benchmark alfworld --framework lobster --memory memcon
    python hmf/alfworld_runners/run_full_experiment.py --all
    python hmf/alfworld_runners/run_full_experiment.py --benchmark aime_2025 --framework langgraph --memory g-memory
"""

import argparse
import asyncio
import json
import os
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tasks"))
sys.path.insert(0, os.path.join(_REPO, "agent_baseline"))
os.chdir(_REPO)

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

INTERACTIVE_BENCHMARKS = ["alfworld", "sciworld", "pddl"]
QA_BENCHMARKS = ["humaneval", "aime_2025", "beyond_aime", "hmmt_feb_2025", "hle"]
ALL_BENCHMARKS = INTERACTIVE_BENCHMARKS + QA_BENCHMARKS
FRAMEWORKS = ["lobster", "langgraph", "agent_framework"]
MEMORIES = ["memcon", "g-memory", "empty"]


def get_runner(framework: str):
    kwargs = dict(model=MODEL, api_key=API_KEY, api_base=API_BASE)
    if framework == "langgraph":
        from hmf.alfworld_runners.langgraph_runner import LangGraphALFWorld
        return LangGraphALFWorld(**kwargs)
    elif framework == "lobster":
        from hmf.alfworld_runners.lobster_runner import LobsterALFWorld
        return LobsterALFWorld(**kwargs)
    elif framework == "agent_framework":
        from hmf.alfworld_runners.agent_framework_runner import AgentFrameworkALFWorld
        return AgentFrameworkALFWorld(**kwargs)
    raise ValueError(f"Unknown framework: {framework}")


def get_memory(memory_name: str, framework: str):
    if memory_name == "empty":
        return None

    from mas.module_map import module_map
    from mas.llm import GPTChat
    from mas.utils import EmbeddingFunc

    _, mem_cls = module_map("io", memory_name)
    llm = GPTChat(model_name=MODEL)
    embed = EmbeddingFunc("sentence-transformers/all-MiniLM-L6-v2")

    working_dir = os.path.join(".db", MODEL.replace("/", "_"), f"exp_{framework}_{memory_name}")
    os.makedirs(working_dir, exist_ok=True)

    return mem_cls(
        namespace=memory_name,
        global_config={"working_dir": working_dir, "hop": 1},
        llm_model=llm,
        embedding_func=embed,
    )


def run_interactive(benchmark: str, framework: str, memory_name: str, out_dir: str):
    """Run ALFWorld/SciWorld/PDDL via the generic runner."""
    runner = get_runner(framework)
    mem_backend = get_memory(memory_name, framework)
    out_path = os.path.join(out_dir, f"{framework}_{memory_name}_{benchmark}.json")

    summary = runner.run_alfworld(
        memory_backend=mem_backend,
        output_path=out_path,
        max_tasks=134 if benchmark == "alfworld" else 100,
    )
    return summary


def run_qa(benchmark: str, framework: str, memory_name: str, out_dir: str):
    """Run QA/code benchmarks via agent_baseline runners."""
    from runners import get_runner as get_baseline_runner
    from runners.base_runner import GenerateResult

    fw_name = framework
    if memory_name == "memcon":
        fw_name = f"{framework}_hmf"
    elif memory_name == "g-memory":
        fw_name = f"{framework}_gmemory" if framework == "langgraph" else framework

    limit = 250 if benchmark == "hle" else None

    config = {"base_url": API_BASE, "api_key": API_KEY, "model": MODEL}

    try:
        runner = get_baseline_runner(fw_name, **config)
    except ValueError:
        from hmf.alfworld_runners.lobster_runner import LobsterALFWorld
        print(f"  [WARN] No baseline runner for {fw_name}, using lobster OpenAI client")
        return _run_qa_simple(benchmark, memory_name, out_dir, framework, limit)

    from dataset.datasets import get_dataset
    from dataset.prompt import PromptSetRegistry
    from run_benchmark import DOMAIN_MAP, resolve_benchmark

    bench_key = resolve_benchmark(benchmark)
    domain = DOMAIN_MAP[bench_key]

    ds = get_dataset(bench_key, limit=limit)
    prompt_set = PromptSetRegistry[domain]

    out_path = os.path.join(out_dir, f"{framework}_{memory_name}_{benchmark}.json")
    summary = asyncio.run(runner.run_benchmark(
        dataset=ds, prompt_set=prompt_set,
        output_dir=out_dir, benchmark_name=f"{framework}_{memory_name}_{benchmark}",
    ))
    return summary


def _load_local_qa(benchmark: str, limit=None):
    """Load QA data from local JSONL files — no HuggingFace dependency."""
    import jsonlines
    _map = {
        "humaneval": "data/code_test/humaneval_test.jsonl",
        "aime_2025": "data/qa_test/aime_2025.jsonl",
        "beyond_aime": "data/math_test/BeyondAIME__test.jsonl",
        "hmmt_feb_2025": "data/math_test/MathArena__hmmt_feb_2025.jsonl",
        "hotpotqa": "data/qa_test/hotpotqa.jsonl",
        "webwalkerqa": "data/qa_test/webwalkerqa.jsonl",
        "travelplanner": "data/qa_test/travelplanner.jsonl",
        "popqa": "data/qa_test/popqa.jsonl",
        "triviaqa": "data/qa_test/triviaqa.jsonl",
        "gaia": "data/qa_test/gaia.jsonl",
        "assistantbench": "data/qa_test/assistantbench.jsonl",
        "hle": None,
    }
    path = _map.get(benchmark)
    if not path or not os.path.exists(path):
        # Fallback to HF loader
        from dataset.datasets import get_dataset
        from run_benchmark import resolve_benchmark
        return get_dataset(resolve_benchmark(benchmark), limit=limit), True

    samples = []
    with jsonlines.open(path) as reader:
        for item in reader:
            samples.append(item)
            if limit and len(samples) >= limit:
                break
    return samples, False


def _run_qa_simple(benchmark: str, memory_name: str, out_dir: str, framework: str, limit=None):
    """Run QA with simple OpenAI client + optional memory context."""
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    _use_new = any(t in MODEL for t in ("gpt-5", "o3", "o4"))

    data, is_hf = _load_local_qa(benchmark, limit)

    if is_hf:
        from dataset.prompt import PromptSetRegistry
        from run_benchmark import DOMAIN_MAP, resolve_benchmark
        bench_key = resolve_benchmark(benchmark)
        domain = DOMAIN_MAP[bench_key]
        prompt_set = PromptSetRegistry[domain]
    else:
        prompt_set = None

    mem_backend = get_memory(memory_name, framework) if memory_name != "empty" else None

    # Build system prompt based on benchmark type
    if benchmark in ("humaneval",):
        system_prompt = "You are an expert programmer. Write the complete function implementation. Output ONLY code, no explanation."
    elif benchmark in ("aime_2025", "beyond_aime", "hmmt_feb_2025"):
        system_prompt = ("You are a math competition expert. Solve the problem step by step. "
                        "Put your final numerical answer in \\boxed{}.")
    elif benchmark == "hle":
        system_prompt = "You are a knowledgeable expert. Answer the multiple choice question. Output ONLY the letter (A/B/C/D/E)."
    elif benchmark == "hotpotqa":
        system_prompt = ("You are a multi-hop reasoning expert. Use the provided context paragraphs to answer the question. "
                        "Give a short, precise answer (a few words).")
    elif benchmark == "webwalkerqa":
        system_prompt = ("You are a knowledgeable web research expert. Answer the question based on your knowledge. "
                        "Give a precise, factual answer.")
    elif benchmark == "travelplanner":
        system_prompt = ("You are a travel planning expert. Create a day-by-day itinerary that satisfies all constraints. "
                        "Include transportation, meals, attractions, and accommodation for each day.")
    elif benchmark in ("popqa", "triviaqa"):
        system_prompt = ("You are a knowledgeable trivia expert. Answer the question with a short, precise factual answer. "
                        "Output ONLY the answer, nothing else.")
    elif benchmark == "gaia":
        system_prompt = ("You are a capable AI assistant. Answer the question precisely and concisely. "
                        "Think step by step if needed, then give your final answer on the last line.")
    elif benchmark == "assistantbench":
        system_prompt = ("You are a capable web research assistant. Answer the question as precisely as possible. "
                        "Give a short, factual answer.")
    else:
        system_prompt = "Solve the following problem carefully."

    if prompt_set:
        try:
            system_prompt = prompt_set.get_decision_role().strip() + "\n\n" + prompt_set.get_decision_constraint().strip()
            few_shot = prompt_set.get_decision_few_shot()
            if few_shot:
                system_prompt += "\n\n" + few_shot.strip()
        except Exception:
            pass

    results = []
    correct = 0.0
    total_pt = total_ct = 0
    sample_list = list(data)

    for idx, sample in enumerate(sample_list):
        # Extract task/question and ground truth
        if is_hf:
            task = prompt_set.get_answer_prompt(sample.task)
            gt = sample.ground_truth
            task_id = sample.task_id
        elif benchmark == "hotpotqa":
            task = f"Context:\n{sample.get('context', '')}\n\nQuestion: {sample['question']}"
            gt = sample.get("answer", "")
            task_id = idx
        elif benchmark == "webwalkerqa":
            task = f"Question: {sample['question']}\n(Source website: {sample.get('root_url', '')})"
            gt = sample.get("answer", "")
            task_id = idx
        elif benchmark == "travelplanner":
            task = sample.get("query", "")
            gt = ""
            task_id = idx
        elif benchmark in ("popqa", "triviaqa"):
            task = f"Question: {sample['question']}"
            gt = sample.get("answer", [])
            task_id = idx
        elif benchmark in ("gaia", "assistantbench"):
            task = f"Question: {sample['question']}"
            gt = str(sample.get("answer", ""))
            task_id = idx
        else:
            task = sample.get("problem", sample.get("prompt", sample.get("task", str(sample))))
            gt = str(sample.get("answer", sample.get("solution", sample.get("ground_truth", ""))))
            task_id = sample.get("id", sample.get("task_id", idx))
        
        user_prompt = task

        mem_ctx = ""
        if mem_backend:
            short_task = task[:200].split("\n")[0]  # First line only for graph nodes
            mem_backend.init_task_context(short_task, short_task)
            mem_result = mem_backend.retrieve_memory(query_task=short_task, successful_topk=2, insight_topk=5)
            if mem_result[2]:
                mem_ctx = "\n\nRelevant insights:\n" + "\n".join(f"- {i}" for i in mem_result[2][:5])

        full_system = system_prompt + mem_ctx

        params = dict(
            model=MODEL,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
        )
        if _use_new:
            params["max_completion_tokens"] = 4096
        else:
            params["max_tokens"] = 4096

        try:
            resp = client.chat.completions.create(**params)
            text = resp.choices[0].message.content or ""
            pt = resp.usage.prompt_tokens or 0
            ct = resp.usage.completion_tokens or 0
        except Exception as e:
            text = f"ERROR: {e}"
            pt = ct = 0

        total_pt += pt
        total_ct += ct

        # Score the answer
        import re as _re
        score = 0.0
        if is_hf and prompt_set:
            try:
                processed = prompt_set.postprocess_answer(text)
                score = data.evaluate(processed, gt)
            except Exception:
                score = 0.0
        elif benchmark == "hotpotqa":
            # F1-based scoring for HotpotQA
            pred_tokens = set(text.lower().split())
            gt_tokens = set(str(gt).lower().split())
            if pred_tokens and gt_tokens:
                common = pred_tokens & gt_tokens
                prec = len(common) / len(pred_tokens) if pred_tokens else 0
                rec = len(common) / len(gt_tokens) if gt_tokens else 0
                score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if str(gt).lower() in text.lower():
                score = max(score, 1.0)
        elif benchmark in ("gaia", "assistantbench"):
            # Exact or containment match for GAIA/AssistantBench
            text_lower = text.lower().strip()
            gt_lower = str(gt).lower().strip()
            if gt_lower and gt_lower in text_lower:
                score = 1.0
            elif gt_lower:
                # Check last line
                last_line = text.strip().split("\n")[-1].lower().strip()
                if gt_lower in last_line or last_line == gt_lower:
                    score = 1.0
        elif benchmark in ("popqa", "triviaqa"):
            # Alias-based matching: score 1 if any acceptable answer appears in output
            text_lower = text.lower().strip()
            answers = gt if isinstance(gt, list) else [gt]
            for ans in answers:
                ans_lower = str(ans).lower().strip()
                if ans_lower and (ans_lower in text_lower or text_lower == ans_lower):
                    score = 1.0
                    break
        elif benchmark == "travelplanner":
            # TravelPlanner: check if plan mentions key elements
            text_lower = text.lower()
            has_transport = any(w in text_lower for w in ("flight", "drive", "bus", "train", "taxi"))
            has_hotel = any(w in text_lower for w in ("hotel", "accommodation", "stay", "airbnb"))
            has_food = any(w in text_lower for w in ("breakfast", "lunch", "dinner", "restaurant", "meal"))
            has_attraction = any(w in text_lower for w in ("visit", "museum", "park", "tour", "attraction"))
            score = (has_transport + has_hotel + has_food + has_attraction) / 4.0
        elif benchmark == "webwalkerqa":
            # Simple containment check
            if str(gt).lower() in text.lower():
                score = 1.0
            else:
                pred_tokens = set(text.lower().split())
                gt_tokens = set(str(gt).lower().split())
                common = pred_tokens & gt_tokens
                score = len(common) / max(len(gt_tokens), 1)
                score = min(score, 1.0)
        else:
            # Simple exact/boxed match for local data
            boxed = _re.search(r"\\boxed\{(.+?)\}", text)
            pred = boxed.group(1).strip() if boxed else text.strip().split("\n")[-1].strip()
            gt_clean = str(gt).strip()
            if pred == gt_clean or pred.lower() == gt_clean.lower():
                score = 1.0
            elif benchmark == "humaneval":
                score = 0.0

        correct += score

        if mem_backend:
            mem_backend.save_task_context(label=score > 0, feedback=text[:300])
            mem_backend.backward(score > 0)

        results.append({"task_id": task_id, "score": score, "tokens": pt + ct})

        acc = correct / (idx + 1) * 100
        print(f"  [{idx+1}/{len(sample_list)}] score={score:.1f} acc={acc:.1f}% tok={pt+ct}")

    n = len(results)
    summary = {
        "framework": framework, "memory": memory_name, "benchmark": benchmark,
        "model": MODEL, "total_tasks": n,
        "accuracy": round(correct / max(n, 1), 4),
        "tokens": {"prompt": total_pt, "completion": total_ct, "total": total_pt + total_ct},
        "avg_tokens_per_task": (total_pt + total_ct) // max(n, 1),
    }

    out_path = os.path.join(out_dir, f"{framework}_{memory_name}_{benchmark}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  {framework}+{memory_name} on {benchmark}: {summary['accuracy']:.1%} "
          f"({(total_pt+total_ct):,} tokens)")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--framework", type=str, default=None)
    parser.add_argument("--memory", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    out_dir = f"results/full_experiment_{MODEL.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        combos = [(b, f, m) for b in ALL_BENCHMARKS for f in FRAMEWORKS for m in MEMORIES]
    else:
        benchmarks = [args.benchmark] if args.benchmark else ALL_BENCHMARKS
        frameworks = [args.framework] if args.framework else FRAMEWORKS
        memories = [args.memory] if args.memory else MEMORIES
        combos = [(b, f, m) for b in benchmarks for f in frameworks for m in memories]

    print(f"Running {len(combos)} experiments with {MODEL}")
    print(f"Output: {out_dir}/\n")

    for i, (bench, fw, mem) in enumerate(combos):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(combos)}] {fw} + {mem} on {bench}")
        print(f"{'='*60}\n")

        try:
            if bench in INTERACTIVE_BENCHMARKS:
                run_interactive(bench, fw, mem, out_dir)
            else:
                _run_qa_simple(bench, mem, out_dir, fw,
                               limit=250 if bench == "hle" else None)
        except Exception as e:
            print(f"  [ERROR] {fw}+{mem}+{bench}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
