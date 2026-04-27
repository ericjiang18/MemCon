#!/usr/bin/env python3
"""
Generate LaTeX table from MemCon experiment result JSONs.

Reads results from a directory containing files like:
    {framework}_{memory}_{benchmark}.json

Outputs a LaTeX table* environment matching the paper format.

Usage:
    python scripts/generate_table.py results/exp_gpt-4.1-mini_20260419/
    python scripts/generate_table.py results/exp_gpt-4.1-mini_20260419/ --output latex/table_generated.tex
    python scripts/generate_table.py results/exp_*/ --merge   # merge multiple dirs
    python3 scripts/generate_table.py results/exp_sonnet-4/ --model sonnet-4 --output results/exp_sonnet-4/table.tex 2>&1
"""

import argparse
import glob
import json
import os
import sys


FRAMEWORKS = ["lobster", "langgraph", "agent_framework"]
FW_DISPLAY = {"lobster": "Lobster", "langgraph": "LangGraph", "agent_framework": "Agent-FW"}

MEMORIES = ["empty", "g-memory", "metagpt", "voyager", "generative", "chatdev", "memorybank", "oagent", "experiencebank", "latentmem", "memcon"]
MEM_DISPLAY = {
    "empty": "Empty",
    "g-memory": "G-Memory",
    "metagpt": "MetaGPT",
    "voyager": "Voyager",
    "generative": "Generative",
    "chatdev": "ChatDev",
    "memorybank": "MemoryBank",
    "oagent": "OAgent",
    "experiencebank": "ExpBank",
    "latentmem": "LatentMem",
    "memcon": r"\ours",
}

INTERACTIVE_BENCHMARKS = ["alfworld", "pddl", "sciworld"]
QA_BENCHMARKS = ["triviaqa", "webwalkerqa", "gaia"]
ALL_BENCHMARKS = INTERACTIVE_BENCHMARKS + QA_BENCHMARKS
BENCH_DISPLAY = {
    "alfworld": "ALFWorld",
    "pddl": "PDDL",
    "sciworld": "SciWorld",
    "triviaqa": "TriviaQA",
    "webwalkerqa": "WebWalkerQA",
    "gaia": "GAIA",
}

# Baseline to compute delta against
DELTA_BASELINE = "g-memory"


def load_results(result_dirs):
    """Load all result JSONs from one or more directories."""
    data = {}  # (framework, memory, benchmark) -> dict
    for d in result_dirs:
        for path in glob.glob(os.path.join(d, "*.json")):
            if "/logs/" in path:
                continue
            fname = os.path.basename(path)
            # Parse {framework}_{memory}_{benchmark}.json
            # Memory names can contain hyphens, so parse carefully
            parts = fname.replace(".json", "")
            for fw in FRAMEWORKS:
                if parts.startswith(fw + "_"):
                    rest = parts[len(fw) + 1:]
                    for bench in ALL_BENCHMARKS:
                        if rest.endswith("_" + bench):
                            mem = rest[:-(len(bench) + 1)]
                            try:
                                with open(path) as f:
                                    result = json.load(f)
                                data[(fw, mem, bench)] = result
                            except (json.JSONDecodeError, IOError) as e:
                                print(f"  [WARN] Failed to read {path}: {e}", file=sys.stderr)
                            break
                    break
    return data


def get_sa(result):
    """Extract success/accuracy percentage."""
    if "success_rate" in result:
        return result["success_rate"] * 100
    if "accuracy" in result:
        return result["accuracy"] * 100
    return None


def get_tok(result):
    """Extract avg tokens per task."""
    if "avg_tokens_per_task" in result:
        return result["avg_tokens_per_task"]
    tokens = result.get("tokens", {})
    total = tokens.get("total", tokens.get("total_tokens", 0))
    n = result.get("total_tasks", 1)
    return total // max(n, 1) if total else None


def fmt_sa(val):
    """Format S/A value."""
    if val is None:
        return "--"
    return f"{val:.1f}"


def fmt_tok(val):
    """Format token count."""
    if val is None:
        return "--"
    if val >= 1000:
        return f"{val / 1000:.0f}K"
    return str(int(val))


def fmt_delta(val, is_best=False):
    """Format delta. Only highlight green if is_best (max delta per fw×bench)."""
    if val is None:
        return "--"
    if val > 0:
        if is_best:
            return rf"\cellcolor{{mygreen}}+{val:.1f}"
        return f"+{val:.1f}"
    elif val == 0:
        return "0.0"
    else:
        return f"{val:.1f}"


def generate_latex(data, model_name=""):
    """Generate the LaTeX table string."""
    lines = []
    n_mem = len(MEMORIES)
    n_bench = len(ALL_BENCHMARKS)

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    caption = (
        r"\caption{Overall performance across interactive and QA benchmarks. "
        r"Succ./Acc. = success rate or accuracy (\%). "
        r"Tok/T = average tokens per task. "
        r"$\Delta$ is improvement over G-Memory. \textbf{Bold}: best per row."
    )
    if model_name:
        caption += f" Model: \\texttt{{{model_name}}}."
    caption += "}"
    lines.append(caption)
    lines.append(r"\label{tab:combined-results}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{")

    # Column spec: ll + 3*n_bench columns
    col_spec = "ll " + " ".join(["ccc"] * n_bench)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: Interactive / QA grouping
    n_interactive = len(INTERACTIVE_BENCHMARKS)
    n_qa = len(QA_BENCHMARKS)
    lines.append(
        f"& & \\multicolumn{{{n_interactive * 3}}}{{c}}{{\\textbf{{Interactive Benchmarks}}}} "
        f"& \\multicolumn{{{n_qa * 3}}}{{c}}{{\\textbf{{QA Benchmarks}}}} \\\\"
    )
    # cmidrule for interactive and QA groups
    int_start, int_end = 3, 3 + n_interactive * 3 - 1
    qa_start, qa_end = int_end + 1, int_end + n_qa * 3
    lines.append(f"\\cmidrule(lr){{{int_start}-{int_end}}} \\cmidrule(lr){{{qa_start}-{qa_end}}}")

    # Header row 2: benchmark names
    bench_headers = []
    for bench in ALL_BENCHMARKS:
        bench_headers.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{BENCH_DISPLAY[bench]}}}}}")
    lines.append("& & " + " \n  & ".join(bench_headers) + " \\\\")

    # cmidrule per benchmark
    cmidrules = []
    for i, bench in enumerate(ALL_BENCHMARKS):
        s = 3 + i * 3
        e = s + 2
        cmidrules.append(f"\\cmidrule(lr){{{s}-{e}}}")
    lines.append(" ".join(cmidrules))

    # Header row 3: S/A Tok Delta for each
    metric_headers = " & ".join(["S/A & Tok & $\\Delta$"] * n_bench)
    lines.append(f"\\textbf{{Framework}} & \\textbf{{Memory}} & {metric_headers} \\\\")
    lines.append(r"\midrule")

    # Data rows
    for fi, fw in enumerate(FRAMEWORKS):
        # Find best S/A per benchmark in this framework block
        best_sa = {}
        best_tok = {}
        for bench in ALL_BENCHMARKS:
            sas = []
            toks = []
            for mem in MEMORIES:
                r = data.get((fw, mem, bench))
                if r:
                    sa = get_sa(r)
                    tok = get_tok(r)
                    if sa is not None:
                        sas.append((sa, mem))
                    if tok is not None:
                        toks.append((tok, mem))
            if sas:
                best_sa[bench] = max(sas, key=lambda x: x[0])[0]
            if toks:
                best_tok[bench] = min(toks, key=lambda x: x[0])[0]

        # Get G-Memory baseline values for delta
        baseline_sa = {}
        for bench in ALL_BENCHMARKS:
            r = data.get((fw, DELTA_BASELINE, bench))
            if r:
                baseline_sa[bench] = get_sa(r)

        # Pre-compute best (max) delta per benchmark in this framework block
        best_delta = {}
        for bench in ALL_BENCHMARKS:
            deltas = []
            for mem in MEMORIES:
                if mem == "empty" or mem == DELTA_BASELINE:
                    continue
                r = data.get((fw, mem, bench))
                if r and bench in baseline_sa and baseline_sa[bench] is not None:
                    sa = get_sa(r)
                    if sa is not None:
                        d = round(sa - baseline_sa[bench], 1)
                        if d > 0:
                            deltas.append(d)
            if deltas:
                best_delta[bench] = max(deltas)

        lines.append(f"\\multirow{{{n_mem}}}{{*}}{{{FW_DISPLAY[fw]}}}")

        for mi, mem in enumerate(MEMORIES):
            cells = []
            for bench in ALL_BENCHMARKS:
                r = data.get((fw, mem, bench))
                if r:
                    sa = get_sa(r)
                    tok = get_tok(r)

                    # Bold best
                    sa_str = fmt_sa(sa)
                    tok_str = fmt_tok(tok)
                    if sa is not None and bench in best_sa and sa == best_sa[bench]:
                        sa_str = f"\\textbf{{{sa_str}}}"
                    if tok is not None and bench in best_tok and tok == best_tok[bench]:
                        tok_str = f"\\textbf{{{tok_str}}}"

                    # Delta
                    if mem == "empty" or mem == DELTA_BASELINE:
                        delta_str = "--"
                    elif sa is not None and bench in baseline_sa and baseline_sa[bench] is not None:
                        delta_val = round(sa - baseline_sa[bench], 1)
                        is_best = (delta_val > 0 and bench in best_delta
                                   and delta_val == best_delta[bench])
                        delta_str = fmt_delta(delta_val, is_best=is_best)
                    else:
                        delta_str = "--"

                    cells.append(f"{sa_str} & {tok_str} & {delta_str}")
                else:
                    cells.append("-- & -- & --")

            row = f"& {MEM_DISPLAY[mem]} \n& " + "\n& ".join(cells) + " \\\\"
            lines.append(row)

        # Add midrule between frameworks (not after last)
        if fi < len(FRAMEWORKS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\end{tabular}")
    lines.append("}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from MemCon results")
    parser.add_argument("result_dirs", nargs="+", help="Result directories to read")
    parser.add_argument("--output", "-o", help="Output .tex file (default: stdout)")
    parser.add_argument("--model", default="", help="Model name for caption")
    args = parser.parse_args()

    # Expand globs
    dirs = []
    for d in args.result_dirs:
        expanded = glob.glob(d)
        dirs.extend(expanded if expanded else [d])
    dirs = [d for d in dirs if os.path.isdir(d)]

    if not dirs:
        print("No valid result directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading results from: {dirs}", file=sys.stderr)
    data = load_results(dirs)
    print(f"Loaded {len(data)} result files", file=sys.stderr)

    if not data:
        print("No results found! Check directory and file naming.", file=sys.stderr)
        sys.exit(1)

    # Detect model from results
    model = args.model
    if not model:
        for v in data.values():
            if "model" in v:
                model = v["model"]
                break

    latex = generate_latex(data, model)

    if args.output:
        with open(args.output, "w") as f:
            f.write(latex)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(latex)


if __name__ == "__main__":
    main()
