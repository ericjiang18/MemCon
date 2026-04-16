from datasets import load_dataset
import os
import json
import datetime
import numpy as np

save_dir = "./code_test"
os.makedirs(save_dir, exist_ok=True)

def safe_json(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)

# -----------------------------
# 1. LiveCodeBench
# -----------------------------
lcb = load_dataset("livecodebench/code_generation", split="test")

lcb_path = os.path.join(save_dir, "livecodebench_test.jsonl")
with open(lcb_path, "w", encoding="utf-8") as f:
    for item in lcb:
        f.write(json.dumps(item, ensure_ascii=False, default=safe_json) + "\n")

print("Saved LiveCodeBench test to:", lcb_path)

# -----------------------------
# 2. HumanEval
# -----------------------------
humaneval = load_dataset("openai/openai_humaneval", split="test")

he_path = os.path.join(save_dir, "humaneval_test.jsonl")
with open(he_path, "w", encoding="utf-8") as f:
    for item in humaneval:
        f.write(json.dumps(item, ensure_ascii=False, default=safe_json) + "\n")

print("Saved HumanEval test to:", he_path)