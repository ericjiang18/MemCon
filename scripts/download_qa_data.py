"""Download AIME-24, AIME-25, GPQA Diamond, MMLU-Pro (Eng.) from HuggingFace."""
import json
import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qa_test")
os.makedirs(OUT_DIR, exist_ok=True)


def _save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} samples → {path}")


def download_aime24():
    from datasets import load_dataset
    print("Downloading AIME 2024 ...")
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train", token=HF_TOKEN)
    records = []
    for i, item in enumerate(ds):
        problem = item.get("Problem") or item.get("problem")
        answer = item.get("Answer") or item.get("answer")
        if not problem:
            continue
        records.append({
            "problem": problem,
            "answer": str(answer),
            "choices": [],
            "source": "aime_2024",
            "id": i,
        })
    _save_jsonl(records, os.path.join(OUT_DIR, "aime_2024.jsonl"))
    return len(records)


def download_aime25():
    from datasets import load_dataset
    print("Downloading AIME 2025 ...")
    ds = load_dataset("MathArena/aime_2025", split="train", token=HF_TOKEN)
    records = []
    for i, item in enumerate(ds):
        records.append({
            "problem": item["problem"],
            "answer": str(item["answer"]),
            "choices": [],
            "source": "aime_2025",
            "id": i,
        })
    _save_jsonl(records, os.path.join(OUT_DIR, "aime_2025.jsonl"))
    return len(records)


def download_gpqa():
    from datasets import load_dataset
    print("Downloading GPQA Diamond ...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=HF_TOKEN)
    records = []
    for i, item in enumerate(ds):
        correct = item["Correct Answer"]
        distractors = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        options = [correct] + distractors
        import random
        random.seed(i)
        random.shuffle(options)
        correct_letter = chr(65 + options.index(correct))
        records.append({
            "problem": item["Question"],
            "answer": correct_letter,
            "choices": options,
            "source": "gpqa_diamond",
            "id": i,
        })
    _save_jsonl(records, os.path.join(OUT_DIR, "gpqa_diamond.jsonl"))
    return len(records)


def download_mmlu_pro_eng():
    from datasets import load_dataset
    print("Downloading MMLU-Pro (Engineering) ...")
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", token=HF_TOKEN)
    except Exception:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        path = hf_hub_download(
            repo_id="TIGER-Lab/MMLU-Pro",
            filename="data/test-00000-of-00001.parquet",
            repo_type="dataset",
            token=HF_TOKEN,
        )
        table = pq.read_table(path)
        ds = table.to_pylist()

    records = []
    idx = 0
    for item in ds:
        cat = item.get("category", "")
        if cat != "engineering":
            continue
        options = item.get("options", [])
        answer_idx = item.get("answer_index", None)
        answer_letter = item.get("answer", "")
        if answer_idx is not None and isinstance(answer_idx, int):
            answer_letter = chr(65 + answer_idx)
        records.append({
            "problem": item["question"],
            "answer": answer_letter,
            "choices": options,
            "source": "mmlu_pro_eng",
            "id": idx,
        })
        idx += 1
    _save_jsonl(records, os.path.join(OUT_DIR, "mmlu_pro_eng.jsonl"))
    return len(records)


if __name__ == "__main__":
    total = 0
    for fn in [download_aime24, download_aime25, download_gpqa, download_mmlu_pro_eng]:
        try:
            n = fn()
            total += n
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print(f"\nDone. Total samples: {total}")
