import json
import re
from typing import List, Optional
from .base_dataset import BaseDataset, DataSample


class MMLUProDataset(BaseDataset):
    """MMLU-Pro benchmark for advanced multi-domain QA."""

    def __init__(self, split="validation", limit=None, data_path=None, subject=None):
        super().__init__(split, limit, data_path)
        self.domain = "mmlu_pro"
        self.subject = subject

    def _load(self) -> List[DataSample]:
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("TIGER-Lab/MMLU-Pro", split=self.split)
        except Exception:
            if self.data_path:
                return self._load_from_file(self.data_path)
            return []

        samples = []
        for item in ds:
            if self.subject and item.get("category", "") != self.subject:
                continue
            options = item.get("options", [])
            option_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            task = f"Question: {item['question']}\n\nOptions:\n{option_str}"

            samples.append(DataSample(
                task_id=str(item.get("question_id", len(samples))),
                task=task,
                ground_truth=item.get("answer", ""),
                metadata={"category": item.get("category", ""), "options": options},
                domain=self.domain,
            ))
        return samples

    def _load_from_file(self, path: str) -> List[DataSample]:
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("id", str(len(samples))),
                    task=item.get("question", ""),
                    ground_truth=item.get("answer", ""),
                    metadata=item,
                    domain=self.domain,
                ))
        return samples

    def evaluate(self, prediction: str, ground_truth: str) -> float:
        pred = re.search(r"\b([A-J])\b", prediction.upper())
        pred_letter = pred.group(1) if pred else prediction.strip()[:1].upper()
        return 1.0 if pred_letter == ground_truth.strip().upper() else 0.0
