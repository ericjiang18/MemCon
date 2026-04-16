"""Humanity's Last Exam (HLE) — cais/hle benchmark.

Supports both multipleChoice and exactMatch answer types.
Image-based questions are included as text-only (image content is omitted).
"""

import json
import re
from typing import List, Optional
from .base_dataset import BaseDataset, DataSample


class HLEDataset(BaseDataset):
    """Humanity's Last Exam benchmark (multiple-choice only)."""

    def __init__(self, split="test", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "hle"

    def _load(self) -> List[DataSample]:
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("cais/hle", split=self.split)
        except Exception:
            if self.data_path:
                return self._load_from_file(self.data_path)
            return []

        samples = []
        for item in ds:
            if item.get("answer_type") != "multipleChoice":
                continue

            question = item.get("question", "")
            samples.append(DataSample(
                task_id=str(item.get("id", len(samples))),
                task=question,
                ground_truth=item.get("answer", ""),
                metadata={
                    "answer_type": "multipleChoice",
                    "category": item.get("category", ""),
                    "raw_subject": item.get("raw_subject", ""),
                    "has_image": bool(item.get("image")),
                },
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
        pred = prediction.strip()
        gt = ground_truth.strip()

        if not pred or not gt:
            return 0.0

        if pred.lower() == gt.lower():
            return 1.0

        mc_match = re.search(r"\b([A-Z])\b", pred)
        if mc_match and len(gt) == 1 and gt.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return 1.0 if mc_match.group(1).upper() == gt.upper() else 0.0

        pred_clean = re.sub(r"[^\w\s]", "", pred.lower())
        gt_clean = re.sub(r"[^\w\s]", "", gt.lower())
        if pred_clean == gt_clean:
            return 1.0

        if gt_clean in pred_clean or pred_clean in gt_clean:
            longer = max(len(pred_clean), len(gt_clean))
            shorter = min(len(pred_clean), len(gt_clean))
            if shorter > 0 and shorter / longer > 0.8:
                return 1.0

        return 0.0
