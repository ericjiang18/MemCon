"""Math competition datasets: AIME 2024/2025/2026, HMMT, BeyondAIME."""

import json
import re
from .base_dataset import BaseDataset, DataSample


def _extract_number(text):
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return matches[-1] if matches else None


def _numbers_match(pred, truth, tol=1e-6):
    try:
        return abs(float(pred) - float(truth)) < tol
    except (ValueError, TypeError):
        return pred.strip().lower() == truth.strip().lower()


class _MathBaseDataset(BaseDataset):
    """Shared evaluator for all math competition datasets."""

    def evaluate(self, prediction, ground_truth):
        p, t = _extract_number(prediction), _extract_number(ground_truth)
        if p and t:
            return 1.0 if _numbers_match(p, t) else 0.0
        return 1.0 if prediction.strip() == ground_truth.strip() else 0.0

    def _load_from_file(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("id", item.get("ID", item.get("problem_idx", str(len(samples))))),
                    task=item.get("problem", item.get("Problem", item.get("question", ""))),
                    ground_truth=str(item.get("answer", item.get("Answer", ""))),
                    metadata=item, domain=self.domain,
                ))
        return samples


class AIME2024Dataset(_MathBaseDataset):
    """Maxwell-Jia/AIME_2024 — used for TRAINING."""

    def __init__(self, split="train", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "aime"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("Maxwell-Jia/AIME_2024", split="train")
            return [DataSample(
                task_id=str(item.get("ID", i)),
                task=item.get("Problem", ""),
                ground_truth=str(item.get("Answer", "")),
                metadata={"solution": item.get("Solution", "")},
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[aime_2024] HF load failed: {e}")
            return []


class AIME2025Dataset(_MathBaseDataset):
    """MathArena/aime_2025"""

    def __init__(self, split="train", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "aime"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("MathArena/aime_2025", split="train")
            return [DataSample(
                task_id=str(item.get("problem_idx", i)),
                task=item["problem"],
                ground_truth=str(item["answer"]),
                metadata={"type": item.get("problem_type", "")},
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[aime_2025] HF load failed: {e}")
            return []


class AIME2026Dataset(_MathBaseDataset):
    """MathArena/aime_2026"""

    def __init__(self, split="train", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "aime"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("MathArena/aime_2026", split="train")
            return [DataSample(
                task_id=str(item.get("problem_idx", i)),
                task=item["problem"],
                ground_truth=str(item["answer"]),
                metadata={},
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[aime_2026] HF load failed: {e}")
            return []


class BeyondAIMEDataset(_MathBaseDataset):
    """ByteDance-Seed/BeyondAIME"""

    def __init__(self, split="test", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "aime"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("ByteDance-Seed/BeyondAIME", split="test")
            return [DataSample(
                task_id=str(i),
                task=item["problem"],
                ground_truth=str(item["answer"]),
                metadata={},
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[beyond_aime] HF load failed: {e}")
            return []


class HMMT2025Dataset(_MathBaseDataset):
    """MathArena/hmmt_feb_2025"""

    def __init__(self, split="train", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "hmmt"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("MathArena/hmmt_feb_2025", split="train")
            return [DataSample(
                task_id=str(item.get("problem_idx", i)),
                task=item["problem"],
                ground_truth=str(item["answer"]),
                metadata={"type": item.get("problem_type", "")},
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[hmmt_2025] HF load failed: {e}")
            return []
