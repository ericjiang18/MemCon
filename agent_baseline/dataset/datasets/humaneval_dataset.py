"""HumanEval benchmark with real code execution evaluation."""

import json
import re
import subprocess
import tempfile
import os
from typing import List, Optional
from .base_dataset import BaseDataset, DataSample


class HumanEvalDataset(BaseDataset):
    """OpenAI HumanEval benchmark for code generation."""

    def __init__(self, split="test", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "humaneval"

    def _load(self) -> List[DataSample]:
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("openai_humaneval", split="test")
        except Exception:
            if self.data_path:
                return self._load_from_file(self.data_path)
            return []

        samples = []
        for item in ds:
            samples.append(DataSample(
                task_id=item["task_id"],
                task=item["prompt"],
                ground_truth=item["canonical_solution"],
                metadata={"test": item["test"], "entry_point": item["entry_point"]},
                domain=self.domain,
            ))
        return samples

    def _load_from_file(self, path: str) -> List[DataSample]:
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("task_id", ""),
                    task=item.get("prompt", ""),
                    ground_truth=item.get("canonical_solution", ""),
                    metadata=item,
                    domain=self.domain,
                ))
        return samples

    def evaluate(self, prediction: str, ground_truth: str) -> float:
        """Execute code with HumanEval test cases. Returns 1.0 (pass) or 0.0 (fail)."""
        from utils.code_extract import extract_code
        code = extract_code(prediction)
        if not code:
            return 0.0

        sample = self._find_sample(ground_truth)
        if sample is None or "test" not in sample.metadata:
            return 0.0

        prompt = sample.task
        test_fn = sample.metadata["test"]
        entry_point = sample.metadata.get("entry_point", "")

        # If model returned a full function, use it directly; otherwise indent as body
        if re.search(r"^def\s+\w+\s*\(", code.strip(), re.MULTILINE):
            full_code = f"{code}\n\n{test_fn}\n\ncheck({entry_point})\n"
        else:
            full_code = f"{prompt}\n{code}\n\n{test_fn}\n\ncheck({entry_point})\n"

        try:
            tmp = None
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                tmp = f.name

            result = subprocess.run(
                ["python3", tmp], capture_output=True, text=True, timeout=15
            )
            os.unlink(tmp)
            return 1.0 if result.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
            return 0.0
        except Exception:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
            return 0.0

    def _find_sample(self, ground_truth: str):
        """Find the matching sample by ground_truth or task_id."""
        for s in self.samples:
            if s.ground_truth == ground_truth:
                return s
        for s in self.samples:
            if ground_truth[:50] in s.ground_truth or s.ground_truth[:50] in ground_truth:
                return s
        return self.samples[0] if self.samples else None
