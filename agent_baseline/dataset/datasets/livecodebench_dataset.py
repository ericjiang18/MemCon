"""LiveCodeBench datasets: code_generation (test) and test_generation (train)."""

import json
import subprocess
import tempfile
import os
from .base_dataset import BaseDataset, DataSample


class LiveCodeBenchDataset(BaseDataset):
    """livecodebench/code_generation — for testing."""

    def __init__(self, split="test", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "livecodebench"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("livecodebench/code_generation", split="test")
            return [DataSample(
                task_id=str(item.get("question_id", i)),
                task=item.get("question_content", ""),
                ground_truth=json.dumps(
                    item.get("private_test_cases") or item.get("public_test_cases", [])
                ),
                metadata={
                    "title": item.get("question_title", ""),
                    "starter_code": item.get("starter_code", ""),
                    "difficulty": item.get("difficulty", ""),
                },
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[livecodebench] HF load failed: {e}")
            return []

    def _load_from_file(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("id", str(len(samples))),
                    task=item.get("question", item.get("prompt", "")),
                    ground_truth=item.get("solution", ""),
                    metadata=item, domain=self.domain,
                ))
        return samples

    def evaluate(self, prediction, ground_truth):
        if "```python" in prediction:
            prediction = prediction.split("```python")[1].split("```")[0]
        return 1.0 if prediction.strip() and len(prediction.strip()) > 10 else 0.0


class LiveCodeBenchTestGenDataset(BaseDataset):
    """livecodebench/test_generation — for training.

    Has function_name, starter_code, and test cases (input/output pairs).
    Can evaluate immediately by running the code against test cases.
    """

    def __init__(self, split="test", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "livecodebench"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("livecodebench/test_generation", split="test")
            return [DataSample(
                task_id=str(item.get("question_id", item.get("test_id", i))),
                task=self._format_task(item),
                ground_truth=json.dumps(item.get("test", [])),
                metadata={
                    "title": item.get("question_title", ""),
                    "function_name": item.get("function_name", ""),
                    "starter_code": item.get("starter_code", ""),
                    "difficulty": item.get("difficulty", ""),
                },
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[livecodebench_testgen] HF load failed: {e}")
            return []

    def _format_task(self, item):
        task = item.get("question_content", "")
        starter = item.get("starter_code", "")
        fname = item.get("function_name", "")
        if starter:
            task += f"\n\nFunction signature:\n```python\n{starter}\n```"
        if fname:
            task += f"\n\nFunction name: {fname}"
        return task

    def _load_from_file(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("id", str(len(samples))),
                    task=item.get("question", ""),
                    ground_truth=json.dumps(item.get("test", [])),
                    metadata=item, domain=self.domain,
                ))
        return samples

    def evaluate(self, prediction, ground_truth):
        """Run prediction code against test cases. Returns fraction of tests passed."""
        from utils.code_extract import extract_code
        code = extract_code(prediction)
        if not code:
            return 0.0

        try:
            test_cases = json.loads(ground_truth)
            if isinstance(test_cases, str):
                test_cases = json.loads(test_cases)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        if not test_cases or not isinstance(test_cases, list):
            return 0.0

        parsed_tcs = []
        for tc in test_cases:
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(tc, dict):
                parsed_tcs.append(tc)

        if not parsed_tcs:
            return 0.0

        # Handle LeetCode-style Solution class
        import re
        if "class Solution" in code:
            # Option A: Instantiate Solution
            instance_code = f"{code}\n_sol = Solution()"
            prefix = "_sol."
        else:
            instance_code = code
            prefix = ""

        # Find function name
        # We look for the last def that isn't __init__
        funcs = re.findall(r"def\s+(\w+)\s*\(", code)
        func_name = None
        for f in reversed(funcs):
            if f != "__init__":
                func_name = f
                break

        passed = 0
        for tc in parsed_tcs:
            inp = tc.get("input", "")
            expected = str(tc.get("output", "")).strip()
            testtype = tc.get("testtype", "functional")

            if testtype == "functional" and func_name:
                test_code = f"{instance_code}\n\n_result = {prefix}{func_name}({inp})\nprint(_result)"
            elif testtype == "stdin":
                test_code = instance_code
            else:
                test_code = f"{instance_code}\n\nprint({inp})" if inp else instance_code

            try:
                tmp = None
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(test_code)
                    tmp = f.name

                run_kwargs = {"capture_output": True, "text": True, "timeout": 10}
                if testtype == "stdin":
                    run_kwargs["input"] = inp

                result = subprocess.run(["python3", tmp], **run_kwargs)
                os.unlink(tmp)
                tmp = None

                actual = result.stdout.strip()
                # Debug logging for failed cases (print first failure)
                if actual != expected and passed == 0:
                    print(f"  [DEBUG] Failed case. Input: {inp[:50]}...")
                    print(f"  [DEBUG] Expected: {expected[:50]}")
                    print(f"  [DEBUG] Actual:   {actual[:50]}")
                    print(f"  [DEBUG] Code snippet: {code[:100]}...")

                if actual == expected:
                    passed += 1
                elif result.returncode == 0:
                    try:
                        if float(actual) == float(expected):
                            passed += 1
                    except (ValueError, TypeError):
                        pass
            except Exception:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)

        return passed / len(parsed_tcs)
