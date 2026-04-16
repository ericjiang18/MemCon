"""Safe Python code execution with timeout."""

import subprocess
import tempfile
import os
from typing import Tuple, Optional


def execute_code_get_return(code: str, timeout: int = 30) -> str:
    """Execute Python code safely in a subprocess and return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        os.unlink(tmp_path)


def check_correctness(code: str, tests: list, timeout: int = 10) -> Tuple[float, str]:
    """Run code against test cases and return (score, feedback)."""
    if not tests:
        return 0.0, "No tests provided"

    passed = 0
    feedback_lines = []

    for test in tests:
        full_code = f"{code}\n{test}"
        try:
            result = subprocess.run(
                ["python3", "-c", full_code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                passed += 1
                feedback_lines.append(f"PASS: {test}")
            else:
                feedback_lines.append(f"FAIL: {test} -> {result.stderr.strip()[:100]}")
        except subprocess.TimeoutExpired:
            feedback_lines.append(f"TIMEOUT: {test}")
        except Exception as e:
            feedback_lines.append(f"ERROR: {test} -> {str(e)[:100]}")

    score = passed / len(tests) if tests else 0.0
    return score, "\n".join(feedback_lines)
