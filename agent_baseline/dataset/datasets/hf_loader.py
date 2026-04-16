"""Safe HuggingFace datasets import avoiding local package collision.

Strategy: swap sys.modules and sys.path on EVERY call, because
the HF load_dataset function itself needs HF's submodules present.
"""

import sys
import os
import importlib

_hf_modules_cache = {}
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _swap_in_hf():
    """Remove local datasets from sys.modules/path, return restore info."""
    removed_paths = []
    for p in list(sys.path):
        if os.path.abspath(p) == os.path.abspath(_project_root):
            sys.path.remove(p)
            removed_paths.append(p)

    local_modules = {}
    for k in list(sys.modules.keys()):
        if k == "datasets" or k.startswith("datasets."):
            local_modules[k] = sys.modules.pop(k)

    if _hf_modules_cache:
        sys.modules.update(_hf_modules_cache)

    return removed_paths, local_modules


def _swap_back(removed_paths, local_modules):
    """Restore local datasets and save HF modules for next call."""
    global _hf_modules_cache
    _hf_modules_cache = {}
    for k in list(sys.modules.keys()):
        if k == "datasets" or k.startswith("datasets."):
            _hf_modules_cache[k] = sys.modules.pop(k)

    sys.modules.update(local_modules)

    for p in removed_paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_hf_dataset(path, *args, **kwargs):
    """Load a HuggingFace dataset, handling the local package name collision."""
    removed_paths, local_modules = _swap_in_hf()
    try:
        if "datasets" not in sys.modules:
            importlib.import_module("datasets")
        hf = sys.modules["datasets"]
        return hf.load_dataset(path, *args, **kwargs)
    finally:
        _swap_back(removed_paths, local_modules)
