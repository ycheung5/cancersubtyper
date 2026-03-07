from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path


def _version_2_dir() -> Path:
    return Path(__file__).resolve().parent / "version_2_assets"


@lru_cache(maxsize=2)
def _load_module(filename: str, module_name: str):
    module_path = _version_2_dir() / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bctypefinder_module():
    return _load_module("bctypefinder_v2.py", "bctypefinder_v2")


def _cancersubminer_module():
    return _load_module("cancersubminer_v2.py", "cancersubminer_v2")


def preprocess_bctypefinder(*args, **kwargs):
    return _bctypefinder_module().preprocess_bctypefinder(*args, **kwargs)


def run_bctypefinder(*args, **kwargs):
    return _bctypefinder_module().run_bctypefinder(*args, **kwargs)


def preprocess_cancersubminer(*args, **kwargs):
    return _cancersubminer_module().preprocess_cancersubminer(*args, **kwargs)


def run_cancersubminer(*args, **kwargs):
    return _cancersubminer_module().run_cancersubminer(*args, **kwargs)
