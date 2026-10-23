"""Config loading, seeding, hashing and environment capture."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config; an `extends:` key deep-merges onto a parent file."""
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else REPO_ROOT / path
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def resolve(cfg: dict, key: str) -> Path:
    """Return a path from cfg['paths'], anchored at the repo root, creating it."""
    p = REPO_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: a few CUDA kernels (e.g. attention backward) have no
    # deterministic variant; GPU runs are reproducible to ~1e-6, not bit-exact.
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def environment_snapshot() -> dict:
    import matplotlib
    import pandas
    import scipy
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": gpu,
        "gpu_total_memory_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / 2**20) if gpu else None
        ),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": yaml.__version__,
        "git_commit": git_commit(),
    }


def write_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
