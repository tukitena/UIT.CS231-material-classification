"""Tái lập: seed, hash mã nguồn, metadata mỗi lần chạy."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import inspect
import json
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

LIBRARIES = (
    "numpy",
    "opencv-python-headless",
    "scikit-image",
    "scikit-learn",
    "datasets",
    "pyyaml",
)


def set_seed(seed: int) -> None:
    """Cố định RNG toàn cục cho thư viện nào còn dùng nó. Code của nhóm nên truyền seed tường
    minh (``np.random.default_rng(seed)``, ``random_state=seed``) thay vì dựa vào hàm này."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


def code_hash(*modules: ModuleType) -> str:
    """Hash mã nguồn các module. Sửa mã trích xuất ⇒ hash đổi ⇒ cache cũ tự bị bỏ qua."""
    h = hashlib.sha256()
    for mod in sorted(modules, key=lambda m: m.__name__):
        h.update(mod.__name__.encode())
        h.update(inspect.getsource(mod).encode())
    return h.hexdigest()[:16]


def array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def git_commit() -> str | None:
    """SHA của HEAD, thêm ``-dirty`` nếu có thay đổi chưa commit. ``None`` nếu chưa có Git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = "missing"
    return out


def machine_info() -> dict[str, str]:
    return {"platform": platform.platform(), "processor": platform.processor()}


def write_metadata(run_dir: Path, cfg: dict[str, Any], **extra: Any) -> Path:
    """Ghi ``metadata.json``: commit, seed, config, phiên bản thư viện, máy, cùng các trường thêm
    (revision dataset, cache signature, K, hash codebook, thời gian từng bước...)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": cfg["experiment"]["seed"],
        "dataset_revision": cfg["dataset"]["revision"],
        "libraries": library_versions(),
        "machine": machine_info(),
        "config": cfg,
        **extra,
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
