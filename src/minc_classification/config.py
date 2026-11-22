"""Đọc config YAML và đường dẫn riêng của từng máy (``.env``).

- Config có thể khai báo ``base: <file>`` để kế thừa file khác (đường dẫn tương đối so với file
  hiện tại). Dict được trộn đệ quy; list và giá trị đơn bị ghi đè hoàn toàn.
- Schema đầy đủ: ``doc/config-schema.md``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = ("hsv", "bow_full", "bow_center", "bow_multi", "hsv_bow_multi")
"""Biểu diễn hợp lệ, lần lượt ứng với E1 / E2 / E3 / E4 / E5–E7."""

CLASSIFIERS = ("linear_svm", "knn")


@dataclass(frozen=True)
class Paths:
    dataset: Path
    cache: Path
    runs: Path


def paths() -> Paths:
    load_dotenv(REPO_ROOT / ".env")

    def resolve(var: str, default: str) -> Path:
        p = Path(os.environ.get(var, default))
        return p if p.is_absolute() else REPO_ROOT / p

    return Paths(
        dataset=resolve("DATASET_ROOT", "data/minc-2500"),
        cache=resolve("CACHE_ROOT", "cache"),
        runs=resolve("RUNS_ROOT", "runs"),
    )


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    cfg["_source"] = path.as_posix()
    return cfg


def validate(cfg: dict[str, Any]) -> None:
    """Bắt sớm lỗi config hay gặp, trước khi tốn thời gian trích đặc trưng."""
    for key in ("experiment", "dataset", "features", "bow", "representation", "classifier"):
        if key not in cfg:
            raise ValueError(f"Config thiếu khối '{key}'")
    if not cfg["experiment"].get("name"):
        raise ValueError("experiment.name đang trống")
    rep, clf = cfg["representation"], cfg["classifier"]

    if clf["type"] not in CLASSIFIERS:
        raise ValueError(f"classifier.type phải thuộc {CLASSIFIERS}, nhận {clf['type']!r}")

    if "survey" in cfg:  # E8: K lấy từ survey.k_grid, chỉ đánh giá trên validation
        for v in cfg["survey"]["variants"]:
            if v not in VARIANTS:
                raise ValueError(f"survey.variants chứa biến thể lạ {v!r}")
        return

    if rep["variant"] not in VARIANTS:
        raise ValueError(f"representation.variant phải thuộc {VARIANTS}, nhận {rep['variant']!r}")
    if rep["variant"] != "hsv" and cfg["bow"].get("k") is None:
        raise ValueError("bow.k đang trống. Chạy E8 trước rồi ghi K* vào configs/_base.yaml.")
    alphas = rep.get("alpha_grid", [0.5])
    if not alphas or any(not 0.0 <= a <= 1.0 for a in alphas):
        raise ValueError(
            f"representation.alpha_grid phải gồm các giá trị trong [0, 1], nhận {alphas}"
        )
