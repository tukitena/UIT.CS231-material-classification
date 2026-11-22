"""Gộp kết quả các lần chạy thành bảng cho báo cáo (Bảng 9–12).

Phụ trách: C.

Mỗi lần chạy nằm trong ``<RUNS_ROOT>/<experiment.name>/`` với ``metadata.json``,
``selection.json``, ``metrics_test.json`` (E1–E7) hoặc ``survey.json`` (E8).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .._todo import todo

ABLATION_PAIRS: list[tuple[str, str, str]] = [
    ("E2", "E3", "Vùng tâm ở độ phân giải gốc có tốt hơn toàn ảnh không"),
    ("E2", "E4", "Đa tỉ lệ có tốt hơn chỉ dùng toàn ảnh không"),
    ("E3", "E4", "Đa tỉ lệ có tốt hơn chỉ dùng vùng tâm không"),
    ("E1", "E5", "Thêm cạnh vào màu có ích không"),
    ("E4", "E5", "Thêm màu vào cạnh có ích không"),
    ("E5", "E6", "Có cần chỉnh trọng số α không"),
    ("E5", "E7", "LinearSVC hay KNN phù hợp hơn"),
]
"""Cặp (A, B, câu hỏi). Δ = điểm(B) − điểm(A) trên test, công thức (23)."""


def survey_table(runs_root: Path) -> pd.DataFrame:
    """Bảng 9 (E8): mỗi hàng là (K, biến thể) với số chiều, Macro-F1 val, Accuracy val, thời gian
    học codebook / mã hóa / fit."""
    todo("C", "survey_table")


def main_table(runs_root: Path) -> pd.DataFrame:
    """Bảng 10: E1–E7 với số chiều, tham số đã chọn, Macro-F1 val, Macro-F1 test, Accuracy test."""
    todo("C", "main_table")


def ablation_table(main: pd.DataFrame) -> pd.DataFrame:
    """Bảng 11: Δ Macro-F1 và Δ Accuracy (test) cho từng cặp trong ``ABLATION_PAIRS``."""
    todo("C", "ablation_table")


def per_class_table(runs_root: Path, experiment: str = "e5_hsv_bow_multi") -> pd.DataFrame:
    """Bảng 12: Precision / Recall / F1 / support từng lớp của E5 trên test, kèm hàng macro."""
    todo("C", "per_class_table")


def write_all(runs_root: Path, out_dir: Path) -> None:
    """Ghi các bảng trên ra ``out_dir`` dưới dạng CSV và Markdown để dán vào báo cáo."""
    todo("C", "write_all")
