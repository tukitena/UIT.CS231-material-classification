"""Độ đo đánh giá (báo cáo mục IV.1.4, công thức (20)–(22)).

Phụ trách: C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._todo import todo


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    precision: np.ndarray  # (23,)
    recall: np.ndarray  # (23,)
    f1: np.ndarray  # (23,)
    support: np.ndarray  # (23,)
    confusion: np.ndarray  # (23, 23), hàng = nhãn thật, cột = nhãn dự đoán, số đếm thô

    def to_dict(self) -> dict:
        """Dạng JSON được: mảng → list. Dùng để ghi ``metrics_<split>.json``."""
        todo("C", "Metrics.to_dict")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 23) -> Metrics:
    """Accuracy, Macro-F1 trên đủ ``n_classes`` lớp, P/R/F1 từng lớp, confusion matrix.

    - Dùng ``labels=range(n_classes)`` để lớp vắng mặt vẫn được tính.
    - Lớp không có dự đoán đúng nào có F1 = 0 (``zero_division=0``).
    """
    todo("C", "evaluate")


def save_confusion_png(confusion: np.ndarray, class_names: list[str], path: Path) -> None:
    """Vẽ confusion matrix **chuẩn hóa theo hàng** (chia cho support) và lưu PNG (Hình 13)."""
    todo("C", "save_confusion_png")
