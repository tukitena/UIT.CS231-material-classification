"""Bộ phân loại và chọn siêu tham số trên validation (báo cáo mục II.5, II.6, III.6).

Phụ trách: C.

Quy tắc bắt buộc:
- Chỉ chọn theo **Macro-F1 trên validation**. Hàm ở đây không bao giờ nhận dữ liệu test.
- Hòa điểm ⇒ giữ ứng viên **đứng trước** trong lưới (lưới xếp theo thứ tự trong config).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin

from .._todo import todo


@dataclass
class Selection:
    best_params: dict[str, Any]
    best_score: float  # Macro-F1 validation của best_params
    model: ClassifierMixin  # đã fit trên train với best_params
    scores: list[tuple[dict[str, Any], float]] = field(default_factory=list)  # theo thứ tự lưới
    fit_seconds: float = 0.0  # tổng thời gian fit toàn lưới


def param_grid(clf_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Lưới tham số theo đúng thứ tự trong config. ``clf_cfg`` là khối ``classifier``; tham số
    riêng nằm ở ``clf_cfg[clf_cfg["type"]]``.

    - ``linear_svm``: ``[{"C": c} for c in clf_cfg["linear_svm"]["C_grid"]]``
    - ``knn``: ``[{"n_neighbors": k} for k in clf_cfg["knn"]["k_grid"]]``
    """
    todo("C", "param_grid")


def build_classifier(clf_cfg: dict[str, Any], params: dict[str, Any], seed: int) -> ClassifierMixin:
    """Tạo estimator sklearn chưa fit.

    - ``linear_svm``: ``LinearSVC(C=..., penalty="l2", loss="squared_hinge", max_iter=5000,
      tol=1e-4, random_state=seed)`` (penalty/loss/max_iter/tol lấy từ ``clf_cfg["linear_svm"]``).
    - ``knn``: ``KNeighborsClassifier(n_neighbors=..., weights="uniform", metric="euclidean")``
      (weights/metric lấy từ ``clf_cfg["knn"]``).
    """
    todo("C", "build_classifier")


def pick_best(scores: list[float]) -> int:
    """Chỉ số của điểm lớn nhất; hòa thì lấy chỉ số **nhỏ nhất** (ứng viên đứng trước)."""
    todo("C", "pick_best")


def select_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    clf_cfg: dict[str, Any],
    seed: int,
) -> Selection:
    """Fit từng ứng viên của ``param_grid`` trên train, chấm Macro-F1 trên validation, chọn bằng
    ``pick_best`` và trả model tốt nhất (đã fit trên train)."""
    todo("C", "select_on_validation")
