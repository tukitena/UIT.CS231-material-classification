"""Chuẩn hóa và kết hợp đặc trưng (báo cáo mục III.3.4, III.4, III.5, công thức (16)–(19)).

Phụ trách: B.

Mọi hàm làm việc theo hàng: input shape ``(N, d)``.
"""

from __future__ import annotations

import numpy as np

from .._todo import todo

EPS = 1e-12


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 từng hàng. Hàng toàn 0 giữ nguyên 0 (chia cho ``max(‖x‖, EPS)``)."""
    todo("B", "l2_normalize")


def combine_views(full: np.ndarray, center: np.ndarray) -> np.ndarray:
    """``L2([L2(full); L2(center)])`` — dùng cho cả BoW (``h_BoW``) và HSV (``c``).

    Mỗi khối đóng góp đúng 1/2 vào bình phương chuẩn của kết quả.

    Returns:
        shape ``(N, d_full + d_center)``, chuẩn L2 từng hàng bằng 1.
    """
    todo("B", "combine_views")


def fuse(h_bow: np.ndarray, c: np.ndarray, alpha: float) -> np.ndarray:
    """``z = [√α · h_BoW ; √(1-α) · c]``, khối cạnh đứng trước, công thức (18).

    Giả định ``h_bow`` và ``c`` đã có chuẩn L2 bằng 1 ⇒ ``‖z‖² = 1``, và α là tỉ lệ đóng góp của
    khối cạnh. ``0 ≤ α ≤ 1``, ngoài khoảng → ``ValueError``.

    Returns:
        shape ``(N, 2K + 64)``.
    """
    todo("B", "fuse")
