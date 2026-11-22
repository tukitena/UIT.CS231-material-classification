"""Hai góc nhìn của mỗi ảnh (báo cáo mục III.2, công thức (13)).

Phụ trách: A.

| Góc nhìn | Cách lấy                       | Kích thước | HOG     | HSV                    |
|----------|--------------------------------|------------|---------|------------------------|
| full     | resize RGB, INTER_AREA         | 128 × 128  | ảnh xám | ảnh 128 × 128          |
| center   | cắt giữa, cạnh s = ⌊min(h,w)/2⌋ | 181 × 181  | ảnh xám | resize về 128 × 128    |
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .._todo import todo

View = Literal["full", "center"]
VIEWS: tuple[View, ...] = ("full", "center")
FULL_SIZE = 128


def full_view(rgb: np.ndarray, size: int = FULL_SIZE) -> np.ndarray:
    """Resize ảnh RGB ``uint8`` về ``(size, size, 3)`` bằng ``cv2.INTER_AREA``."""
    todo("A", "full_view")


def center_view(rgb: np.ndarray) -> np.ndarray:
    """Cắt hình vuông chính giữa cạnh ``s = min(h, w) // 2``, giữ độ phân giải gốc.

    Gốc trên-trái: ``((h - s) // 2, (w - s) // 2)``. Ảnh 362 × 362 → 181 × 181.
    """
    todo("A", "center_view")


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB ``uint8`` → ảnh xám ``uint8`` bằng ``cv2.COLOR_RGB2GRAY``."""
    todo("A", "to_gray")


def extract_views(rgb: np.ndarray) -> dict[View, np.ndarray]:
    """Trả ``{"full": full_view(rgb), "center": center_view(rgb)}`` (vẫn là RGB)."""
    return {"full": full_view(rgb), "center": center_view(rgb)}
