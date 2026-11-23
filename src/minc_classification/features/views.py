"""Hai góc nhìn của mỗi ảnh (báo cáo mục III.2, công thức (13)).

Phụ trách: A.

| Góc nhìn | Cách lấy                       | Kích thước | HOG     | HSV                    |
|----------|--------------------------------|------------|---------|------------------------|
| full     | resize RGB, INTER_AREA         | 128 × 128  | ảnh xám | ảnh 128 × 128          |
| center   | cắt giữa, cạnh s = ⌊min(h,w)/2⌋ | 181 × 181  | ảnh xám | resize về 128 × 128    |
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

View = Literal["full", "center"]
VIEWS: tuple[View, ...] = ("full", "center")
FULL_SIZE = 128


def full_view(rgb: np.ndarray, size: int = FULL_SIZE) -> np.ndarray:
    """Resize ảnh RGB ``uint8`` về ``(size, size, 3)`` bằng ``cv2.INTER_AREA``."""
    _check_rgb(rgb)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def center_view(rgb: np.ndarray) -> np.ndarray:
    """Cắt hình vuông chính giữa cạnh ``s = min(h, w) // 2``, giữ độ phân giải gốc.

    Gốc trên-trái: ``((h - s) // 2, (w - s) // 2)``. Ảnh 362 × 362 → 181 × 181.
    """
    _check_rgb(rgb)
    h, w = rgb.shape[:2]
    s = min(h, w) // 2
    top, left = (h - s) // 2, (w - s) // 2
    return np.ascontiguousarray(rgb[top : top + s, left : left + s])


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB ``uint8`` → ảnh xám ``uint8`` bằng ``cv2.COLOR_RGB2GRAY``."""
    _check_rgb(rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def extract_views(rgb: np.ndarray) -> dict[View, np.ndarray]:
    """Trả ``{"full": full_view(rgb), "center": center_view(rgb)}`` (vẫn là RGB)."""
    return {"full": full_view(rgb), "center": center_view(rgb)}


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Cần ảnh RGB uint8 (H, W, 3), nhận {rgb.dtype} {rgb.shape}")
