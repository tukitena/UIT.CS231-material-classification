"""Histogram màu HSV (báo cáo mục II.2, III.4, công thức (3)).

Phụ trách: B.

Ảnh được resize về 128 × 128 trước khi chuyển sang HSV (cả với vùng tâm 181 × 181).

| Kênh | Miền (OpenCV, 8 bit) | Số bin |
|------|----------------------|--------|
| H    | 0–179                | 16     |
| S    | 0–255                | 8      |
| V    | 0–255                | 8      |
"""

from __future__ import annotations

import numpy as np

from .._todo import todo

HSV_SIZE = 128
BINS = (16, 8, 8)
RANGES = ((0, 180), (0, 256), (0, 256))
HSV_DIM = sum(BINS)  # 32


def hsv_hist(rgb: np.ndarray) -> np.ndarray:
    """Histogram HSV của một góc nhìn.

    Các bước: resize về 128 × 128 (INTER_AREA) → ``cv2.COLOR_RGB2HSV`` → histogram từng kênh
    với ``BINS``/``RANGES`` → chuẩn hóa L1 **từng kênh** → nối theo thứ tự H, S, V.

    Returns:
        ``float32`` shape ``(32,)``; mỗi đoạn H/S/V có tổng bằng 1.
    """
    todo("B", "hsv_hist")
