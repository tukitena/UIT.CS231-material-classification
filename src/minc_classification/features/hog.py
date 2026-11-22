"""Descriptor block HOG (báo cáo mục II.3, III.3.1, công thức (14)).

Phụ trách: B.

Mỗi block 2 × 2 cell cho một descriptor 36 chiều. **Không** nối các descriptor theo vị trí: ảnh
được xem như một túi descriptor để mã hóa BoW.

Số block với ảnh vuông cạnh n: ``(⌊n/8⌋ - 1)²`` → 128: 225 block, 181: 441 block.
"""

from __future__ import annotations

import numpy as np

from .._todo import todo

ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
BLOCK_NORM = "L2-Hys"
DESC_DIM = ORIENTATIONS * CELLS_PER_BLOCK[0] * CELLS_PER_BLOCK[1]  # 36


def n_blocks(side: int) -> int:
    """Số descriptor block của ảnh vuông cạnh ``side``."""
    return (side // PIXELS_PER_CELL[0] - CELLS_PER_BLOCK[0] + 1) ** 2


def hog_blocks(gray: np.ndarray) -> np.ndarray:
    """Descriptor block HOG của một ảnh xám.

    Gợi ý: ``skimage.feature.hog(..., block_norm="L2-Hys", feature_vector=False)`` trả mảng
    ``(rows, cols, 2, 2, 9)``; reshape thành ``(rows * cols, 36)``.

    Returns:
        ``float32`` shape ``(n_blocks(side), 36)``, mỗi hàng đã chuẩn hóa L2-Hys.
    """
    todo("B", "hog_blocks")
