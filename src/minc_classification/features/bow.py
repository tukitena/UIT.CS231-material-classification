"""Túi từ thị giác trên descriptor block HOG (báo cáo mục II.4, III.3.2–III.3.3).

Phụ trách: B.

Quy tắc bắt buộc: codebook **chỉ học từ train**; validation và test chỉ được mã hóa bằng
codebook đã học. Mỗi góc nhìn (full, center) có một codebook riêng.
"""

from __future__ import annotations

import numpy as np

from .._todo import todo

N_SAMPLES = 200_000
BATCH_SIZE = 8192
N_INIT = 3


def sample_descriptors(train_desc: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Lấy mẫu descriptor để học codebook.

    Cách lấy: chọn ngẫu nhiên ``n_samples`` chỉ số ảnh (có lặp lại), rồi với mỗi ảnh chọn ngẫu
    nhiên một block. Dùng ``np.random.default_rng(seed)``.

    Args:
        train_desc: descriptor của **tập train**, shape ``(N, n_blocks, 36)`` (có thể là memmap
            float16).

    Returns:
        ``float32`` shape ``(n_samples, 36)``. Cùng seed ⇒ cùng kết quả.
    """
    todo("B", "sample_descriptors")


def fit_codebook(samples: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Học codebook bằng ``MiniBatchKMeans(n_clusters=k, batch_size=8192, n_init=3,
    random_state=seed)``.

    Returns:
        Tâm cụm ``float32`` shape ``(k, 36)``.
    """
    todo("B", "fit_codebook")


def encode_counts(desc: np.ndarray, codebook: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Gán cứng mỗi descriptor vào từ gần nhất (Euclid) rồi đếm histogram, công thức (8).

    Xử lý theo lô ``chunk`` ảnh để không hết RAM.

    Args:
        desc: shape ``(N, n_blocks, 36)``.
        codebook: shape ``(K, 36)``.

    Returns:
        ``int32`` shape ``(N, K)``; mỗi hàng có tổng bằng ``n_blocks``.
    """
    todo("B", "encode_counts")


def hellinger_l2(counts: np.ndarray) -> np.ndarray:
    """Chuẩn hóa Hellinger + L2 theo từng hàng, công thức (15).

    ``p = n / Σn``, ``b = √p / ‖√p‖₂``.

    Returns:
        ``float32`` cùng shape với ``counts``; mỗi hàng có chuẩn L2 bằng 1, mọi phần tử ≥ 0.
    """
    todo("B", "hellinger_l2")
