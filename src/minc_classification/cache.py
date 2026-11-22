"""Cache mảng trên đĩa, khoá bằng signature = hash(tham số + hash mã trích xuất + revision).

Định dạng ``.npy``, đọc lại bằng memmap. Descriptor HOG lưu float16 (~2,8 GB cho ba split).
Tên file: ``<CACHE_ROOT>/<name>-<signature>.npy``, ví dụ ``hog-center-train-3f9a1c2b7d10.npy``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def signature(params: dict[str, Any], code: str, revision: str) -> str:
    payload = json.dumps(
        {"params": params, "code": code, "revision": revision}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def array_path(cache_dir: Path, name: str, sig: str) -> Path:
    return Path(cache_dir) / f"{name}-{sig}.npy"


@contextmanager
def writing(path: Path, shape: tuple[int, ...], dtype: Any) -> Iterator[np.memmap]:
    """Ghi dần từng ảnh vào memmap ``*.partial.npy``; chỉ khi khối ``with`` kết thúc không lỗi
    mới đổi tên thành ``path``. File dở dang (bị ngắt giữa chừng) không bao giờ thành cache.

        with cache.writing(path, (n, 441, 36), np.float16) as arr:
            for i, img in enumerate(images):
                arr[i] = ...
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial(path)
    arr = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)
    try:
        yield arr
        arr.flush()
    finally:
        # Windows không cho đổi tên/xoá file đang được map: phải đóng mmap tường minh.
        arr._mmap.close()
    partial.replace(path)


def load(path: Path, mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)


def _partial(path: Path) -> Path:
    return path.with_suffix(".partial.npy")
