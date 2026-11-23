"""MINC-2500 Split 1 từ Hugging Face (`mcimpoi/minc-2500_split_1`, revision ghim trong config).

Phụ trách: A.

Nguồn có hai cột ``image`` (PIL) và ``label`` (ClassLabel 23 lớp). Bản local là
``DatasetDict.save_to_disk()`` kèm ``source.json`` (id, revision, số mẫu lúc tải). Mẫu được định
danh bằng ``split:index`` theo thứ tự cố định của bản Arrow. Không chia lại dữ liệu.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from PIL import Image

Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "test")

CLASS_NAMES: tuple[str, ...] = (
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass", "hair",
    "leather", "metal", "mirror", "other", "painted", "paper", "plastic",
    "polishedstone", "skin", "sky", "stone", "tile", "wallpaper", "water", "wood",
)  # fmt: skip
N_CLASSES = len(CLASS_NAMES)  # 23

IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"
SOURCE_FILE = "source.json"


def download(dataset_cfg: dict[str, Any], root: Path, force: bool = False) -> None:
    """Tải ``dataset_cfg['id']`` đúng ``dataset_cfg['revision']`` rồi ``save_to_disk(root)``.

    - Nếu ``root`` đã có dữ liệu và ``force`` là False: báo lỗi, không tải lại.
    - Kiểm tra schema và số mẫu trước khi lưu; ghi ``root/source.json`` (id, revision, số mẫu).
    """
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        if not force:
            raise FileExistsError(f"Đã có dữ liệu tại {root}. Dùng --force để xoá và tải lại.")
        shutil.rmtree(root)
    _open.cache_clear()

    print(f"Tải {dataset_cfg['id']} @ {dataset_cfg['revision'][:10]} ...")
    ds = load_dataset(dataset_cfg["id"], revision=dataset_cfg["revision"])
    _check_dict(ds, dataset_cfg)

    root.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(root))
    source = {
        "dataset_id": dataset_cfg["id"],
        "revision": dataset_cfg["revision"],
        "splits": {s: len(ds[s]) for s in dataset_cfg["splits"]},
    }
    (root / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Đã lưu tại {root}")


def check_dataset(root: Path, dataset_cfg: dict[str, Any]) -> None:
    """Kiểm tra bản local khớp config: đủ 3 split, số mẫu bằng ``expected_counts``, đúng thứ tự
    ``class_names``, và ``source.json`` có cùng revision. Sai bất kỳ điểm nào → ``ValueError``."""
    source_path = Path(root) / SOURCE_FILE
    if not source_path.is_file():
        raise ValueError(f"Thiếu {source_path}; không xác định được revision của bản local.")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("revision") != dataset_cfg["revision"]:
        raise ValueError(
            f"Revision bản local ({source.get('revision')}) khác config "
            f"({dataset_cfg['revision']}). Chạy lại: uv run minc download --force"
        )
    _check_dict(_open(str(Path(root).resolve())), dataset_cfg)


def split_size(root: Path, split: Split) -> int:
    """Số ảnh của split: train 48.875, validation 2.875, test 5.750."""
    return len(_split(root, split))


def load_labels(root: Path, split: Split) -> np.ndarray:
    """Nhãn của split, shape ``(N,)``, dtype ``int64``, giá trị 0..22 theo ``CLASS_NAMES``."""
    ds = _split(root, split).select_columns([LABEL_COLUMN]).with_format("numpy")
    return np.asarray(ds[LABEL_COLUMN], dtype=np.int64)


def iter_images(
    root: Path, split: Split, start: int = 0, stop: int | None = None
) -> Iterator[np.ndarray]:
    """Lần lượt trả ảnh RGB ``uint8`` shape ``(H, W, 3)`` theo đúng thứ tự index của split.

    ``start``/``stop`` giới hạn đoạn index ``[start, stop)`` — dùng khi trích xuất song song.
    Ảnh xám, RGBA hay palette đều được chuyển về RGB.
    """
    ds = _split(root, split).select_columns([IMAGE_COLUMN])
    stop = len(ds) if stop is None else stop
    for i in range(start, stop):
        yield to_rgb(ds[i][IMAGE_COLUMN])


def to_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------- nội bộ


@lru_cache(maxsize=4)
def _open(root: str) -> DatasetDict:
    path = Path(root)
    if not (path / "dataset_dict.json").is_file():
        raise FileNotFoundError(
            f"Không thấy dataset tại {path}. Kiểm tra DATASET_ROOT trong .env "
            "hoặc chạy: uv run minc download"
        )
    return load_from_disk(str(path), keep_in_memory=False)


def _split(root: Path, split: Split):
    if split not in SPLITS:
        raise ValueError(f"split phải thuộc {SPLITS}, nhận {split!r}")
    return _open(str(Path(root).resolve()))[split]


def _check_dict(ds: DatasetDict, dataset_cfg: dict[str, Any]) -> None:
    expected_splits = list(dataset_cfg["splits"])
    if sorted(ds.keys()) != sorted(expected_splits):
        raise ValueError(f"Split không khớp: {sorted(ds.keys())} ≠ {sorted(expected_splits)}")
    for split in expected_splits:
        features = ds[split].features
        for col in (IMAGE_COLUMN, LABEL_COLUMN):
            if col not in features:
                raise ValueError(f"Split '{split}' thiếu cột '{col}'")
        names = list(getattr(features[LABEL_COLUMN], "names", []) or [])
        if names != list(dataset_cfg["class_names"]) or names != list(CLASS_NAMES):
            raise ValueError(f"Thứ tự nhãn của split '{split}' khác config / CLASS_NAMES")
        expected = dataset_cfg["expected_counts"][split]
        if len(ds[split]) != expected:
            raise ValueError(f"Split '{split}' có {len(ds[split])} mẫu, config cần {expected}")
