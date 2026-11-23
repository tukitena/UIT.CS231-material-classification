"""Phần của A: data.minc, kiểm tra trên một DatasetDict nhỏ dựng tại chỗ."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features
from datasets import Image as HFImage
from PIL import Image

from minc_classification.data import minc

REVISION = "rev-test"
COUNTS = {"train": 6, "validation": 3, "test": 3}


def make_dict(counts: dict[str, int] = COUNTS, size: int = 40) -> DatasetDict:
    rng = np.random.default_rng(0)
    features = Features({"image": HFImage(), "label": ClassLabel(names=list(minc.CLASS_NAMES))})
    out = {}
    for split, n in counts.items():
        images = []
        for i in range(n):
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            # Trộn các mode ảnh để kiểm tra chuyển về RGB
            images.append(img.convert(("RGB", "L", "RGBA")[i % 3]))
        out[split] = Dataset.from_dict(
            {"image": images, "label": [i % minc.N_CLASSES for i in range(n)]}, features=features
        )
    return DatasetDict(out)


def write_dataset(root: Path, counts: dict[str, int] = COUNTS, size: int = 40) -> Path:
    make_dict(counts, size).save_to_disk(str(root))
    source = {"dataset_id": "fake", "revision": REVISION, "splits": counts}
    (root / "source.json").write_text(json.dumps(source), encoding="utf-8")
    return root


def dataset_cfg(counts: dict[str, int] = COUNTS) -> dict:
    return {
        "id": "fake",
        "revision": REVISION,
        "splits": list(counts),
        "expected_counts": dict(counts),
        "class_names": list(minc.CLASS_NAMES),
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "ds")


def test_check_dataset_ok(root):
    minc.check_dataset(root, dataset_cfg())


def test_check_dataset_rejects_other_revision(root):
    cfg = {**dataset_cfg(), "revision": "khac"}
    with pytest.raises(ValueError, match="Revision"):
        minc.check_dataset(root, cfg)


def test_check_dataset_rejects_wrong_counts(root):
    cfg = dataset_cfg({**COUNTS, "test": 4})
    with pytest.raises(ValueError, match="test"):
        minc.check_dataset(root, cfg)


def test_check_dataset_requires_source_json(root):
    (root / "source.json").unlink()
    with pytest.raises(ValueError, match="source.json"):
        minc.check_dataset(root, dataset_cfg())


def test_missing_dataset_gives_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="minc download"):
        minc.split_size(tmp_path / "khong-co", "train")


def test_split_size_and_labels(root):
    assert minc.split_size(root, "validation") == 3
    labels = minc.load_labels(root, "train")
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, np.arange(6))


def test_iter_images_rgb_in_order(root):
    imgs = list(minc.iter_images(root, "train"))
    assert len(imgs) == 6
    for img in imgs:
        assert img.shape == (40, 40, 3) and img.dtype == np.uint8
    # Ảnh mode L sau khi chuyển RGB có ba kênh bằng nhau
    np.testing.assert_array_equal(imgs[1][..., 0], imgs[1][..., 2])


def test_iter_images_range(root):
    all_imgs = list(minc.iter_images(root, "train"))
    part = list(minc.iter_images(root, "train", 2, 5))
    assert len(part) == 3
    for a, b in zip(part, all_imgs[2:5], strict=True):
        np.testing.assert_array_equal(a, b)


def test_unknown_split_rejected(root):
    with pytest.raises(ValueError):
        minc.split_size(root, "dev")


def test_download_refuses_existing_without_force(root):
    with pytest.raises(FileExistsError):
        minc.download(dataset_cfg(), root)


def test_download_saves_and_writes_source(tmp_path, monkeypatch):
    calls = []

    def fake_load_dataset(name, revision):
        calls.append((name, revision))
        return make_dict()

    monkeypatch.setattr(minc, "load_dataset", fake_load_dataset)
    target = tmp_path / "new"
    minc.download(dataset_cfg(), target)
    assert calls == [("fake", REVISION)]
    minc.check_dataset(target, dataset_cfg())
    assert json.loads((target / "source.json").read_text())["splits"] == COUNTS
