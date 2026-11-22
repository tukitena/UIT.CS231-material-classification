import numpy as np
import pytest

from minc_classification import cache, repro
from minc_classification.features import hog, hsv


def test_signature_changes_with_params_code_and_revision():
    base = cache.signature({"bins": [16, 8, 8]}, "code1", "rev1")
    assert base == cache.signature({"bins": [16, 8, 8]}, "code1", "rev1")
    assert base != cache.signature({"bins": [8, 8, 8]}, "code1", "rev1")
    assert base != cache.signature({"bins": [16, 8, 8]}, "code2", "rev1")
    assert base != cache.signature({"bins": [16, 8, 8]}, "code1", "rev2")


def test_partial_file_is_invisible_until_block_exits(tmp_path):
    path = cache.array_path(tmp_path, "hog-full-train", "abc")
    with cache.writing(path, (3, 2), np.float16) as arr:
        arr[:] = 1
        assert not path.exists()
    assert path.exists()
    np.testing.assert_array_equal(cache.load(path, mmap=False), np.ones((3, 2), np.float16))


def test_interrupted_write_leaves_no_cache(tmp_path):
    path = cache.array_path(tmp_path, "hsv-full-train", "abc")
    with pytest.raises(KeyboardInterrupt), cache.writing(path, (3, 2), np.float32) as arr:
        arr[0] = 1
        raise KeyboardInterrupt
    assert not path.exists()


def test_code_hash_is_stable_and_order_independent():
    assert repro.code_hash(hog, hsv) == repro.code_hash(hsv, hog)
    assert repro.code_hash(hog) != repro.code_hash(hsv)


def test_write_metadata(tmp_path):
    cfg = {"experiment": {"seed": 42}, "dataset": {"revision": "abc"}}
    path = repro.write_metadata(tmp_path / "run", cfg, k=512)
    text = path.read_text(encoding="utf-8")
    assert '"seed": 42' in text and '"k": 512' in text and "scikit-learn" in text
