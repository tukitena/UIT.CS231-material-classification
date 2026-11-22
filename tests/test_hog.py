"""Hợp đồng của B: features.hog."""

import numpy as np
import pytest

from minc_classification.features import hog

RNG = np.random.default_rng(0)


def test_n_blocks_matches_report():
    assert hog.n_blocks(128) == 225
    assert hog.n_blocks(181) == 441


@pytest.mark.parametrize("side", [128, 181])
def test_hog_blocks_shape_and_dtype(side):
    gray = RNG.integers(0, 256, (side, side), dtype=np.uint8)
    out = hog.hog_blocks(gray)
    assert out.shape == (hog.n_blocks(side), 36)
    assert out.dtype == np.float32


def test_hog_blocks_are_l2hys_normalized():
    gray = RNG.integers(0, 256, (128, 128), dtype=np.uint8)
    out = hog.hog_blocks(gray)
    assert np.all(out >= 0)
    assert np.all(np.linalg.norm(out, axis=1) <= 1 + 1e-3)
