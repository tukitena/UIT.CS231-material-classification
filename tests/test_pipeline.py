"""Ghép biểu diễn (cần fusion của B) và số chiều theo Bảng 8 của báo cáo."""

import numpy as np
import pytest

from minc_classification.pipeline import build_representation

RNG = np.random.default_rng(0)
N, K = 6, 32
HSV = {"full": RNG.random((N, 32)), "center": RNG.random((N, 32))}
BOW = {"full": RNG.random((N, K)), "center": RNG.random((N, K))}


@pytest.mark.parametrize(
    ("variant", "dim"),
    [("hsv", 64), ("bow_full", K), ("bow_center", K), ("bow_multi", 2 * K),
     ("hsv_bow_multi", 2 * K + 64)],
)  # fmt: skip
def test_representation_dims_and_unit_norm(variant, dim):
    x = build_representation(variant, HSV, None if variant == "hsv" else BOW, alpha=0.5)
    assert x.shape == (N, dim)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)


def test_bow_variant_without_bow_fails():
    with pytest.raises(ValueError):
        build_representation("bow_multi", HSV, None)
