"""Hợp đồng của A: features.views."""

import numpy as np

from minc_classification.features import views

RNG = np.random.default_rng(0)
IMG = RNG.integers(0, 256, (362, 362, 3), dtype=np.uint8)


def test_full_view_is_128_rgb():
    out = views.full_view(IMG)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8


def test_center_view_is_181_native_crop():
    out = views.center_view(IMG)
    assert out.shape == (181, 181, 3)
    np.testing.assert_array_equal(out, IMG[90:271, 90:271])


def test_center_view_non_square():
    img = RNG.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    out = views.center_view(img)
    assert out.shape == (150, 150, 3)
    np.testing.assert_array_equal(out, img[75:225, 125:275])


def test_to_gray():
    out = views.to_gray(IMG)
    assert out.shape == (362, 362) and out.dtype == np.uint8
