"""Hợp đồng của B: features.hsv."""

import numpy as np
import pytest

from minc_classification.features import hsv

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("side", [128, 181])
def test_hsv_hist_is_32_dims_l1_per_channel(side):
    img = RNG.integers(0, 256, (side, side, 3), dtype=np.uint8)
    out = hsv.hsv_hist(img)
    assert out.shape == (32,) and out.dtype == np.float32
    for part in (out[:16], out[16:24], out[24:]):
        assert part.sum() == pytest.approx(1.0, abs=1e-5)


def test_pure_red_falls_in_expected_bins():
    img = np.zeros((128, 128, 3), np.uint8)
    img[..., 0] = 255
    out = hsv.hsv_hist(img)
    assert out[0] == pytest.approx(1.0)  # H = 0
    assert out[16 + 7] == pytest.approx(1.0)  # S = 255 → bin cuối
    assert out[24 + 7] == pytest.approx(1.0)  # V = 255 → bin cuối
