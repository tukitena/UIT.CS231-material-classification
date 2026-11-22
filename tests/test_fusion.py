"""Hợp đồng của B: features.fusion."""

import numpy as np
import pytest

from minc_classification.features import fusion

RNG = np.random.default_rng(0)


def test_l2_normalize_rows_and_keeps_zero_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    np.testing.assert_allclose(fusion.l2_normalize(x), [[0.6, 0.8], [0, 0]])


def test_combine_views_gives_each_view_half_the_energy():
    a, b = RNG.random((5, 8)), RNG.random((5, 3)) * 100
    out = fusion.combine_views(a, b)
    assert out.shape == (5, 11)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1)
    np.testing.assert_allclose(np.sum(out[:, :8] ** 2, axis=1), 0.5)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_fuse_alpha_is_edge_share_of_squared_norm(alpha):
    k = 16
    h = fusion.l2_normalize(RNG.random((4, 2 * k)))
    c = fusion.l2_normalize(RNG.random((4, 64)))
    z = fusion.fuse(h, c, alpha)
    assert z.shape == (4, 2 * k + 64)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.sum(z[:, : 2 * k] ** 2, axis=1), alpha, atol=1e-6)


def test_fuse_rejects_alpha_out_of_range():
    with pytest.raises(ValueError):
        fusion.fuse(np.ones((1, 2)), np.ones((1, 2)), 1.5)
