"""Hợp đồng của B: features.bow."""

import numpy as np
import pytest

from minc_classification.features import bow

RNG = np.random.default_rng(0)
TRAIN = RNG.random((50, 225, 36)).astype(np.float16)


def test_sample_descriptors_shape_and_determinism():
    a = bow.sample_descriptors(TRAIN, 1000, seed=42)
    b = bow.sample_descriptors(TRAIN, 1000, seed=42)
    c = bow.sample_descriptors(TRAIN, 1000, seed=7)
    assert a.shape == (1000, 36) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_samples_come_from_train_rows():
    s = bow.sample_descriptors(TRAIN, 200, seed=42)
    pool = {row.tobytes() for row in TRAIN.reshape(-1, 36).astype(np.float32)}
    assert all(row.tobytes() in pool for row in s)


def test_fit_codebook_shape_and_determinism():
    s = bow.sample_descriptors(TRAIN, 2000, seed=42)
    a = bow.fit_codebook(s, k=16, seed=42)
    b = bow.fit_codebook(s, k=16, seed=42)
    assert a.shape == (16, 36)
    np.testing.assert_allclose(a, b)


def test_encode_counts_hard_assignment():
    codebook = np.eye(3, 36, dtype=np.float32)
    desc = np.zeros((2, 4, 36), np.float32)
    desc[0, :, 0] = 1  # ảnh 0: cả 4 block gần từ 0
    desc[1, :2, 1] = 1  # ảnh 1: 2 block gần từ 1, 2 block gần từ 2
    desc[1, 2:, 2] = 1
    out = bow.encode_counts(desc, codebook, chunk=1)
    np.testing.assert_array_equal(out, [[4, 0, 0], [0, 2, 2]])


def test_hellinger_l2_unit_norm_nonnegative():
    counts = np.array([[4, 0, 0], [1, 1, 2]], np.int32)
    out = bow.hellinger_l2(counts)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1, atol=1e-6)
    assert np.all(out >= 0)
    np.testing.assert_allclose(out[1], np.sqrt([0.25, 0.25, 0.5]), atol=1e-6)


def test_hellinger_kernel_equals_bhattacharyya():
    counts = RNG.integers(1, 20, (2, 64))
    b = bow.hellinger_l2(counts)
    p = counts / counts.sum(axis=1, keepdims=True)
    assert b[0] @ b[1] == pytest.approx(np.sqrt(p[0] * p[1]).sum(), abs=1e-5)
