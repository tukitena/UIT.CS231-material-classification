"""Hợp đồng của C: evaluation.metrics."""

import json

import numpy as np
import pytest

from minc_classification.evaluation import evaluate


def test_perfect_prediction():
    y = np.arange(23).repeat(2)
    m = evaluate(y, y)
    assert m.accuracy == 1.0 and m.macro_f1 == 1.0
    assert m.confusion.shape == (23, 23)
    np.testing.assert_array_equal(np.diag(m.confusion), 2)


def test_macro_f1_counts_all_23_classes_with_zero_division():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    m = evaluate(y_true, y_pred)
    assert m.f1.shape == (23,)
    assert m.f1[1] == 0.0
    assert m.macro_f1 == pytest.approx(m.f1.mean())
    assert m.confusion[1, 0] == 2  # hàng = nhãn thật, cột = dự đoán


def test_to_dict_is_json_serializable():
    y = np.arange(23)
    json.dumps(evaluate(y, y).to_dict())
