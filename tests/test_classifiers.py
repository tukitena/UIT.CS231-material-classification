"""Hợp đồng của C: models.classifiers."""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from minc_classification.config import REPO_ROOT, load_config
from minc_classification.models import build_classifier, param_grid, pick_best, select_on_validation

CLF = load_config(REPO_ROOT / "configs" / "_base.yaml")["classifier"]
KNN = {**CLF, "type": "knn"}


def test_param_grid_keeps_config_order():
    assert param_grid(CLF) == [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    assert param_grid(KNN) == [{"n_neighbors": k} for k in (3, 5, 7, 9)]


def test_build_classifier_uses_report_settings():
    svm = build_classifier(CLF, {"C": 1.0}, seed=42)
    assert isinstance(svm, LinearSVC)
    assert (svm.C, svm.penalty, svm.loss, svm.max_iter, svm.tol, svm.random_state) == (
        1.0, "l2", "squared_hinge", 5000, 1e-4, 42,
    )  # fmt: skip
    knn = build_classifier(KNN, {"n_neighbors": 5}, seed=42)
    assert isinstance(knn, KNeighborsClassifier)
    assert (knn.n_neighbors, knn.weights, knn.metric) == (5, "uniform", "euclidean")


@pytest.mark.parametrize(
    ("scores", "best"), [([0.1, 0.3, 0.2], 1), ([0.3, 0.3, 0.1], 0), ([0.2, 0.5, 0.5], 1)]
)
def test_pick_best_ties_keep_earliest(scores, best):
    assert pick_best(scores) == best


def test_select_on_validation_returns_fitted_best():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] > 0).astype(int)
    sel = select_on_validation(X[:80], y[:80], X[80:], y[80:], CLF, seed=42)
    assert sel.best_params in param_grid(CLF)
    assert [p for p, _ in sel.scores] == param_grid(CLF)
    assert sel.best_score == max(s for _, s in sel.scores)
    assert sel.model.predict(X[80:]).shape == (40,)
