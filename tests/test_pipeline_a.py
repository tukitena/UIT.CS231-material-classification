"""Phần điều phối của A trong pipeline: cache, codebook, chọn tham số, khoá test một lần, E8.

Hàm của B (features) và C (models, evaluation) được thay bằng bản giả đơn giản, để test này
kiểm tra riêng phần của A và chạy được cả khi B, C chưa viết xong.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from minc_classification import pipeline
from minc_classification.config import REPO_ROOT, Paths, deep_merge, load_config
from minc_classification.features import bow, fusion, hog, hsv
from minc_classification.models import classifiers

from .test_data import write_dataset

COUNTS = {"train": 12, "validation": 6, "test": 6}
SIZE = 64  # ảnh 64 × 64 → vùng tâm 32 × 32 → 9 block; toàn ảnh 128 → 225 block


# --------------------------------------------------------------------------- bản giả của B, C


def _l2(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _fake_hog(gray):
    nb = hog.n_blocks(gray.shape[0])
    base = np.resize(gray.astype(np.float32).ravel() / 255.0, nb * 36)
    return base.reshape(nb, 36)


def _fake_hsv(rgb):
    return np.resize(rgb.mean(axis=(0, 1)).astype(np.float32) / 255.0, 32)


def _fake_sample(train_desc, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, train_desc.shape[0], n)
    blk = rng.integers(0, train_desc.shape[1], n)
    return np.asarray(train_desc[idx, blk], dtype=np.float32)


def _fake_encode(desc, codebook, chunk=2048):
    d = np.asarray(desc, dtype=np.float32)
    dist = ((d[:, :, None, :] - codebook[None, None]) ** 2).sum(-1)
    words = dist.argmin(-1)
    return np.stack([np.bincount(w, minlength=len(codebook)) for w in words]).astype(np.int32)


def _fake_hellinger(counts):
    p = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    return _l2(np.sqrt(p)).astype(np.float32)


def _fake_select(X_tr, y_tr, X_val, y_val, clf_cfg, seed):
    grid = classifiers.param_grid(clf_cfg)
    model = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    scores = [(prm, 0.5) for prm in grid]  # mọi ứng viên hòa điểm
    return classifiers.Selection(grid[0], 0.5, model, scores, fit_seconds=0.01)


def _fake_evaluate(y_true, y_pred, n_classes=23):
    acc = float(np.mean(y_true == y_pred))
    conf = np.zeros((n_classes, n_classes), int)
    np.add.at(conf, (y_true, y_pred), 1)
    return SimpleNamespace(
        accuracy=acc,
        macro_f1=acc,
        confusion=conf,
        to_dict=lambda: {"accuracy": acc, "macro_f1": acc},
    )


@pytest.fixture
def fake_bc(monkeypatch):
    m = monkeypatch.setattr
    m(hog, "hog_blocks", _fake_hog)
    m(hsv, "hsv_hist", _fake_hsv)
    m(bow, "sample_descriptors", _fake_sample)
    m(bow, "fit_codebook", lambda s, k, seed: s[:k].copy())
    m(bow, "encode_counts", _fake_encode)
    m(bow, "hellinger_l2", _fake_hellinger)
    m(fusion, "l2_normalize", _l2)
    m(fusion, "combine_views", lambda a, b: _l2(np.hstack([_l2(a), _l2(b)])))
    m(fusion, "fuse", lambda h, c, a: np.hstack([np.sqrt(a) * h, np.sqrt(1 - a) * c]))
    m(
        classifiers,
        "param_grid",
        lambda c: (
            [{"C": x} for x in c["linear_svm"]["C_grid"]]
            if c["type"] == "linear_svm"
            else [{"n_neighbors": x} for x in c["knn"]["k_grid"]]
        ),
    )
    m(classifiers, "pick_best", lambda s: int(np.argmax(s)))
    m(classifiers, "select_on_validation", _fake_select)
    m(pipeline.metrics, "evaluate", _fake_evaluate)
    m(pipeline.metrics, "save_confusion_png", lambda c, n, path: Path(path).write_bytes(b"png"))


# --------------------------------------------------------------------------- dựng môi trường


@pytest.fixture
def env(tmp_path, fake_bc):
    p = Paths(dataset=tmp_path / "ds", cache=tmp_path / "cache", runs=tmp_path / "runs")
    write_dataset(p.dataset, COUNTS, SIZE)
    return p


def cfg_for(name: str, **override) -> dict:
    cfg = load_config(REPO_ROOT / "configs" / f"{name}.yaml")
    cfg = deep_merge(cfg, {"bow": {"k": 4, "n_samples": 50}})
    return deep_merge(cfg, override)


def extracted(p: Paths, splits=("train", "validation", "test")) -> Paths:
    pipeline.extract(cfg_for("e1_hsv"), p, splits, chunk=5)
    return p


# --------------------------------------------------------------------------- extract / load


def test_extract_writes_expected_arrays(env):
    extracted(env)
    f = pipeline.load_features(cfg_for("e1_hsv"), env, "train")
    assert f.hog["full"].shape == (12, 225, 36) and f.hog["full"].dtype == np.float16
    assert f.hog["center"].shape == (12, 9, 36)
    assert f.hsv["center"].shape == (12, 32) and f.hsv["center"].dtype == np.float32
    np.testing.assert_array_equal(f.labels, np.arange(12))
    assert not list(env.cache.glob("*.partial.npy"))


def test_extract_chunks_match_one_pass(env, tmp_path):
    extracted(env)
    other = Paths(env.dataset, tmp_path / "cache2", env.runs)
    pipeline.extract(cfg_for("e1_hsv"), other, ("train",), chunk=1000)
    a = pipeline.load_features(cfg_for("e1_hsv"), env, "train")
    b = pipeline.load_features(cfg_for("e1_hsv"), other, "train")
    np.testing.assert_array_equal(a.hog["center"], b.hog["center"])


def test_extract_skips_existing_cache(env, monkeypatch):
    extracted(env)
    monkeypatch.setattr(hog, "hog_blocks", lambda g: pytest.fail("không được trích lại"))
    extracted(env)


def test_changing_feature_params_invalidates_cache(env):
    extracted(env)
    cfg = cfg_for("e1_hsv", features={"hsv": {"bins": [8, 8, 8]}})
    with pytest.raises(FileNotFoundError, match="minc extract"):
        pipeline.load_features(cfg, env, "train")


# --------------------------------------------------------------------------- codebook


def test_codebook_only_from_train(env):
    extracted(env)
    val = pipeline.load_features(cfg_for("e1_hsv"), env, "validation")
    with pytest.raises(ValueError, match="train"):
        pipeline.get_codebook(cfg_for("e1_hsv"), env, val, "full", 4)


def test_codebook_is_cached_with_original_fit_time(env, monkeypatch):
    extracted(env)
    cfg = cfg_for("e4_bow_multi")
    train = pipeline.load_features(cfg, env, "train")
    first = pipeline.get_codebook(cfg, env, train, "center", 4)
    monkeypatch.setattr(bow, "fit_codebook", lambda *a: pytest.fail("không được học lại"))
    again = pipeline.get_codebook(cfg, env, train, "center", 4)
    assert (first.cached, again.cached) == (False, True)
    assert first.hash == again.hash and again.fit_seconds == first.fit_seconds
    assert first.centers.shape == (4, 36)


# --------------------------------------------------------------------------- E1–E7


@pytest.mark.parametrize(
    ("name", "dim"),
    [("e1_hsv", 64), ("e2_bow_full", 4), ("e4_bow_multi", 8), ("e5_hsv_bow_multi", 72)],
)
def test_run_experiment_outputs(env, name, dim):
    extracted(env)
    run_dir = pipeline.run_experiment(cfg_for(name), env)
    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert sel["dim"] == dim
    assert sel["best_params"] == {"C": 0.01}  # hòa điểm → giữ ứng viên đầu
    for f in ("metrics_test.json", "pred_test.npy", "confusion_test.png", "metadata.json"):
        assert (run_dir / f).exists(), f
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["retest"] is False and meta["seed"] == 42
    assert set(meta["codebooks"]) == set(pipeline.BOW_VIEWS[sel["variant"]])


def test_test_set_is_evaluated_only_once(env):
    extracted(env)
    cfg = cfg_for("e1_hsv")
    run_dir = pipeline.run_experiment(cfg, env)
    with pytest.raises(RuntimeError, match="allow-retest"):
        pipeline.run_experiment(cfg, env)
    pipeline.run_experiment(cfg, env, allow_retest=True)
    assert len(list(run_dir.glob("metrics_test.*.json"))) == 1  # bản cũ được giữ lại
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["retest"] is True


def test_e6_alpha_tie_keeps_first_alpha(env):
    extracted(env)
    run_dir = pipeline.run_experiment(cfg_for("e6_hsv_bow_multi_alpha"), env)
    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert sel["alpha"] == 0.25
    assert [c["alpha"] for c in sel["candidates"]] == [0.25] * 4 + [0.5] * 4 + [0.75] * 4


def test_e7_uses_knn_grid(env):
    extracted(env)
    run_dir = pipeline.run_experiment(cfg_for("e7_hsv_bow_multi_knn"), env)
    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert sel["classifier"] == "knn" and sel["best_params"] == {"n_neighbors": 3}


# --------------------------------------------------------------------------- E8


def test_survey_never_touches_test_and_ties_keep_smaller_k(env):
    extracted(env, splits=("train", "validation"))  # không có cache test
    cfg = cfg_for("e8_codebook_k_survey", survey={"k_grid": [8, 4]})
    run_dir = pipeline.run_survey(cfg, env)
    survey = json.loads((run_dir / "survey.json").read_text(encoding="utf-8"))
    assert survey["k_grid"] == [4, 8]
    assert survey["k_star"] == 4
    assert [(r["k"], r["variant"]) for r in survey["rows"]] == [
        (4, "bow_multi"),
        (4, "hsv_bow_multi"),
        (8, "bow_multi"),
        (8, "hsv_bow_multi"),
    ]
    assert not (run_dir / "metrics_test.json").exists()
