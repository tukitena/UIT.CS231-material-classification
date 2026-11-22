from pathlib import Path

import pytest

from minc_classification.config import REPO_ROOT, VARIANTS, deep_merge, load_config, validate
from minc_classification.data import CLASS_NAMES

CONFIGS = sorted((REPO_ROOT / "configs").glob("e*.yaml"))


def test_deep_merge_overrides_leaves_and_keeps_siblings():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    out = deep_merge(base, {"a": {"y": [3]}})
    assert out == {"a": {"x": 1, "y": [3]}, "b": 1}
    assert base["a"]["y"] == [1, 2]  # không sửa input


def test_all_eight_experiments_present():
    assert [p.name[:2] for p in CONFIGS] == [f"e{i}" for i in range(1, 9)]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_inherits_base(path: Path):
    cfg = load_config(path)
    assert cfg["experiment"]["seed"] == 42
    assert cfg["experiment"]["name"] == path.stem
    assert cfg["dataset"]["class_names"] == list(CLASS_NAMES)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_valid_once_k_star_is_set(path: Path):
    cfg = load_config(path)
    cfg["bow"]["k"] = 512
    validate(cfg)
    if "survey" not in cfg:
        assert cfg["representation"]["variant"] in VARIANTS


def test_bow_experiments_require_k_star():
    cfg = load_config(REPO_ROOT / "configs" / "e5_hsv_bow_multi.yaml")
    assert cfg["bow"]["k"] is None
    with pytest.raises(ValueError, match="E8"):
        validate(cfg)


def test_hsv_and_survey_do_not_need_k_star():
    validate(load_config(REPO_ROOT / "configs" / "e1_hsv.yaml"))
    validate(load_config(REPO_ROOT / "configs" / "e8_codebook_k_survey.yaml"))


def test_e7_switches_to_knn_only():
    cfg = load_config(REPO_ROOT / "configs" / "e7_hsv_bow_multi_knn.yaml")
    assert cfg["classifier"]["type"] == "knn"
    assert cfg["classifier"]["knn"]["k_grid"] == [3, 5, 7, 9]
