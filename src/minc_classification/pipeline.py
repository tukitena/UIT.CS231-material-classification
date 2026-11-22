"""Luồng chạy một thí nghiệm (báo cáo mục IV.1.3). Sơ đồ: ``doc/pipeline.md``.

Phụ trách: A (điều phối). ``build_representation`` chỉ ghép các hàm của B nên đã viết sẵn.

    extract  ─►  codebook (train)  ─►  encode BoW  ─►  build_representation  ─►  select (val)
                                                                                  │
                                                       evaluate test đúng một lần ◄┘
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._todo import todo
from .config import Paths, load_config, paths, validate
from .data import Split
from .features import fusion
from .features.views import View


@dataclass
class SplitFeatures:
    """Đặc trưng thô đã cache của một split. Mảng HOG là memmap float16."""

    split: Split
    labels: np.ndarray  # (N,) int64
    hog: dict[View, np.ndarray]  # full: (N, 225, 36), center: (N, 441, 36)
    hsv: dict[View, np.ndarray]  # mỗi view: (N, 32)
    signature: str  # cache signature dùng để ghi metadata


@dataclass
class Codebook:
    view: View
    k: int
    centers: np.ndarray  # (k, 36)
    hash: str  # repro.array_hash(centers)
    fit_seconds: float  # 0 nếu lấy từ cache


# --------------------------------------------------------------------------- bước 1: trích xuất


def extract(cfg: dict[str, Any], p: Paths, splits: tuple[Split, ...]) -> None:
    """Trích và cache đặc trưng thô cho từng split.

    Với mỗi ảnh (``data.iter_images``): ``extract_views`` → ``to_gray`` → ``hog_blocks`` cho cả
    hai view; ``hsv_hist`` cho cả hai view. Ghi bằng ``with cache.writing(...)``:

    - ``hog-full-<split>``   (N, 225, 36) float16
    - ``hog-center-<split>`` (N, 441, 36) float16
    - ``hsv-full-<split>``, ``hsv-center-<split>`` (N, 32) float32
    - ``labels-<split>``     (N,) int64

    Signature = ``cache.signature(cfg["features"], code_hash(views, hog, hsv), revision)``.
    File đã tồn tại với đúng signature thì bỏ qua.
    """
    todo("A", "pipeline.extract")


def load_features(cfg: dict[str, Any], p: Paths, split: Split) -> SplitFeatures:
    """Mở cache của ``split`` (memmap). Thiếu file → lỗi gợi ý chạy ``uv run minc extract``."""
    todo("A", "pipeline.load_features")


# --------------------------------------------------------------------------- bước 2: codebook


def get_codebook(
    cfg: dict[str, Any], p: Paths, train: SplitFeatures, view: View, k: int
) -> Codebook:
    """Codebook của ``view`` với ``k`` từ, cache theo (view, k, seed, signature của descriptor).

    Chưa có cache: ``bow.sample_descriptors(train.hog[view], n_samples, seed)`` →
    ``bow.fit_codebook`` → lưu ``codebook-<view>-k<k>-<sig>.npy``. Chỉ nhận ``train``.
    """
    todo("A", "pipeline.get_codebook")


def encode_split(feats: SplitFeatures, codebooks: dict[View, Codebook]) -> dict[View, np.ndarray]:
    """``hellinger_l2(encode_counts(feats.hog[view], cb.centers))`` cho từng view → (N, K)."""
    todo("A", "pipeline.encode_split")


# --------------------------------------------------------------------------- bước 3: biểu diễn


def build_representation(
    variant: str,
    hsv: dict[View, np.ndarray],
    bow: dict[View, np.ndarray] | None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Vector đặc trưng cuối cho từng biến thể (Bảng 8 của báo cáo).

    | variant         | Thí nghiệm | Công thức                                   | Số chiều |
    |-----------------|------------|---------------------------------------------|----------|
    | hsv             | E1         | c = L2([L2(c_full); L2(c_center)])          | 64       |
    | bow_full        | E2         | L2(b_full)                                  | K        |
    | bow_center      | E3         | L2(b_center)                                | K        |
    | bow_multi       | E4         | h = L2([L2(b_full); L2(b_center)])          | 2K       |
    | hsv_bow_multi   | E5–E7      | z = [√α·h ; √(1-α)·c]                       | 2K + 64  |
    """
    if variant == "hsv":
        return fusion.combine_views(hsv["full"], hsv["center"])
    if bow is None:
        raise ValueError(f"Biến thể {variant!r} cần đặc trưng BoW")
    if variant == "bow_full":
        return fusion.l2_normalize(bow["full"])
    if variant == "bow_center":
        return fusion.l2_normalize(bow["center"])
    h_bow = fusion.combine_views(bow["full"], bow["center"])
    if variant == "bow_multi":
        return h_bow
    if variant == "hsv_bow_multi":
        c = fusion.combine_views(hsv["full"], hsv["center"])
        return fusion.fuse(h_bow, c, alpha)
    raise ValueError(f"Biến thể lạ: {variant!r}")


# --------------------------------------------------------------------------- bước 4–5: chạy


def run_experiment(cfg: dict[str, Any], p: Paths, allow_retest: bool = False) -> Path:
    """Chạy một cấu hình E1–E7, trả thư mục run ``<RUNS_ROOT>/<experiment.name>``.

    1. ``load_features`` cho train, validation; codebook từ train với ``bow.k`` (bỏ qua nếu
       variant là ``hsv``); ``encode_split``.
    2. Với từng α trong ``alpha_grid`` (thứ tự config): ``build_representation`` →
       ``models.select_on_validation``. Chọn (α, tham số) bằng ``models.pick_best`` trên toàn bộ
       điểm theo thứ tự (α ngoài, tham số trong) ⇒ hòa giữ ứng viên đứng trước.
       Ghi ``selection.json``.
    3. **Test đúng một lần**: nếu đã có ``metrics_test.json`` và không có ``allow_retest`` → báo
       lỗi. Ngược lại mới ``load_features(test)``, dự đoán, ``evaluation.evaluate``, ghi
       ``metrics_test.json``, ``confusion_test.png``; ``allow_retest`` được ghi vào metadata.
    4. ``repro.write_metadata`` với revision, cache signature, K, hash codebook, α, tham số đã
       chọn, thời gian từng bước.
    """
    todo("A", "pipeline.run_experiment")


def run_survey(cfg: dict[str, Any], p: Paths) -> Path:
    """E8: với mỗi K trong ``survey.k_grid``, học hai codebook, mã hóa train + validation, chạy
    từng biến thể trong ``survey.variants`` với lưới C. **Không đụng tới test.**

    Chọn ``K*`` theo Macro-F1 validation của ``survey.select_k_by`` (hòa giữ K nhỏ hơn). Ghi
    ``survey.json`` (đủ bảng K × biến thể, kèm thời gian codebook / mã hóa / fit) và in K* để
    nhóm ghi vào ``configs/_base.yaml``.
    """
    todo("A", "pipeline.run_survey")


def run(config_path: str | Path, allow_retest: bool = False) -> Path:
    cfg = load_config(config_path)
    validate(cfg)
    p = paths()
    if "survey" in cfg:
        return run_survey(cfg, p)
    return run_experiment(cfg, p, allow_retest=allow_retest)
