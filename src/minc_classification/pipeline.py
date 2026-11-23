"""Luồng chạy một thí nghiệm (báo cáo mục IV.1.3). Sơ đồ: ``doc/pipeline.md``.

Phụ trách: A (điều phối). Các bước gọi hàm của B (``features``) và C (``models``,
``evaluation``) qua module, ví dụ ``hog.hog_blocks``, nên test có thể thay thế từng hàm.

    extract  ─►  codebook (train)  ─►  encode BoW  ─►  build_representation  ─►  select (val)
                                                                                  │
                                                       evaluate test đúng một lần ◄┘
"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from . import cache, repro
from .config import Paths, load_config, paths, validate
from .data import minc
from .data.minc import CLASS_NAMES, Split
from .evaluation import metrics
from .features import bow, fusion, hog, hsv, views
from .features.views import VIEWS, View
from .models import classifiers

FEATURE_MODULES = (views, hog, hsv)
"""Mã nguồn các module này tham gia cache signature của đặc trưng thô."""

BOW_VIEWS: dict[str, tuple[View, ...]] = {
    "hsv": (),
    "bow_full": ("full",),
    "bow_center": ("center",),
    "bow_multi": VIEWS,
    "hsv_bow_multi": VIEWS,
}
"""Góc nhìn cần codebook cho từng biến thể."""


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
    fit_seconds: float  # thời gian học lúc tạo (giữ nguyên khi lấy lại từ cache)
    cached: bool  # True nếu lấy từ cache, không học lại lần này


# --------------------------------------------------------------------------- bước 1: trích xuất


def feature_signature(cfg: dict[str, Any]) -> str:
    return cache.signature(
        cfg["features"], repro.code_hash(*FEATURE_MODULES), cfg["dataset"]["revision"]
    )


def _array_keys() -> list[str]:
    return [f"{kind}-{v}" for kind in ("hog", "hsv") for v in VIEWS]


def _image_features(rgb: np.ndarray) -> dict[str, np.ndarray]:
    vs = views.extract_views(rgb)
    out: dict[str, np.ndarray] = {}
    for v in VIEWS:
        out[f"hog-{v}"] = hog.hog_blocks(views.to_gray(vs[v]))
        out[f"hsv-{v}"] = hsv.hsv_hist(vs[v])
    return out


def _chunk_features(
    root: Path, split: Split, start: int, stop: int, dtypes: dict[str, np.dtype]
) -> dict[str, np.ndarray]:
    rows = [_image_features(img) for img in minc.iter_images(root, split, start, stop)]
    return {key: np.stack([r[key] for r in rows]).astype(dtypes[key]) for key in dtypes}


def extract(
    cfg: dict[str, Any],
    p: Paths,
    splits: tuple[Split, ...],
    n_jobs: int = 1,
    chunk: int = 256,
) -> None:
    """Trích và cache đặc trưng thô cho từng split.

    Với mỗi ảnh (``data.iter_images``): ``extract_views`` → ``to_gray`` → ``hog_blocks`` cho cả
    hai view; ``hsv_hist`` cho cả hai view. Ghi bằng ``with cache.writing(...)``:

    - ``hog-full-<split>``   (N, 225, 36) float16
    - ``hog-center-<split>`` (N, 441, 36) float16
    - ``hsv-full-<split>``, ``hsv-center-<split>`` (N, 32) float32
    - ``labels-<split>``     (N,) int64

    Signature = ``cache.signature(cfg["features"], code_hash(views, hog, hsv), revision)``.
    File đã tồn tại với đúng signature thì bỏ qua. ``n_jobs`` > 1 chia ảnh thành từng đoạn
    ``chunk`` ảnh và xử lý song song bằng joblib; thứ tự ghi luôn theo index.
    """
    sig = feature_signature(cfg)
    hog_dtype = np.dtype(cfg["features"]["hog"]["dtype"])
    dtypes = {
        key: (hog_dtype if key.startswith("hog") else np.dtype(np.float32)) for key in _array_keys()
    }

    for split in splits:
        targets = {
            key: cache.array_path(p.cache, f"{key}-{split}", sig)
            for key in [*_array_keys(), "labels"]
        }
        if all(path.exists() for path in targets.values()):
            print(f"[extract] {split}: đã có cache {sig}, bỏ qua")
            continue

        n = minc.split_size(p.dataset, split)
        labels = minc.load_labels(p.dataset, split)
        if labels.shape != (n,):
            raise ValueError(f"Số nhãn {labels.shape} khác số ảnh {n} của split {split}")

        # Shape từng mảng lấy từ ảnh đầu tiên (225/441 block với ảnh 362 × 362).
        first = _image_features(next(minc.iter_images(p.dataset, split, 0, 1)))
        shapes = {key: (n, *first[key].shape) for key in dtypes}
        bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

        t0 = time.perf_counter()
        with ExitStack() as stack:
            outs = {
                key: stack.enter_context(cache.writing(targets[key], shapes[key], dtypes[key]))
                for key in dtypes
            }
            results = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_chunk_features)(p.dataset, split, s, e, dtypes) for s, e in bounds
            )
            for (s, e), res in tqdm(
                zip(bounds, results, strict=True), total=len(bounds), desc=f"extract {split}"
            ):
                for key, arr in res.items():
                    outs[key][s:e] = arr
        with cache.writing(targets["labels"], labels.shape, np.int64) as out:
            out[:] = labels
        print(f"[extract] {split}: {n} ảnh, {time.perf_counter() - t0:.0f}s, cache {sig}")


def load_features(cfg: dict[str, Any], p: Paths, split: Split) -> SplitFeatures:
    """Mở cache của ``split`` (memmap). Thiếu file → lỗi gợi ý chạy ``uv run minc extract``."""
    sig = feature_signature(cfg)

    def get(key: str) -> np.ndarray:
        path = cache.array_path(p.cache, f"{key}-{split}", sig)
        if not path.exists():
            raise FileNotFoundError(
                f"Thiếu cache {path.name} (signature {sig}). "
                f"Chạy: uv run minc extract --split {split}"
            )
        return cache.load(path)

    return SplitFeatures(
        split=split,
        labels=np.asarray(get("labels")),
        hog={v: get(f"hog-{v}") for v in VIEWS},
        hsv={v: np.asarray(get(f"hsv-{v}")) for v in VIEWS},
        signature=sig,
    )


# --------------------------------------------------------------------------- bước 2: codebook


def get_codebook(
    cfg: dict[str, Any], p: Paths, train: SplitFeatures, view: View, k: int
) -> Codebook:
    """Codebook của ``view`` với ``k`` từ, cache theo (view, k, seed, signature của descriptor).

    Chưa có cache: ``bow.sample_descriptors(train.hog[view], n_samples, seed)`` →
    ``bow.fit_codebook`` → lưu ``codebook-<view>-k<k>-<sig>.npy`` (kèm ``.json`` ghi thời gian
    học). Chỉ nhận ``train``.
    """
    if train.split != "train":
        raise ValueError(f"Codebook chỉ được học từ train, nhận split {train.split!r}")
    bcfg, seed = cfg["bow"], cfg["experiment"]["seed"]
    if (bcfg["batch_size"], bcfg["n_init"]) != (bow.BATCH_SIZE, bow.N_INIT):
        raise ValueError(
            "bow.batch_size / bow.n_init trong config khác hằng số trong features/bow.py; "
            "sửa cả hai cho khớp."
        )

    params = {
        "view": view,
        "k": k,
        "seed": seed,
        "n_samples": bcfg["n_samples"],
        "batch_size": bcfg["batch_size"],
        "n_init": bcfg["n_init"],
        "features": train.signature,
    }
    sig = cache.signature(params, repro.code_hash(bow), cfg["dataset"]["revision"])
    path = cache.array_path(p.cache, f"codebook-{view}-k{k}", sig)
    info_path = path.with_suffix(".json")

    if path.exists() and info_path.exists():
        centers = np.asarray(cache.load(path, mmap=False))
        fit_seconds = json.loads(info_path.read_text(encoding="utf-8"))["fit_seconds"]
        cached = True
    else:
        t0 = time.perf_counter()
        samples = bow.sample_descriptors(train.hog[view], bcfg["n_samples"], seed)
        centers = np.asarray(bow.fit_codebook(samples, k, seed))
        fit_seconds = time.perf_counter() - t0
        if centers.shape != (k, samples.shape[1]):
            raise ValueError(f"fit_codebook trả shape {centers.shape}, cần ({k}, 36)")
        with cache.writing(path, centers.shape, centers.dtype) as out:
            out[:] = centers
        _write_json(info_path, {**params, "fit_seconds": fit_seconds})
        cached = False

    return Codebook(view, k, centers, repro.array_hash(centers), fit_seconds, cached)


def encode_split(feats: SplitFeatures, codebooks: dict[View, Codebook]) -> dict[View, np.ndarray]:
    """``hellinger_l2(encode_counts(feats.hog[view], cb.centers))`` cho từng view → (N, K)."""
    return {
        v: bow.hellinger_l2(bow.encode_counts(feats.hog[v], cb.centers))
        for v, cb in codebooks.items()
    }


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


def _alphas(variant: str, cfg: dict[str, Any]) -> list[float | None]:
    """α chỉ có nghĩa với ``hsv_bow_multi``; biến thể khác dùng một ứng viên ``None``."""
    return list(cfg["representation"]["alpha_grid"]) if variant == "hsv_bow_multi" else [None]


def _represent(variant: str, hsv_: dict, bow_: dict | None, alpha: float | None) -> np.ndarray:
    return build_representation(variant, hsv_, bow_, 0.5 if alpha is None else alpha)


@dataclass
class _Choice:
    alpha: float | None
    selection: classifiers.Selection
    candidates: list[dict[str, Any]]  # mọi (α, tham số, điểm) theo thứ tự lưới
    dim: int
    fit_seconds: float


def _select(
    variant: str,
    cfg: dict[str, Any],
    train: SplitFeatures,
    train_bow: dict | None,
    val: SplitFeatures,
    val_bow: dict | None,
) -> _Choice:
    """Chọn (α, tham số) theo Macro-F1 validation. Duyệt α ngoài, tham số trong.

    ``select_on_validation`` đã giữ ứng viên đứng trước trong cùng một α; ``pick_best`` trên điểm
    tốt nhất của từng α giữ α đứng trước — tương đương ``pick_best`` trên toàn lưới phẳng.
    """
    alphas = _alphas(variant, cfg)
    seed = cfg["experiment"]["seed"]
    sels, dim = [], 0
    for a in alphas:
        X_tr = _represent(variant, train.hsv, train_bow, a)
        X_val = _represent(variant, val.hsv, val_bow, a)
        dim = X_tr.shape[1]
        sels.append(
            classifiers.select_on_validation(
                X_tr, train.labels, X_val, val.labels, cfg["classifier"], seed
            )
        )
    best = classifiers.pick_best([s.best_score for s in sels])
    candidates = [
        {"alpha": a, "params": prm, "val_macro_f1": score}
        for a, s in zip(alphas, sels, strict=True)
        for prm, score in s.scores
    ]
    return _Choice(alphas[best], sels[best], candidates, dim, sum(s.fit_seconds for s in sels))


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
       lỗi (kiểm tra ngay từ đầu, trước khi tốn thời gian). Ngược lại mới
       ``load_features(test)``, dự đoán, ``evaluation.evaluate``, ghi ``metrics_test.json``,
       ``pred_test.npy``, ``confusion_test.png``. Kết quả test cũ (nếu có) được giữ lại dưới
       tên ``metrics_test.<thời điểm>.json``.
    4. ``repro.write_metadata`` với revision, cache signature, K, hash codebook, α, tham số đã
       chọn, thời gian từng bước, và cờ ``retest``.
    """
    name, seed = cfg["experiment"]["name"], cfg["experiment"]["seed"]
    variant = cfg["representation"]["variant"]
    run_dir = p.runs / name
    metrics_path = run_dir / "metrics_test.json"
    retest = metrics_path.exists()
    if retest and not allow_retest:
        raise RuntimeError(
            f"{name} đã được đánh giá trên test ({metrics_path}). Test chỉ được dùng một lần "
            "cho mỗi cấu hình; nếu thật sự cần chạy lại, thêm --allow-retest."
        )
    repro.set_seed(seed)
    timings: dict[str, float] = {}

    train = load_features(cfg, p, "train")
    val = load_features(cfg, p, "validation")

    k = cfg["bow"]["k"] if BOW_VIEWS[variant] else None
    codebooks = {v: get_codebook(cfg, p, train, v, k) for v in BOW_VIEWS[variant]}
    timings["codebook_s"] = sum(cb.fit_seconds for cb in codebooks.values())

    t0 = time.perf_counter()
    train_bow = encode_split(train, codebooks) if codebooks else None
    val_bow = encode_split(val, codebooks) if codebooks else None
    timings["encode_train_val_s"] = time.perf_counter() - t0

    choice = _select(variant, cfg, train, train_bow, val, val_bow)
    timings["fit_s"] = choice.fit_seconds
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "selection.json",
        {
            "experiment": name,
            "variant": variant,
            "classifier": cfg["classifier"]["type"],
            "k": k,
            "dim": choice.dim,
            "alpha": choice.alpha,
            "best_params": choice.selection.best_params,
            "val_macro_f1": choice.selection.best_score,
            "candidates": choice.candidates,
        },
    )

    # ---- test: đúng một lần, sau khi mọi tham số đã chốt
    test = load_features(cfg, p, "test")
    t0 = time.perf_counter()
    test_bow = encode_split(test, codebooks) if codebooks else None
    X_test = _represent(variant, test.hsv, test_bow, choice.alpha)
    y_pred = np.asarray(choice.selection.model.predict(X_test))
    timings["test_s"] = time.perf_counter() - t0
    m = metrics.evaluate(test.labels, y_pred)

    if retest:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        metrics_path.replace(run_dir / f"metrics_test.{stamp}.json")
    _write_json(metrics_path, {"experiment": name, "split": "test", **m.to_dict()})
    np.save(run_dir / "pred_test.npy", y_pred)
    metrics.save_confusion_png(m.confusion, list(CLASS_NAMES), run_dir / "confusion_test.png")

    repro.write_metadata(
        run_dir,
        cfg,
        cache_signature=train.signature,
        k=k,
        codebooks={v: _codebook_info(cb) for v, cb in codebooks.items()},
        alpha=choice.alpha,
        best_params=choice.selection.best_params,
        val_macro_f1=choice.selection.best_score,
        test_macro_f1=m.macro_f1,
        test_accuracy=m.accuracy,
        timings=timings,
        retest=retest,
    )
    print(
        f"[{name}] val Macro-F1 {choice.selection.best_score:.4f} | "
        f"test Macro-F1 {m.macro_f1:.4f}, Accuracy {m.accuracy:.4f}"
    )
    return run_dir


def run_survey(cfg: dict[str, Any], p: Paths) -> Path:
    """E8: với mỗi K trong ``survey.k_grid``, học hai codebook, mã hóa train + validation, chạy
    từng biến thể trong ``survey.variants`` với lưới C. **Không đụng tới test.**

    Chọn ``K*`` theo Macro-F1 validation của ``survey.select_k_by`` (hòa giữ K nhỏ hơn — lưới
    được sắp tăng dần trước khi chạy). Ghi ``survey.json`` (đủ bảng K × biến thể, kèm thời gian
    codebook / mã hóa / fit) và in K* để nhóm ghi vào ``configs/_base.yaml``.
    """
    name, seed = cfg["experiment"]["name"], cfg["experiment"]["seed"]
    survey = cfg["survey"]
    run_dir = p.runs / name
    repro.set_seed(seed)

    train = load_features(cfg, p, "train")
    val = load_features(cfg, p, "validation")
    k_grid = sorted(survey["k_grid"])

    rows: list[dict[str, Any]] = []
    codebook_info: dict[str, dict] = {}
    for k in k_grid:
        codebooks = {v: get_codebook(cfg, p, train, v, k) for v in VIEWS}
        codebook_info[str(k)] = {v: _codebook_info(cb) for v, cb in codebooks.items()}
        t0 = time.perf_counter()
        train_bow, val_bow = encode_split(train, codebooks), encode_split(val, codebooks)
        encode_s = time.perf_counter() - t0

        for variant in survey["variants"]:
            choice = _select(variant, cfg, train, train_bow, val, val_bow)
            X_val = _represent(variant, val.hsv, val_bow, choice.alpha)
            val_m = metrics.evaluate(val.labels, choice.selection.model.predict(X_val))
            rows.append(
                {
                    "k": k,
                    "variant": variant,
                    "dim": choice.dim,
                    "alpha": choice.alpha,
                    "best_params": choice.selection.best_params,
                    "val_macro_f1": choice.selection.best_score,
                    "val_accuracy": val_m.accuracy,
                    "codebook_s": sum(cb.fit_seconds for cb in codebooks.values()),
                    "encode_s": encode_s,
                    "fit_s": choice.fit_seconds,
                    "candidates": choice.candidates,
                }
            )
            print(
                f"[{name}] K={k} {variant}: val Macro-F1 {choice.selection.best_score:.4f}, "
                f"Accuracy {val_m.accuracy:.4f}"
            )

    by = [r for r in rows if r["variant"] == survey["select_k_by"]]
    k_star = by[classifiers.pick_best([r["val_macro_f1"] for r in by])]["k"]

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "survey.json",
        {"select_k_by": survey["select_k_by"], "k_grid": k_grid, "k_star": k_star, "rows": rows},
    )
    repro.write_metadata(
        run_dir, cfg, cache_signature=train.signature, k_star=k_star, codebooks=codebook_info
    )
    print(f"[{name}] K* = {k_star}. Ghi vào configs/_base.yaml:  bow.k: {k_star}")
    return run_dir


def run(config_path: str | Path, allow_retest: bool = False) -> Path:
    cfg = load_config(config_path)
    validate(cfg)
    p = paths()
    if "survey" in cfg:
        return run_survey(cfg, p)
    return run_experiment(cfg, p, allow_retest=allow_retest)


# --------------------------------------------------------------------------- tiện ích


def _codebook_info(cb: Codebook) -> dict[str, Any]:
    return {"k": cb.k, "hash": cb.hash, "fit_seconds": cb.fit_seconds, "cached": cb.cached}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)
    path.write_text(text, encoding="utf-8")
