# Chữ ký hàm (hợp đồng giữa các module)

Đây là hợp đồng để ba người code song song. **Muốn đổi chữ ký thì mở PR riêng sửa file này
trước**, cả nhóm đồng ý rồi mới sửa code.

- Mọi hàm còn là stub đều gọi `todo("<người>", "<tên>")` và ném `NotImplementedError`.
- Test hợp đồng nằm trong `tests/`. Khi gặp stub, test được báo **SKIPPED** (`TODO [B] hog_blocks`);
  khi hàm đã viết xong, test tự chạy thật.
- Quy ước shape: `N` = số ảnh, `K` = kích thước codebook, `nb` = số block (225 hoặc 441).

Người phụ trách: **A** hạ tầng, **B** đặc trưng, **C** phân loại và đánh giá (xem [team.md](team.md)).

## `data/minc.py` — A

```python
Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...]
CLASS_NAMES: tuple[str, ...]          # 23 lớp, thứ tự ClassLabel của nguồn
N_CLASSES = 23

def download(dataset_cfg: dict, root: Path, force: bool = False) -> None
def check_dataset(root: Path, dataset_cfg: dict) -> None          # sai → ValueError
def split_size(root: Path, split: Split) -> int
def load_labels(root: Path, split: Split) -> np.ndarray           # (N,) int64, 0..22
def iter_images(root: Path, split: Split) -> Iterator[np.ndarray] # RGB uint8 (H, W, 3), theo index
```

## `features/views.py` — A

```python
View = Literal["full", "center"]

def full_view(rgb: np.ndarray, size: int = 128) -> np.ndarray     # (128, 128, 3) uint8, INTER_AREA
def center_view(rgb: np.ndarray) -> np.ndarray                    # (s, s, 3), s = min(h, w) // 2
def to_gray(rgb: np.ndarray) -> np.ndarray                        # (H, W) uint8
def extract_views(rgb: np.ndarray) -> dict[View, np.ndarray]      # đã viết sẵn
```

Bất biến: `center_view` lấy gốc trên-trái `((h - s) // 2, (w - s) // 2)`; ảnh 362 → 181.

## `features/hog.py` — B

```python
ORIENTATIONS = 9; PIXELS_PER_CELL = (8, 8); CELLS_PER_BLOCK = (2, 2); BLOCK_NORM = "L2-Hys"
DESC_DIM = 36

def n_blocks(side: int) -> int                                    # đã viết sẵn: 128 → 225, 181 → 441
def hog_blocks(gray: np.ndarray) -> np.ndarray                    # (nb, 36) float32
```

## `features/hsv.py` — B

```python
BINS = (16, 8, 8); RANGES = ((0, 180), (0, 256), (0, 256)); HSV_DIM = 32

def hsv_hist(rgb: np.ndarray) -> np.ndarray                       # (32,) float32
```

Bất biến: resize về 128 trước; mỗi đoạn H `[0:16]`, S `[16:24]`, V `[24:32]` có tổng bằng 1.

## `features/bow.py` — B

```python
def sample_descriptors(train_desc: np.ndarray, n_samples: int, seed: int) -> np.ndarray
    # train_desc (N, nb, 36) → (n_samples, 36) float32; chọn ảnh có lặp, rồi 1 block
def fit_codebook(samples: np.ndarray, k: int, seed: int) -> np.ndarray
    # MiniBatchKMeans(k, batch_size=8192, n_init=3, random_state=seed) → (k, 36) float32
def encode_counts(desc: np.ndarray, codebook: np.ndarray, chunk: int = 2048) -> np.ndarray
    # (N, nb, 36), (K, 36) → (N, K) int32; tổng mỗi hàng = nb
def hellinger_l2(counts: np.ndarray) -> np.ndarray
    # (N, K) → (N, K) float32; √(n/Σn) rồi L2; mỗi hàng chuẩn 1, ≥ 0
```

Bất biến: chỉ `train` được đưa vào `sample_descriptors`; cùng seed ⇒ cùng kết quả.

## `features/fusion.py` — B

```python
def l2_normalize(x: np.ndarray) -> np.ndarray                     # theo hàng; hàng 0 giữ 0
def combine_views(full: np.ndarray, center: np.ndarray) -> np.ndarray
    # L2([L2(full); L2(center)]) → (N, d1 + d2); mỗi khối chiếm 1/2 bình phương chuẩn
def fuse(h_bow: np.ndarray, c: np.ndarray, alpha: float) -> np.ndarray
    # [√α·h_bow ; √(1-α)·c] → (N, 2K + 64); α ngoài [0, 1] → ValueError
```

## `models/classifiers.py` — C

```python
@dataclass
class Selection:
    best_params: dict; best_score: float; model: ClassifierMixin
    scores: list[tuple[dict, float]]; fit_seconds: float

def param_grid(clf_cfg: dict) -> list[dict]
    # linear_svm → [{"C": c}, ...]; knn → [{"n_neighbors": k}, ...]; đúng thứ tự config
def build_classifier(clf_cfg: dict, params: dict, seed: int) -> ClassifierMixin
def pick_best(scores: list[float]) -> int                         # hòa → chỉ số nhỏ nhất
def select_on_validation(X_train, y_train, X_val, y_val, clf_cfg: dict, seed: int) -> Selection
```

Bất biến: không hàm nào ở đây nhận dữ liệu test. Điểm chọn là Macro-F1 trên validation.

## `evaluation/metrics.py` — C

```python
@dataclass
class Metrics:
    accuracy: float; macro_f1: float
    precision, recall, f1, support: np.ndarray    # (23,)
    confusion: np.ndarray                         # (23, 23) đếm thô, hàng = thật, cột = dự đoán
    def to_dict(self) -> dict                     # JSON được

def evaluate(y_true, y_pred, n_classes: int = 23) -> Metrics      # zero_division = 0
def save_confusion_png(confusion, class_names, path: Path) -> None  # chuẩn hóa theo hàng
```

## `evaluation/report.py` — C

```python
ABLATION_PAIRS: list[tuple[str, str, str]]       # (A, B, câu hỏi)

def survey_table(runs_root: Path) -> pd.DataFrame                 # Bảng 9 (E8)
def main_table(runs_root: Path) -> pd.DataFrame                   # Bảng 10 (E1–E7)
def ablation_table(main: pd.DataFrame) -> pd.DataFrame            # Bảng 11
def per_class_table(runs_root: Path, experiment: str = "e5_hsv_bow_multi") -> pd.DataFrame  # Bảng 12
def write_all(runs_root: Path, out_dir: Path) -> None             # CSV + Markdown
```

## `pipeline.py` — A

```python
@dataclass
class SplitFeatures:
    split: Split; labels: np.ndarray
    hog: dict[View, np.ndarray]     # full (N, 225, 36), center (N, 441, 36), memmap float16
    hsv: dict[View, np.ndarray]     # (N, 32)
    signature: str

@dataclass
class Codebook:
    view: View; k: int; centers: np.ndarray; hash: str; fit_seconds: float

def extract(cfg: dict, p: Paths, splits: tuple[Split, ...]) -> None
def load_features(cfg: dict, p: Paths, split: Split) -> SplitFeatures
def get_codebook(cfg: dict, p: Paths, train: SplitFeatures, view: View, k: int) -> Codebook
def encode_split(feats: SplitFeatures, codebooks: dict[View, Codebook]) -> dict[View, np.ndarray]
def build_representation(variant: str, hsv: dict, bow: dict | None, alpha: float = 0.5) -> np.ndarray
    # đã viết sẵn (chỉ ghép hàm của B)
def run_experiment(cfg: dict, p: Paths, allow_retest: bool = False) -> Path   # E1–E7
def run_survey(cfg: dict, p: Paths) -> Path                                    # E8
def run(config_path, allow_retest: bool = False) -> Path                       # đã viết sẵn
```

## Đã viết sẵn (hạ tầng chung)

| Module | Hàm chính |
|---|---|
| `config.py` | `paths() -> Paths` (đọc `.env`), `load_config(path)` (hỗ trợ `base:`), `validate(cfg)` |
| `repro.py` | `set_seed`, `code_hash(*modules)`, `array_hash`, `git_commit`, `library_versions`, `write_metadata(run_dir, cfg, **extra)` |
| `cache.py` | `signature(params, code, revision)`, `array_path(dir, name, sig)`, `with writing(path, shape, dtype) as arr`, `load(path, mmap=True)` |
| `cli.py` | `minc download / check / extract / run / summarize` |
