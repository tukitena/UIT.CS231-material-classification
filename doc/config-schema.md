# Schema config thí nghiệm

Config là file YAML trong `configs/`. Mọi config thí nghiệm kế thừa `configs/_base.yaml` bằng dòng
`base: _base.yaml` và **chỉ ghi phần khác**. Dict được trộn đệ quy; list và giá trị đơn bị ghi đè
hoàn toàn (ví dụ `alpha_grid: [0.25, 0.5, 0.75]` thay hẳn `[0.5]`).

Đường dẫn riêng từng máy **không** nằm trong config mà trong `.env` (xem `.env.example`).

`config.validate()` chạy trước mỗi lần `minc run` và báo lỗi sớm nếu config sai.

## Các khối

### `experiment`

| Khoá | Kiểu | Bắt buộc | Ý nghĩa |
|---|---|---|---|
| `name` | str | có | Tên thí nghiệm, đồng thời là tên thư mục `RUNS_ROOT/<name>/`. Quy ước: trùng tên file. |
| `seed` | int | có (base = 42) | Seed cho lấy mẫu descriptor, K-means, LinearSVC |

### `dataset` — chỉ đặt ở `_base.yaml`

| Khoá | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | str | `mcimpoi/minc-2500_split_1` trên Hugging Face |
| `revision` | str | Commit SHA ghim. Đổi SHA ⇒ đổi dữ liệu ⇒ cache cũ tự mất hiệu lực |
| `splits` | list[str] | `[train, validation, test]` |
| `expected_counts` | dict | `{train: 48875, validation: 2875, test: 5750}` |
| `class_names` | list[str] | 23 lớp theo thứ tự nhãn 0..22 |

### `features` — cố định theo báo cáo, không khảo sát

Toàn bộ khối này tham gia vào **cache signature**: sửa bất kỳ giá trị nào ⇒ phải trích xuất lại.

| Khoá | Giá trị | Ý nghĩa |
|---|---|---|
| `views.full` | `{resize: 128, interpolation: INTER_AREA}` | Góc nhìn toàn ảnh |
| `views.center` | `{crop: half_min_side}` | Cắt giữa cạnh ⌊min(h,w)/2⌋ |
| `hog.orientations` | 9 | |
| `hog.pixels_per_cell` | `[8, 8]` | |
| `hog.cells_per_block` | `[2, 2]` | |
| `hog.block_norm` | `L2-Hys` | |
| `hog.dtype` | `float16` | dtype lưu cache |
| `hsv.image_size` | 128 | |
| `hsv.bins` | `[16, 8, 8]` | H, S, V |

### `bow`

| Khoá | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `k` | int \| null | `null` | **K\***. Để trống tới khi chạy E8, sau đó ghi giá trị vào `_base.yaml`. Các biến thể có BoW báo lỗi nếu còn `null`. |
| `n_samples` | int | 200000 | Số descriptor lấy mẫu từ train để học codebook |
| `batch_size` | int | 8192 | MiniBatchKMeans |
| `n_init` | int | 3 | MiniBatchKMeans |

### `representation`

| Khoá | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `variant` | enum | `hsv_bow_multi` | `hsv` (E1), `bow_full` (E2), `bow_center` (E3), `bow_multi` (E4), `hsv_bow_multi` (E5–E7) |
| `alpha_grid` | list[float] | `[0.5]` | Các giá trị α ∈ [0, 1] để chọn trên validation. Chỉ dùng với `hsv_bow_multi`. Thứ tự quyết định khi hòa. |

### `classifier`

| Khoá | Kiểu | Ý nghĩa |
|---|---|---|
| `type` | `linear_svm` \| `knn` | Bộ phân loại được dùng |
| `linear_svm.C_grid` | list[float] | `[0.01, 0.1, 1.0, 10.0]` |
| `linear_svm.penalty` / `loss` | str | `l2` / `squared_hinge` |
| `linear_svm.max_iter` / `tol` | int / float | `5000` / `0.0001` |
| `knn.k_grid` | list[int] | `[3, 5, 7, 9]` — số láng giềng, **khác** K của codebook |
| `knn.weights` / `metric` | str | `uniform` / `euclidean` |

Cả hai khối con đều nằm trong `_base.yaml`; config chỉ cần đổi `type`.

### `survey` — chỉ có ở E8

Có khối này ⇒ `minc run` chạy khảo sát, **không đánh giá test**.

| Khoá | Kiểu | Ý nghĩa |
|---|---|---|
| `k_grid` | list[int] | `[256, 512, 1024]`, xếp tăng dần (hòa giữ K nhỏ hơn) |
| `variants` | list[str] | `[bow_multi, hsv_bow_multi]` — E4 và E5 |
| `select_k_by` | str | `bow_multi` — chọn K\* theo Macro-F1 validation của E4 |

### `evaluation`

| Khoá | Giá trị | Ý nghĩa |
|---|---|---|
| `selection_metric` | `macro_f1` | Chỉ số chọn tham số trên validation |

## Ví dụ

`configs/e6_hsv_bow_multi_alpha.yaml`:

```yaml
# E6 — Như E5 nhưng chọn α trên validation. Chọn α, C.
base: _base.yaml
experiment: {name: e6_hsv_bow_multi_alpha}
representation:
  variant: hsv_bow_multi
  alpha_grid: [0.25, 0.5, 0.75]
```

Sau khi trộn với `_base.yaml`, config E6 có đủ `dataset`, `features`, `bow`, `classifier`
(LinearSVC với lưới C), và chọn trên lưới (α, C) theo thứ tự α ngoài, C trong.

`configs/e8_codebook_k_survey.yaml`:

```yaml
base: _base.yaml
experiment: {name: e8_codebook_k_survey}
survey:
  k_grid: [256, 512, 1024]
  variants: [bow_multi, hsv_bow_multi]
  select_k_by: bow_multi
representation:
  alpha_grid: [0.5]
```

## Thêm một thí nghiệm mới

1. Tạo `configs/<tên>.yaml` với `base: _base.yaml` và `experiment.name: <tên>`.
2. Chỉ ghi các khoá khác base.
3. Chạy `uv run pytest tests/test_config.py` để chắc config hợp lệ.
