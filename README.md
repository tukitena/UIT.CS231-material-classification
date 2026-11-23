# MINC-2500: HSV + BoW-HOG đa tỉ lệ

Đồ án CS231 — Nhập môn thị giác máy tính. Phân loại **23 lớp chất liệu bề mặt** trên MINC-2500
Split 1 bằng đặc trưng thủ công:

- **Màu:** histogram HSV 16/8/8 trên toàn ảnh và vùng tâm (64 chiều);
- **Cạnh:** túi từ thị giác (BoW) trên descriptor block HOG, ở hai tỉ lệ — toàn ảnh thu nhỏ
  128 × 128 và vùng tâm 181 × 181 ở độ phân giải gốc — chuẩn hóa Hellinger + L2 (2K chiều);
- **Kết hợp** có trọng số $z = [\sqrt{\alpha}\,h_{\text{BoW}};\ \sqrt{1-\alpha}\,c]$, phân loại bằng
  **LinearSVC**; KNN dùng để đối chiếu.

Bộ thí nghiệm E1–E8 kiểm chứng đóng góp của từng thành phần (BoW, đa tỉ lệ, màu, α, kích thước
codebook K, bộ phân loại). Mọi siêu tham số được chọn theo Macro-F1 trên validation; test chỉ được
đánh giá đúng một lần cho mỗi cấu hình.

> **Trạng thái:** khung repo. Hạ tầng (config, cache, metadata, CLI) đã viết; các module còn lại là
> stub có đặc tả và test hợp đồng — xem [doc/team.md](doc/team.md).

## Tài liệu

| File | Nội dung |
|---|---|
| [doc/pipeline.md](doc/pipeline.md) | Luồng xử lý, công thức, bộ thí nghiệm, quy trình đánh giá (tóm tắt từ báo cáo) |
| [doc/api.md](doc/api.md) | Chữ ký hàm và bất biến của từng module — hợp đồng giữa các thành viên |
| [doc/config-schema.md](doc/config-schema.md) | Schema config YAML và ví dụ |
| [doc/team.md](doc/team.md) | Chia việc, thứ tự công việc, quy trình Git |
| [doc/todo.md](doc/todo.md) | Checklist việc cần làm — ai xong thì tick |

## Cài đặt

Cần [uv](https://docs.astral.sh/uv/). uv tự tải Python 3.12 nếu máy chưa có.

```powershell
uv sync
```

Lệnh trên tạo `.venv` và cài đúng phiên bản trong `uv.lock` (gồm cả nhóm `dev` và `notebook`).

Tạo file đường dẫn riêng của máy:

```powershell
Copy-Item .env.example .env
```

Sửa `DATASET_ROOT`, `CACHE_ROOT` (cần ~3 GB trống), `RUNS_ROOT` trong `.env` nếu muốn đặt ở chỗ khác.

## Cách chạy

Mọi lệnh chạy qua `uv run`, không cần kích hoạt `.venv`.

**1. Tải và kiểm tra dữ liệu** (MINC-2500 Split 1 từ Hugging Face, revision ghim trong
`configs/_base.yaml`):

```powershell
uv run minc download
```

```powershell
uv run minc check
```

**2. Trích xuất và cache** HOG + HSV hai góc nhìn cho cả ba split (`--jobs -1` dùng mọi lõi CPU):

```powershell
uv run minc extract --jobs -1
```

Lệnh này kiểm tra dataset trước (như `minc check`), và bỏ qua split nào đã có cache đúng signature.

Nếu đã tải dataset ở chỗ khác (ví dụ bản `save_to_disk` của dự án cũ), chỉ cần trỏ `DATASET_ROOT`
trong `.env` tới thư mục chứa `dataset_dict.json` và `source.json`, không cần tải lại.

**3. Khảo sát K (E8)** — chỉ dùng train + validation:

```powershell
uv run minc run configs/e8_codebook_k_survey.yaml
```

Ghi K\* in ra vào `bow.k` trong `configs/_base.yaml`.

**4. Chạy các cấu hình chính E1–E7**, ví dụ phương án chính E5:

```powershell
uv run minc run configs/e5_hsv_bow_multi.yaml
```

Mỗi config chỉ được đánh giá test **một lần**. Chạy lại sẽ báo lỗi trừ khi thêm `--allow-retest`
(và việc này được ghi vào metadata).

**5. Tổng hợp bảng cho báo cáo** (Bảng 9–12, confusion matrix):

```powershell
uv run minc summarize
```

Kết quả từng lần chạy nằm trong `runs/<tên thí nghiệm>/`: `metadata.json` (commit, revision dataset,
cache signature, seed, K, hash codebook, phiên bản thư viện), `selection.json`, `metrics_test.json`,
`confusion_test.png`.

## Phát triển

```powershell
uv run pytest
```

```powershell
uv run ruff format
```

```powershell
uv run ruff check --fix
```

Test của hàm chưa cài đặt được báo `SKIPPED  TODO [B] hog_blocks` — `uv run pytest -ra` cho biết còn
lại những gì. Thêm thư viện bằng `uv add <gói>` và commit cả `uv.lock`.

## Cấu trúc

```
configs/            _base.yaml + e1…e8 (kế thừa base)
doc/                tài liệu thiết kế
notebooks/          phân tích, vẽ hình cho báo cáo
src/minc_classification/
  config.py repro.py cache.py cli.py pipeline.py
  data/minc.py
  features/views.py hog.py hsv.py bow.py fusion.py
  models/classifiers.py
  evaluation/metrics.py report.py
tests/              test hợp đồng cho từng module
```

## Nhóm

| Thành viên | MSSV | Vai trò |
|---|---|---|
| Lưu Tổng Triều Lam | 24520930 | Trưởng nhóm |
| Trần Anh Kiên | 24520889 | Thành viên |
| Đỗ Ngọc Tuấn Kiệt | 24520896 | Thành viên |

Giảng viên hướng dẫn: Mai Tiến Dũng.
