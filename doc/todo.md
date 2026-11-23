# Việc cần làm

Ai làm xong việc nào thì đổi `- [ ]` thành `- [x]` **trong cùng PR với việc đó**, và ghi tên mình
nếu việc chưa có người nhận. Vai trò A / B / C: xem [team.md](team.md).

Một module chỉ được tick khi: không còn `todo(...)`, test hợp đồng không còn SKIPPED
(`uv run pytest -ra`), và PR đã được review, merge vào `main`.

## 0. Chuẩn bị

- [x] Dựng khung repo bằng uv, tài liệu `doc/`
- [x] Đẩy repo lên GitHub, mời đủ thành viên
- [x] Chốt ai giữ phần A / B / C — A: Kiệt · B: Kiên · C: Lam
- [x] Cả nhóm đọc và thống nhất [api.md](api.md), [config-schema.md](config-schema.md)
- [x] Mỗi máy chạy `uv sync`, tạo `.env` từ `.env.example`, `uv run pytest` qua

## 1. Code

### A — hạ tầng và tích hợp (nhánh `feat/preprocess`)

- [x] Sửa `.gitignore` bỏ qua nhầm `src/minc_classification/data/`
- [x] `data/minc.py`: `download`, `check_dataset`, `split_size`, `load_labels`, `iter_images`
- [x] `features/views.py`: `full_view`, `center_view`, `to_gray`
- [x] `pipeline.py`: `extract` (có `--jobs`), `load_features`
- [x] `pipeline.py`: `get_codebook`, `encode_split`
- [x] `pipeline.py`: `run_experiment` (khoá test một lần), `run_survey` (E8)
- [x] Test: `tests/test_data.py`, `tests/test_pipeline_a.py`

### B — đặc trưng

- [ ] `features/hog.py`: `hog_blocks`
- [ ] `features/hsv.py`: `hsv_hist`
- [ ] `features/bow.py`: `sample_descriptors`, `fit_codebook`
- [ ] `features/bow.py`: `encode_counts`, `hellinger_l2`
- [ ] `features/fusion.py`: `l2_normalize`, `combine_views`, `fuse`

### C — phân loại và đánh giá

- [ ] `models/classifiers.py`: `param_grid`, `build_classifier`, `pick_best`
- [ ] `models/classifiers.py`: `select_on_validation`
- [ ] `evaluation/metrics.py`: `evaluate`, `Metrics.to_dict`, `save_confusion_png`
- [ ] `evaluation/report.py`: `survey_table`, `main_table`, `ablation_table`, `per_class_table`, `write_all`

## 2. Thực nghiệm

Chỉ bắt đầu khi mục 1 của A và B đã merge. Ghi máy đã chạy và commit vào cuối dòng.

- [ ] `uv run minc check` qua trên máy chạy thí nghiệm
- [ ] Thử `uv run minc extract --split validation --jobs -1` (kiểm tra chế độ song song)
- [ ] `uv run minc extract --jobs -1` cho cả ba split (~3 GB cache)
- [ ] **E1** — HSV (kiểm tra pipeline từ đầu đến cuối)
- [ ] **E8** — khảo sát kích thước codebook, **chạy trên một máy duy nhất** (Bảng 9 so sánh
      thời gian) — máy: …
  - Lệnh: `uv run minc run configs/e8_codebook_k_survey.yaml`
  - Lưới **K ∈ {256, 512, 1024}**; mỗi K học hai codebook (toàn ảnh, vùng tâm) **chỉ trên
    train**, rồi chạy **E4** (2K chiều) và **E5** (2K + 64 chiều) với lưới C.
  - Chỉ dùng train + validation, **không đụng test**.
  - **K\*** = K có Macro-F1 validation cao nhất của **E4** (không chọn theo E5); hòa thì giữ
    K nhỏ hơn.
  - [ ] K = 256 xong
  - [ ] K = 512 xong
  - [ ] K = 1024 xong (chưa đo thời gian trước, có thể lâu nhất)
  - [ ] `runs/e8_codebook_k_survey/survey.json` có đủ 6 dòng (3 K × E4/E5): Macro-F1 val,
        Accuracy val, số chiều, thời gian codebook / mã hóa / fit
- [ ] Ghi K\* vào `bow.k` trong `configs/_base.yaml` (lệnh E8 in sẵn giá trị) — K\* = …
- [ ] **E2** — BoW toàn ảnh
- [ ] **E3** — BoW vùng tâm
- [ ] **E4** — BoW đa tỉ lệ
- [ ] **E5** — HSV + BoW đa tỉ lệ, α = 0,5 (phương án chính)
- [ ] **E6** — chọn α
- [ ] **E7** — KNN
- [ ] `uv run minc summarize` ra Bảng 9–12 và confusion matrix

## 3. Báo cáo

- [ ] Điền "STT nhóm" và "Thời gian kết thúc" ở trang bìa
- [ ] IV.1.1: phiên bản thư viện và cấu hình máy (lấy từ `runs/*/metadata.json`)
- [ ] IV.2: Bảng 9 (E8) và nhận xét xu hướng theo K
- [ ] IV.3: Bảng 10 (E1–E7) và nhận xét
- [ ] IV.4: Bảng 11 (ablation) và nhận xét từng cặp
- [ ] IV.5: Bảng 12, Hình 13 (confusion matrix E5) và phân tích lớp
- [ ] Chương V — Kết luận
- [ ] Nguồn tham khảo
- [ ] Đọc lại toàn bộ, thống nhất văn phong
- [ ] Xuất PDF nộp