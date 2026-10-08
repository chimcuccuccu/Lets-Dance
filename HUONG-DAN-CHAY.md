# Hướng dẫn chạy — Let's Dance

Mọi lệnh chạy từ thư mục gốc repo: `Lets-Dance/`.

Python **3.10+**. Windows dùng PowerShell như các ví dụ bên dưới.

---

## 1. Cài môi trường (lần đầu)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Kiểm tra import
python -c "import torch, mediapipe, xgboost, faiss, dtaidistance"
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tuỳ chọn: copy `.env.example` → `.env` nếu cần đường dẫn video fallback.

---

## 2. Pose extract (Person 1)

### Batch — mọi video trong `data/dances/`

```powershell
python -m src.pose.extract_all
```

- Input: `data/dances/dance_xxx/.../*.mp4`
- Output: `poses/dance_xxx/<video_id>.npy` shape `(T, 33, 4)`
- Mặc định **bỏ qua** file `.npy` đã có (`skip_existing=True`)

### Một video

```powershell
python src/pose/pose_extractor.py data/dances/dance_001/performers/D01_P003_T01.mp4
```

Hoặc dạng module (không phụ thuộc `sys.path` của script):

```powershell
python -m src.pose.pose_extractor data/dances/dance_001/performers/D01_P003_T01.mp4
```

> **Lưu ý:** Không đặt file tên `mediapipe.py` trong `src/pose/` — sẽ che package MediaPipe khi chạy `python src/pose/pose_extractor.py`. Alias cũ nằm ở `src/pose/mp_alias.py`.

---

## 3. Visualize skeleton

```powershell
python src/visualize.py poses/dance_001/D01_P003_T01.npy
python src/visualize.py test_output.npy
```

Mở cửa sổ matplotlib animation (đóng cửa sổ để thoát).

---

## 4. Smoke test pipeline (pose → preprocess → diff)

```powershell
# Mặc định: D01_P003_T01 + reference dance_001
python src/test_pipeline.py

# Có kèm forward SpatialModelV3 (cần torch + checkpoint)
python src/test_pipeline.py --model

# Chỉ định file
python src/test_pipeline.py poses/dance_001/D01_P003_T01.npy poses/dance_001/dance_001_ref.npy
```

Xuất thêm: `test_output.npy`, `test_diff.npy` ở thư mục gốc.

---

## 5. Build diff + train Spatial DL + eval

Thứ tự bắt buộc: **diff → train → eval**.

```powershell
# 1) Diff performer − reference (sau preprocess + align)
python -m src.preprocessing.build_diffs --scores annotations/scores.csv --poses poses --out poses/diffs

# Build lại dù đã có file
python -m src.preprocessing.build_diffs --scores annotations/scores.csv --poses poses --out poses/diffs --force

# 2) Train SpatialModelV3
python -m src.spatial_dl.train --official --epochs 40

# (Khuyến nghị) Improved: mix pretrain + freeze/unfreeze + khop_dong_tac + early-stop
python -m src.spatial_dl.pretrain --improved
python -m src.spatial_dl.train --improved --official --epochs 40
python scripts/eval_spatial.py --ckpt experiments/spatial_dl_improved/spatial_model_v3_best.pth

# Tuần 5: PCA/t-SNE + ablation XGBoost (Spatial vs raw geometry)
python scripts/visualize_spatial_embeddings.py
python scripts/eval_spatial_xgb.py

# (Tuỳ chọn) Fine-tune đơn giản từ AIST++ pretrain
python -m src.spatial_dl.train --official --epochs 40 --pretrained checkpoints/spatial_pretrained.pt --target tong_diem

# 3) Eval
python scripts/eval_spatial.py
```

Checkpoint: `experiments/spatial_dl/spatial_model_v3_best.pth`  
Improved: `experiments/spatial_dl_improved/spatial_model_v3_best.pth`

### 5b. AIST++ pretrain encoder (tuỳ chọn)

Chi tiết: [`docs/aist_pretrain.md`](docs/aist_pretrain.md)

```powershell
python scripts/download_aistpp.py
python -m src.spatial_dl.pretrain --epochs 20
# Smoke không cần download:
python scripts/smoke_aist_pretrain.py
```

Một số flag train thường dùng:

| Flag | Ý nghĩa |
|------|---------|
| `--official` | Split official theo person |
| `--epochs N` | Số epoch |
| `--batch-size N` | Batch size (mặc định 16) |
| `--data-path` | Thư mục diff (mặc định `poses/diffs`) |
| `--no-aug` | Tắt augmentation |
| `--raw-score` | Không chia `tong_diem/300` |
| `--pretrained PATH` | Load encoder từ AIST pretrain |

---

## 6. Trích xuất đặc trưng hình học (Geometry Features - Person 2)

```powershell
python -m src.features.geometry
```

- Đây là module trích xuất các góc khớp quan trọng (đầu gối, khuỷu tay, vai, hông) từ dữ liệu pose.
- **Output:** Trả về ma trận đặc trưng hình học có shape `(T, 8)`.

---

## 7. Chạy pipeline DTW (Căn chỉnh thời gian - Person 2)

Phải chạy sau khi đã có toàn bộ dữ liệu pose `.npy` trong thư mục `poses/`.

```powershell
python -m src.features.run_dtw_all
```

Module này sẽ tự động duyệt qua tất cả các bài nhảy, lấy file tham chiếu (`_ref.npy`) và dữ liệu người tập (`Dxx_Pxxx_T01.npy`), sau đó thực hiện Dynamic Time Warping để tính độ lệch thời gian.

**Output quan trọng cho các thành viên khác:**
1. **Cho Person 1:** Trả về `alignment_path` (đường dẫn căn chỉnh thời gian) để Person 1 dùng ánh xạ chuỗi frames giữa người tập và tham chiếu khi tính `diff_sequence`.
2. **Cho Person 3:** Tạo ra file `annotations/dtw_features.csv`. File này chứa khoảng cách DTW (`dtw_dist`) giữa các phân đoạn. Đồng thời engine đã tự động cắm cờ các phân đoạn có khoảng cách lớn (`flag_seg_x = True/False`). Person 3 sẽ dựa vào các cờ này để gán nhãn lỗi cho các phân đoạn.

---

## 8. Train Temporal DL (BiLSTM - Person 2)

Temporal DL phân tích nhịp điệu thời gian dựa trên các góc khớp hình học và khoảng cách DTW.
Chi tiết xem tại [`docs/temporal_pretrain.md`](docs/temporal_pretrain.md).

```powershell
# 1) Pretrain trên AIST++ (Phân loại thể loại nhảy, tạo warm-start)
python -m src.temporal_dl.train pretrain --epochs 20

# 2) Train model chính (Dự đoán điểm khớp nhịp trên video thực)
python -m src.temporal_dl.train train --run-name khopnhip_dtw_official
```

- **Pretrain Output:** `checkpoints/temporal_pretrained.pt` và `experiments/temporal_dl/pretrain_history.csv`
- **Train Output:** `experiments/temporal_dl/temporal_{run_name}.pth` và `train_history_{run_name}.csv`

Hệ thống sẽ tự động quét và load `temporal_pretrained.pt` (nếu có) khi chạy `train` để giúp model train hội tụ nhanh hơn.

> ⚠️ **Luôn truyền `--run-name`.** Bỏ trống thì lệnh ghi đè lên `temporal_best.pth` và `train_history.csv` — là deliverable Tuần 4 đã commit.

---

## 8b. Tuần 5 Person 2 — Ablation 3 kịch bản + correlation

Trả lời câu hỏi: *DL có học thêm được gì ngoài thông tin DTW đã có không?*
Phương pháp đầy đủ xem [`docs/temporal_week5.md`](docs/temporal_week5.md).

```powershell
# 1) Encoder (c) — 17 kênh (có kênh DTW)
python -m src.temporal_dl.train train --split official --target-col khop_nhip `
  --run-name khopnhip_dtw_official `
  --pretrain-ckpt checkpoints/temporal_pretrained.pt --epochs 60 --patience 12 --seed 42

# 2) Encoder (b) — 16 kênh (bỏ kênh DTW)
python -m src.temporal_dl.train train --split official --target-col khop_nhip --no-dtw `
  --run-name khopnhip_nodtw_official `
  --pretrain-ckpt checkpoints/temporal_pretrained.pt --epochs 60 --patience 12 --seed 42

# 3) GroupKFold 5 fold (10 lần train, mỗi lần vài phút CPU)
foreach ($k in 0..4) {
  python -m src.temporal_dl.train train --split groupkfold --fold $k --n-folds 5 `
    --target-col khop_nhip --run-name khopnhip_dtw_fold$k --epochs 60 --patience 12
  python -m src.temporal_dl.train train --split groupkfold --fold $k --n-folds 5 --no-dtw `
    --target-col khop_nhip --run-name khopnhip_nodtw_fold$k --epochs 60 --patience 12
}

# 4) Bảng ablation (single split + CV)
python scripts/eval_temporal_xgb.py --split official --target khop_nhip --with-dance-onehot `
  --ckpt-dtw   experiments/temporal_dl/temporal_khopnhip_dtw_official.pth `
  --ckpt-nodtw experiments/temporal_dl/temporal_khopnhip_nodtw_official.pth
python scripts/eval_temporal_xgb.py --cv-folds 5 --split groupkfold `
  --target khop_nhip --with-dance-onehot --ckpt-dir experiments/temporal_dl

# 5) Pearson / Spearman với khop_nhip
python scripts/corr_temporal_khop_nhip.py --target khop_nhip --fdr `
  --ckpt       experiments/temporal_dl/temporal_khopnhip_dtw_official.pth `
  --ckpt-nodtw experiments/temporal_dl/temporal_khopnhip_nodtw_official.pth

# 6) Biểu đồ (chỉ đọc CSV, không cần checkpoint)
python scripts/plot_temporal_week5.py

# 7) Control S5 — xáo nhãn, mọi MAE phải tụt về ≈ baseline
python scripts/eval_temporal_xgb.py --split official --target khop_nhip --shuffle-labels `
  --ckpt-dtw   experiments/temporal_dl/temporal_khopnhip_dtw_official.pth `
  --ckpt-nodtw experiments/temporal_dl/temporal_khopnhip_nodtw_official.pth
```

**Output** (đều trong `experiments/temporal_dl/`): `ablation_xgb_khop_nhip{,_cv,_folds,_shuffled}.csv`, `ablation_paired_tests_khop_nhip.csv`, `ablation_preds_khop_nhip{,_shuffled}.csv`, `correlation_khop_nhip.csv`, `temporal_embeddings_meta.csv`, `temporal_embeddings_2d.csv`, `embeddings_sanity_khop_nhip{,_shuffled}.txt`, `plots/0{1..5}_*.png`.

**Kiểm tra trước khi tin bất kỳ số nào của model** — lệch là pipeline sai:

| Dòng | Phải bằng |
|---|---|
| `dtw_distance_total` @ split=all | n=217, Pearson **−0.4419**, Spearman **−0.3764** |
| `Baseline global-mean` | MAE **2.850**, RMSE **3.787** |
| `Baseline dance-mean` | MAE **1.902**, RMSE **2.650** |

Hai baseline này là vạch so sánh thật: `khop_nhip` có std chỉ 4.11 điểm nên MAE tuyệt đối vô nghĩa nếu không đặt cạnh chúng.

**Reproduce đường legacy Tuần 4** (có leakage cấp cửa sổ, chỉ để đối chiếu) — dùng `--no-scores`, không dùng `--scores-csv ""` vì PowerShell nuốt chuỗi rỗng:

```powershell
python -m src.temporal_dl.train train --no-scores --target-col dtw_distance_total `
  --allow-dtw-target --split random --epochs 30 --run-name week4_legacy_repro
```

---

## 9. QA / Demo trực quan Person 1

```powershell
# Checklist PASS/FAIL (pose, preprocess, diff)
python scripts/qa_person1.py

# Demo trực quan tuần 2–5 → PNG trong experiments/spatial_dl_improved/demo/
python scripts/demo_person1.py

# Xem skeleton 1 video
python src/visualize.py poses/dance_001/D01_P003_T01.npy

# Smoke pipeline + model
python src/test_pipeline.py --model
```

`demo_person1.py` xuất: heatmap diff, train curve, pred vs actual, PCA embedding, ablation bars.

---

## 10. Thứ tự làm việc gợi ý (Person 1 & Person 2)

```
Video
  → extract_all / pose_extractor     → poses/*.npy (Person 1)
  → visualize (tuỳ chọn)             → xem skeleton
  → test_pipeline [--model]          → smoke test
  → extract_geometry (Person 2)      → Tọa độ góc khớp
  → run_dtw_all (Person 2)           → annotations/dtw_features.csv (Person 3) & alignment path (Person 1)
  → build_diffs (Person 1)           → poses/diffs/
  → spatial_dl.train                 → checkpoint
  → eval_spatial                     → MAE / Pearson
  → temporal_dl.train (Person 2)     → encoder (b) 16-ch + (c) 17-ch
  → eval_temporal_xgb (Person 2)     → bảng ablation 3 kịch bản
  → corr_temporal_khop_nhip          → Pearson/Spearman với khop_nhip
  → plot_temporal_week5              → experiments/temporal_dl/plots/
```

---

## 11. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `ModuleNotFoundError: No module named 'src'` khi import mediapipe | Đã rename `mediapipe.py` → `mp_alias.py`. Dùng `python -m src.pose....` hoặc chạy lại sau khi pull. |
| `extract_all` báo `skipped=...` hết | Pose đã có sẵn. Xoá `.npy` tương ứng hoặc sửa `skip_existing` nếu cần extract lại. |
| Thiếu `annotations/scores.csv` khi build diff / train | Cần file điểm; xem `annotations/`. |
| Visualize không hiện cửa sổ | Cài backend GUI matplotlib; trên remote/SSH thường không hiện được. |
| Import torch / mediapipe lỗi | Active đúng `.venv`, rồi `pip install -r requirements.txt`. |

---

## 12. Tài liệu liên quan

| File | Nội dung |
|------|----------|
| [`README.md`](README.md) | Tổng quan kiến trúc, cấu trúc repo |
| [`src/pose/README.md`](src/pose/README.md) | Pose extract |
| [`src/spatial_dl/README.md`](src/spatial_dl/README.md) | Train Spatial DL |
| [`docs/aist_pretrain.md`](docs/aist_pretrain.md) | Person 1: AIST++ Spatial pretrain |
| [`docs/temporal_pretrain.md`](docs/temporal_pretrain.md) | Person 2: AIST++ Temporal (genre) pretrain |
| [`docs/temporal_week5.md`](docs/temporal_week5.md) | Person 2: phương pháp Tuần 5 (ablation DTW vs LSTM, chống leakage, caveat) |
| [`src/temporal_dl/README.md`](src/temporal_dl/README.md) | Person 2: Temporal DL + API `temporal_model()` |
| [`docs/data_format.md`](docs/data_format.md) | Format dữ liệu bắt buộc |
| [`.env.example`](.env.example) | Biến môi trường mẫu |
