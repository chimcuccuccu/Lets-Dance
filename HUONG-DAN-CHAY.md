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
2. **Cho Person 3:** Tạo ra file `annotations/dtw_features.csv`. File này chứa khoảng cách DTW (`dtw_dist`) giữa các phân đoạn. Đồng đồng thời engine đã tự động cắm cờ các phân đoạn có khoảng cách lớn (`flag_seg_x = True/False`). Person 3 sẽ dựa vào các cờ này để gán nhãn lỗi cho các phân đoạn.

---

## 8. QA / Demo trực quan Person 1

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

## 9. Thứ tự làm việc gợi ý (Person 1 & Person 2)

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
```

---

## 10. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `ModuleNotFoundError: No module named 'src'` khi import mediapipe | Đã rename `mediapipe.py` → `mp_alias.py`. Dùng `python -m src.pose....` hoặc chạy lại sau khi pull. |
| `extract_all` báo `skipped=...` hết | Pose đã có sẵn. Xoá `.npy` tương ứng hoặc sửa `skip_existing` nếu cần extract lại. |
| Thiếu `annotations/scores.csv` khi build diff / train | Cần file điểm; xem `annotations/`. |
| Visualize không hiện cửa sổ | Cài backend GUI matplotlib; trên remote/SSH thường không hiện được. |
| Import torch / mediapipe lỗi | Active đúng `.venv`, rồi `pip install -r requirements.txt`. |

---

## 11. Tài liệu liên quan

| File | Nội dung |
|------|----------|
| [`README.md`](README.md) | Tổng quan kiến trúc, cấu trúc repo |
| [`src/pose/README.md`](src/pose/README.md) | Pose extract |
| [`src/spatial_dl/README.md`](src/spatial_dl/README.md) | Train Spatial DL |
| [`docs/data_format.md`](docs/data_format.md) | Format dữ liệu bắt buộc |
| [`.env.example`](.env.example) | Biến môi trường mẫu |
