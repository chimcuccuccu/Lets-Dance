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

# 3) Eval
python scripts/eval_spatial.py
```

Checkpoint: `experiments/spatial_dl/spatial_model_v3_best.pth`

Một số flag train thường dùng:

| Flag | Ý nghĩa |
|------|---------|
| `--official` | Split official theo person |
| `--epochs N` | Số epoch |
| `--batch-size N` | Batch size (mặc định 16) |
| `--data-path` | Thư mục diff (mặc định `poses/diffs`) |
| `--no-aug` | Tắt augmentation |
| `--raw-score` | Không chia `tong_diem/300` |

---

## 6. QA Person 1

```powershell
python scripts/qa_person1.py
python scripts/qa_person1.py --dance dance_001 --visualize
```

Kiểm tra coverage pose, invariant preprocess, diff vs reference.

---

## 7. Thứ tự làm việc gợi ý (Person 1)

```
Video
  → extract_all / pose_extractor     → poses/*.npy
  → visualize (tuỳ chọn)             → xem skeleton
  → test_pipeline [--model]          → smoke test
  → build_diffs                      → poses/diffs/
  → spatial_dl.train                 → checkpoint
  → eval_spatial                     → MAE / Pearson
```

---

## 8. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `ModuleNotFoundError: No module named 'src'` khi import mediapipe | Đã rename `mediapipe.py` → `mp_alias.py`. Dùng `python -m src.pose....` hoặc chạy lại sau khi pull. |
| `extract_all` báo `skipped=...` hết | Pose đã có sẵn. Xoá `.npy` tương ứng hoặc sửa `skip_existing` nếu cần extract lại. |
| Thiếu `annotations/scores.csv` khi build diff / train | Cần file điểm; xem `annotations/`. |
| Visualize không hiện cửa sổ | Cài backend GUI matplotlib; trên remote/SSH thường không hiện được. |
| Import torch / mediapipe lỗi | Active đúng `.venv`, rồi `pip install -r requirements.txt`. |

---

## 9. Tài liệu liên quan

| File | Nội dung |
|------|----------|
| [`README.md`](README.md) | Tổng quan kiến trúc, cấu trúc repo |
| [`src/pose/README.md`](src/pose/README.md) | Pose extract |
| [`src/spatial_dl/README.md`](src/spatial_dl/README.md) | Train Spatial DL |
| [`docs/data_format.md`](docs/data_format.md) | Format dữ liệu bắt buộc |
| [`.env.example`](.env.example) | Biến môi trường mẫu |
