# Let's Dance — Movement Assessment

Hệ thống chấm điểm nhảy **multi-dance**: so performer với reference của đúng bài (`dance_id`), xuất điểm 0–300, loại lỗi theo thời gian, và feedback bằng LLM.

Ba người, **ba nhánh DL riêng**, rồi mới ghép. XGBoost / DTW / rule engine **không** tính là Deep Learning.

```
performer_video + dance_id
        ↓
load reference_pose[dance_id]
        ↓
MediaPipe → preprocessing → diff_sequence = performer − reference (đã align)
        ↓
┌───────────┬───────────┬───────────┐
│ Spatial DL│ Temporal  │ Error DL  │  ← 3 người, song song, dùng chung mọi bài
│ (Person 1)│ (Person 2)│ (Person 3)│
└───────────┴───────────┴───────────┘
        ↓
   Feature Fusion → XGBoost → Score 0–300
        ↓
   FAISS (lọc theo dance_id) + LLM feedback
```

## Ai phụ trách gì

| Thành viên | Nhánh | Thư mục chính | Deliverable |
|---|---|---|---|
| Đỗ Lý Minh Anh | Spatial DL — hình / tư thế lệch | `src/pose`, `src/preprocessing`, `src/spatial_dl` | embedding không gian |
| Nguyễn Duy Anh | Temporal DL + DTW — nhịp / timing | `src/features`, `src/temporal_dl` | embedding thời gian + DTW |
| Nguyễn Thị Ngân | Error DL — loại lỗi theo đoạn | `src/error_dl`, `docs/` | xác suất lỗi + embedding lỗi |
| Cả nhóm (tuần 6–8) | Fusion, FAISS, LLM, báo cáo | `src/features/fusion.py`, `src/scoring`, `src/retrieval`, `src/feedback` | điểm + feedback |

Đọc README trong từng folder trước khi viết code. Format dữ liệu bắt buộc: [`docs/data_format.md`](docs/data_format.md).

## Cài môi trường (cả nhóm, cùng 1 file)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python -c "import torch, mediapipe, xgboost, faiss, dtaidistance"
```

Python **3.10+**. Import không lỗi trên máy mỗi người thì mới tách nhánh.

**Chạy pipeline / lệnh cụ thể:** xem [`HUONG-DAN-CHAY.md`](HUONG-DAN-CHAY.md).

## Cây thư mục

| Folder | Dùng để làm gì |
|---|---|
| [`data/`](data/README.md) | Video reference + performer theo từng bài |
| [`poses/`](poses/README.md) | Pose `.npy` sau khi extract (Person 1) |
| [`src/`](src/README.md) | Toàn bộ code pipeline |
| [`annotations/`](annotations/README.md) | Nhãn điểm 0–300 và nhãn lỗi |
| [`docs/`](docs/README.md) | Format, taxonomy lỗi, protocol chấm điểm |
| [`configs/`](configs/README.md) | Hyperparameter, đường dẫn, seed |
| [`experiments/`](experiments/README.md) | Log train, ablation, held-out dance |
| [`notebooks/`](notebooks/README.md) | Khảo sát, plot, sanity check |

Video và `.npy` **không** đẩy lên Git (xem `.gitignore`). Chỉ commit code, nhãn, docs, config.
