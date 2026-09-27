# `src/temporal_dl/` — Person 2, tuần 4–5

Học **nhịp / timing lệch** so với reference. Đây là nhánh Deep Learning của Person 2.

Input gợi ý: chuỗi ghép geometry performer, geometry reference (hoặc diff), và `dtw_distance` theo thời gian.

## Việc phải làm

**Tuần 4**
- Dataset: `(performer_geometry_seq, reference_geometry_seq, dtw_distance_seq)`.
- Model đầu: **1-layer BiLSTM/GRU nhỏ**. Loss phải giảm.

**Tuần 5 — 3 kịch bản bắt buộc** (đưa qua XGBoost tạm, so MAE/RMSE):

| Kịch bản | Ý nghĩa |
|---|---|
| (a) DTW-only | Chỉ `dtw_distance`, không DL |
| (b) LSTM-only | Bỏ DTW, chỉ raw sequence |
| (c) DTW+LSTM | Kết hợp |

Tính Pearson/Spearman giữa embedding (hoặc DTW) với cột `khop_nhip` trong form chấm điểm.

## File

| File | Vai trò |
|---|---|
| `dataset.py` | Loader sequence |
| `lstm.py` | BiLSTM/GRU |
| `train.py` | Vòng train |

**Deliverable:** API `temporal_model(seq) -> embedding, dtw_features`.
