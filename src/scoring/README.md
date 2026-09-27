# `src/scoring/` — cả nhóm, tuần 6

Chấm điểm **0–300** bằng XGBoost trên vector fusion. Đây **không** phải Deep Learning.

## Việc phải làm

1. Train trên **dữ liệu gộp mọi bài** (không train riêng từng `dance_id`). Target = `tong_diem`.
2. Chạy đủ **5 ablation**: Geometry+DTW only / Spatial only / Spatial+Temporal / +Error / Full fusion → MAE, RMSE, R².
3. **Held-out dance:** giữ nguyên 1 bài ngoài tập train, chỉ test — đo tổng quát hoá sang bài chưa thấy.

Bảng ablation + held-out là bằng chứng khoa học chính của báo cáo. Checkpoint / log để ở `experiments/`.
