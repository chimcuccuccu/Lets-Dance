# `experiments/` — kết quả chạy, không nhét vào `src/`

Mỗi lần train / ablation một thư mục con, ví dụ `experiments/2026-10-12_spatial_v3/`.

Nên lưu:

- Checkpoint (`.pt`) — đã ignore Git, giữ local hoặc Drive
- `metrics.json` / CSV: MAE, RMSE, R², F1 per-error
- Log loss, lệnh chạy, git commit hash
- Bảng **5 ablation** + **held-out dance** (tuần 6) — copy số liệu vào báo cáo từ đây

Notebook chỉ để vẽ; số liệu “chính thức” để ở folder này.
