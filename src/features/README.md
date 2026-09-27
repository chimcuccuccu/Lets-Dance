# `src/features/` — Person 2 (geometry + DTW) và cả nhóm (fusion)

Ba module độc lập, đừng trộn trách nhiệm.

## `geometry.py` — Person 2, tuần 2

`compute_geometry_features(pose) -> features (T, k)`: góc khớp (gối, khuỷu, vai, …) theo từng frame. Cần pose `.npy` từ Person 1.

## `dtw.py` — Person 2, tuần 3 (việc trọng tâm)

Ba hàm bắt buộc:

| Hàm | Ai dùng kết quả |
|---|---|
| `dtw_align(performer, reference) -> alignment_path` | Person 1 (khớp 2 chuỗi) |
| `dtw_distance_windowed(..., window=2–3s) -> [distance_per_segment]` | Person 3 (cờ nghi ngờ lỗi) |
| `dtw_distance_total(...) -> float` | Fusion / XGBoost |

Chạy DTW **performer × đúng reference của bài đó** (`dance_id`). Xuất `annotations/dtw_features.csv` và bàn giao ngay cho P1 + P3.

DTW là thuật toán cổ điển, **không** thay cho Temporal DL.

## `fusion.py` — cả nhóm, tuần 6

Concatenate: spatial + temporal + error + geometry + dtw + rule → vector cho XGBoost. Chiều embedding chỉ chốt **sau khi** cả 3 nhánh train xong tuần 5.
