# Data format (thống nhất giữa 3 người)

Ghi nhớ: làm sai format ở đây thì tuần sau 3 nhánh không ghép được.

## Pose

- File: `poses/{dance_id}/{video_id}.npy` (hoặc cạnh video trong `data/dances/...`)
- Shape: `T × 33 × 4` — `(x, y, z, visibility)` (MediaPipe Pose)

## Diff sequence (Person 1)

- Shape: `T × 33 × 3` — `performer − reference` sau khi align
- Pipeline chuẩn: confidence filter → interp → center hip → **mean spine scale** → Savitzky–Golay → DTW warp (path từ Person 2) → subtract → optional resample `T=150`
- File gợi ý: `poses/diffs/{dance_id}/{video_id}_diff.npy` (và bản flat `{dance_id}_{video_id}_diff.npy`)
- DTW path (nếu có): `annotations/dtw_paths/{dance_id}/{video_id}_path.npy` shape `(L, 2)` — cột 0 = index performer, cột 1 = index reference
- Khi chưa có DTW path: fallback resample cả hai về cùng `T` rồi trừ (chỉ dùng tạm; Spatial DL chính thức ưu tiên path từ P2)

## DTW (Person 2)

- `dtw_distance(t)`: 1 float mỗi đoạn 2–3s
- Output: `annotations/dtw_features.csv` (dance_id, video_id, dtw_distance_total, dtw_distance_per_segment, ...)

## Nhãn

- `annotations/scores.csv`: dance_id, video_id, person_id, khop_dong_tac, khop_nhip, nang_luong, tong_diem
- `annotations/error_labels.csv`: dance_id, video_id, start_time, end_time, error_type, severity

## Error types (nháp tuần 1)

`off_beat`, `wrong_move`, `low_amplitude`, `wrong_direction`, `missed_move`
