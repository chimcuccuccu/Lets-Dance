# Data format (thống nhất giữa 3 người)

Ghi nhớ: làm sai format ở đây thì tuần sau 3 nhánh không ghép được.

## Pose

- File: `poses/{dance_id}/{video_id}.npy` (hoặc cạnh video trong `data/dances/...`)
- Shape: `T × 33 × 4` — `(x, y, z, visibility)` (MediaPipe Pose)

## Diff sequence (Person 1)

- Shape: `T × 33 × 3` — `performer − reference` sau khi align

## DTW (Person 2)

- `dtw_distance(t)`: 1 float mỗi đoạn 2–3s
- Output: `annotations/dtw_features.csv` (dance_id, video_id, dtw_distance_total, dtw_distance_per_segment, ...)

## Nhãn

- `annotations/scores.csv`: dance_id, video_id, person_id, khop_dong_tac, khop_nhip, nang_luong, tong_diem
- `annotations/error_labels.csv`: dance_id, video_id, start_time, end_time, error_type, severity

## Error types (nháp tuần 1)

`off_beat`, `wrong_move`, `low_amplitude`, `wrong_direction`, `missed_move`
