# `src/preprocessing/` — Person 1, tuần 3

Biến pose thô thành chuỗi so sánh được với reference.

## Việc phải làm

1. Center theo hip, scale theo chiều cao, smoothing (Savitzky-Golay hoặc moving average), chuẩn hoá độ dài sequence.
2. Time-alignment thô với reference: **dùng DTW path của Person 2** (P1 không tự viết DTW song song).
3. `compute_diff(performer_pose, reference_pose) -> diff_sequence` shape **`(T, 33, 3)`**.

## File

| File | Vai trò |
|---|---|
| `preprocess.py` | Chuẩn hoá + `compute_diff` / `build_diff_sequence` |
| `build_diffs.py` | Batch tạo `*_diff.npy` từ `poses/` + optional DTW path |

## Kiểm tra nhanh

Performer khớp tốt → diff nhỏ. Performer lệch nhiều → diff lớn. Plot vài chiều landmark trước khi đưa vào Spatial DL.

P3 cũng đọc `diff_sequence` (cửa sổ trượt) để train Error DL — **đừng đổi shape** mà không ghi vào `docs/data_format.md`.
