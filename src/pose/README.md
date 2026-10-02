# `src/pose/` — Person 1, tuần 2

Trích skeleton từ video bằng MediaPipe Pose.

## Việc phải làm

1. Viết `extract_pose(video_path) -> np.ndarray` shape **`(T, 33, 4)`** = `(x, y, z, visibility)`.
2. Chạy trên **mọi** video reference + performer, mọi bài.
3. Xử lý thiếu người, occlusion, landmark mất khi nhảy nhanh: interpolate hoặc giữ frame trước khi confidence thấp.

## File

| File | Vai trò |
|---|---|
| `pose_extractor.py` | Hàm extract chính (`extract_pose`) |
| `mp_alias.py` | Alias re-export (tương thích docs cũ; không đặt tên `mediapipe.py` vì đụng package) |
| `extract_all.py` | Batch → `poses/{dance_id}/{video_id}.npy` |

## Output

Lưu `.npy` vào [`poses/`](../../poses/README.md), giữ `dance_id` / `video_id` trùng với video gốc.

**Deliverable tuần 2:** `poses/` đủ pose cho toàn bộ dataset. Person 2 cần file này để tính geometry; không xong thì P2 bị chặn.
