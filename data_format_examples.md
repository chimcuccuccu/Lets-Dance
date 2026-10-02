# docs/data_format_examples.md — DRAFT v0.1

**Purpose:** Minh họa cấu trúc dữ liệu bằng pseudo-data và code Python kiểm tra shape, thay cho việc tạo file `.npy` binary thật (không cần thiết ở giai đoạn specification).

---

## 1. `pose.npy` — pseudo-data

```python
import numpy as np

# Giả lập 1 video 3 giây, FPS=30 → T=90 frame, 33 landmark, 4 giá trị mỗi landmark
T, NUM_LANDMARKS, NUM_VALUES = 90, 33, 4
pose = np.random.rand(T, NUM_LANDMARKS, NUM_VALUES).astype(np.float32)

# x, y trong khoảng 0-1 (normalized theo MediaPipe), visibility trong khoảng 0-1
pose[:, :, 0:2] = np.random.rand(T, NUM_LANDMARKS, 2)   # x, y
pose[:, :, 2] = np.random.randn(T, NUM_LANDMARKS) * 0.1  # z (độ sâu tương đối)
pose[:, :, 3] = np.random.rand(T, NUM_LANDMARKS)         # visibility

assert pose.shape == (T, 33, 4), "pose.npy phải đúng shape T x 33 x 4"
print("pose.npy shape OK:", pose.shape)
```

## 2. `diff_sequence` — pseudo-data

```python
# Giả lập diff_sequence sau khi đã align performer và reference, cùng độ dài T
diff_sequence = np.random.randn(T, 33, 3).astype(np.float32) * 0.05

assert diff_sequence.shape == (T, 33, 3), "diff_sequence phải đúng shape T x 33 x 3"
print("diff_sequence shape OK:", diff_sequence.shape)

# Sanity check gợi ý: performer khớp tốt -> diff gần 0; performer lệch nhiều -> diff lớn
mean_abs_diff = np.mean(np.abs(diff_sequence))
print("Mean absolute diff (càng nhỏ càng khớp tốt):", mean_abs_diff)
```

## 3. `dtw_distance_windowed` — pseudo-data (CSV, dạng PROPOSED)

```python
import pandas as pd

# Pseudo-data minh họa — KHÔNG phải schema chính thức, xem data_format.md mục B4
dtw_features_sample = pd.DataFrame([
    {"dance_id": "dance_001", "video_id": "dance001_person001_take1",
     "window_id": 0, "start_time": 0.0, "end_time": 3.0, "dtw_distance": 2.1},
    {"dance_id": "dance_001", "video_id": "dance001_person001_take1",
     "window_id": 1, "start_time": 3.0, "end_time": 6.0, "dtw_distance": 8.7},
    {"dance_id": "dance_001", "video_id": "dance001_person001_take1",
     "window_id": 2, "start_time": 6.0, "end_time": 9.0, "dtw_distance": 1.5},
])
print(dtw_features_sample)
# dtw_distance cao (vd. 8.7) chỉ là GỢI Ý nghi ngờ lệch, không phải nhãn lỗi xác định.
```

## 4. `annotations/scores.csv` — Example only, not real annotation data

```csv
dance_id,video_id,khop_dong_tac,khop_nhip,nang_luong,tong_diem
dance_001,dance001_person001_take1,78,70,82,230
dance_001,dance001_person002_take1,55,60,58,173
```

## 5. `annotations/error_labels.csv` — Example only, not real annotation data

```csv
dance_id,video_id,start_time,end_time,error_type,severity
dance_001,dance001_person001_take1,3.0,6.0,off_beat,0.6
dance_001,dance001_person001_take1,24.0,27.0,low_amplitude,0.4
dance_001,dance001_person002_take1,9.0,12.0,wrong_move,0.8
```

## 6. Kiểm tra nhanh toàn bộ (validation script gợi ý)

```python
def validate_pose(pose: np.ndarray):
    assert pose.ndim == 3, "pose phải có 3 chiều"
    assert pose.shape[1] == 33, "phải có đúng 33 landmark"
    assert pose.shape[2] == 4, "phải có đúng 4 giá trị mỗi landmark (x,y,z,visibility)"
    return True

def validate_diff_sequence(diff_seq: np.ndarray):
    assert diff_seq.ndim == 3, "diff_sequence phải có 3 chiều"
    assert diff_seq.shape[1] == 33, "phải có đúng 33 landmark"
    assert diff_seq.shape[2] == 3, "phải có đúng 3 giá trị mỗi landmark (dx,dy,dz)"
    return True

def validate_error_labels(df: pd.DataFrame):
    required_cols = {"dance_id", "video_id", "start_time", "end_time", "error_type", "severity"}
    assert required_cols.issubset(df.columns), f"Thiếu cột bắt buộc: {required_cols - set(df.columns)}"
    valid_types = {"off_beat", "wrong_move", "low_amplitude", "wrong_direction", "missed_move"}
    assert df["error_type"].isin(valid_types).all(), "error_type chứa giá trị không hợp lệ"
    return True
```

> **Lưu ý:** toàn bộ dữ liệu trong file này là pseudo-data để minh họa shape/format, không phải dữ liệu thật của project.
