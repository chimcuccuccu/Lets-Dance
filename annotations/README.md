# `annotations/` — nhãn (Person 3 chủ trì, cả nhóm review)

Hai file CSV nhóm cùng dùng. Header đã có sẵn — **đừng đổi tên cột** nếu chưa sửa `docs/data_format.md`.

## `scores.csv`

Chấm 0–300, 3 tiêu chí con. Người chấm = người biết nhảy / dạy nhảy, không tự chấm lấy cho dễ.

| Cột | Ý nghĩa |
|---|---|
| `dance_id` | Bài nhảy |
| `video_id` | File performer |
| `person_id` | Người nhảy (dùng để split train/val) |
| `khop_dong_tac` | 0–100 |
| `khop_nhip` | 0–100 — Person 2 đối chiếu Temporal DL |
| `nang_luong` | 0–100 |
| `tong_diem` | 0–300 — target XGBoost |

## `error_labels.csv`

Nhãn lỗi theo đoạn thời gian. Tuần 3: nhận `dtw_distance_windowed` từ P2 → chỉ xem kỹ đoạn có cờ ⚠️, rồi **phân loại lỗi** (DTW không biết loại).

| Cột | Ý nghĩa |
|---|---|
| `start_time`, `end_time` | Đoạn (giây) |
| `error_type` | `off_beat`, `wrong_move`, `low_amplitude`, `wrong_direction`, `missed_move` |
| `severity` | Mức nặng (thống nhất trong taxonomy) |

File `dtw_features.csv` (Person 2 xuất tuần 3) cũng đặt **cùng folder này** khi có dữ liệu.

**Sanity check cuối tuần 3:** scatter số lỗi/độ nặng vs `tong_diem`. Không hợp lý thì dừng train DL, rà lại taxonomy / cách chấm.
