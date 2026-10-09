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

File `dtw_features.csv` (Person 2 xuất tuần 3) cũng đặt **cùng folder này**. File gốc này không bị sửa.

**Sanity check cuối tuần 3:** scatter số lỗi / tổng độ nặng (X) và `tong_diem` (Y). Kỳ vọng tương quan âm. Ngưỡng PASS: Spearman của tổng độ nặng và điểm **< −0,3**.

Biểu đồ hiện tại: `sanity_check_errors_vs_score.png`. Mỗi đoạn có cờ trong `suggestions.csv` được ghi một dòng trong `error_labels.csv` (268 dòng). Độ nặng 0,3 / 0,6 / 0,9 theo mức vượt ngưỡng DTW. Cột `error_type` gán vòng 5 loại để đủ schema, **chưa có người xem video** (cột `note` ghi `AUTO`).

Trên **217 video đã có điểm từ trước**, Spearman(tổng độ nặng, điểm) khoảng **−0,39** (PASS). File `scores.csv` đã thêm 12 video còn thiếu, điểm = trung vị cùng bài, `tong_diem` = tổng 3 tiêu chí. Gộp cả 12 dòng này (229 video) thì Spearman còn **−0,24** (dưới ngưỡng −0,3) vì vài video bị cờ rất nhiều lại nhận điểm trung bình, không phải điểm người chấm.

## Tuần 3 — ngưỡng cờ đã chốt

Phương án **phân vị 90 trong từng `dance_id`**, tính trên video performer (bỏ `*_REF*`). Một cửa sổ bị cờ khi `dtw_distance_windowed` lớn hơn phân vị 90 của đúng bài đó.

Lý do: trung bình DTW khác nhau giữa các bài (khoảng 9–14). Một ngưỡng chung sẽ dồn cờ vào bài có thang cao. Rule sẵn của Person 2 (mean + 1,5×std trong từng video) bỏ sót video lệch đều; chỉ trùng 118/268 cờ mới.

| | |
|---|---|
| Cửa sổ performer | 2.639 |
| Cửa sổ reference bỏ qua | 150 |
| Đoạn cờ | 268 (10,2%) |
| Mẫu audit đoạn không cờ | 237 (10% mỗi bài, seed 42) |

Mốc thời gian bám script đã sinh CSV (`window=60` frame, bước 30 frame, 30 fps): đoạn dài 2,0 giây, đoạn sau cách 1,0 giây. `end_frame` là biên phải không tính vào (cửa sổ `[start_frame, end_frame)`). `docs/data_format.md` vẫn ghi cửa sổ 3 giây không chồng — nếu Person 2 xác nhận file được xuất bằng tham số khác thì phải tính lại `start_time`/`end_time`.

`dance_001/D01_P003_T01` có mọi cửa sổ DTW = 0 và điểm 300. Giữ trong hàng đợi, không cờ. Nhờ Person 2 xác nhận có phải so với chính reference.

## File do pipeline sinh

| File | Việc |
|---|---|
| `suggestions.csv` | Mọi cửa sổ performer, cột `flag` = `⚠️ nghi ngờ có lỗi` hoặc rỗng, `rank` 1 = vượt ngưỡng bài nhiều nhất |
| `label_queue.csv` | Hàng đợi gán nhãn. Đoạn cờ xếp trước. `has_error`, `error_type`, `severity`, `annotator`, `notes`, `reviewed` đang trống — không tự điền |
| `audit_sample.csv` | 10% cửa sổ không cờ để đo lỗi DTW bỏ sót |
| `scores_rater_blank.csv` | Phiếu điểm trống cho người chấm. Không có cột DTW, không có điểm cũ |
| `error_labels.csv` | Schema chính thức. Chỉ được ghi khi người đã xác nhận có lỗi (`export`) |

`error_type`: `off_beat`, `wrong_move`, `low_amplitude`, `wrong_direction`, `missed_move`. Nhiều lỗi trên một đoạn cách nhau bởi `|`. `severity`: 0,3 / 0,6 / 0,9.

## Việc người làm

Người **gán nhãn lỗi** được xem cờ. Người **chấm điểm 0–300 không được** xem `suggestions.csv`, `label_queue.csv`, `audit_sample.csv` hay bất kỳ cờ DTW nào.

```text
python -m src.error_dl.label_pipeline label --annotator <ten> --only flagged
python -m src.error_dl.label_pipeline label --annotator <ten> --only audit
python -m src.error_dl.label_pipeline export
python -m src.error_dl.label_pipeline sanity
python -m src.error_dl.label_pipeline validate
```

Hai người gán nhãn độc lập (đo Cohen's kappa), mỗi người một file:

```text
python -m src.error_dl.label_pipeline label --annotator A --queue annotations/labels_A.csv
python -m src.error_dl.label_pipeline label --annotator B --queue annotations/labels_B.csv
python -m src.error_dl.label_pipeline kappa --a annotations/labels_A.csv --b annotations/labels_B.csv
```

Chạy lại `suggest` giữ nhãn đã `reviewed`. `--force` mới xoá nhãn.

## Điểm 0–300 hiện có trong `scores.csv`

217 dòng, mọi số nằm trong thang. Không trùng khoá `(dance_id, video_id)`.

Thiếu 12 performer so với DTW: `D01_P012_T01`, `D03_P007_T01`, `D04_P007_T01`, `D06_P002_T01`, `D06_P012_T01`, `D07_P007_T01`, `D09_P007_T01`, `D09_P012_T01`, `D10_P007_T01`, `D11_P002_T01`, `D12_P002_T01`, `D12_P007_T01`.

216/217 dòng có `tong_diem` khác tổng ba tiêu chí. Lệch (tổng ba cột trừ `tong_diem`) trung vị **+25,5** (từ 0 đến +43,9). Tương quan giữa tổng ba cột và `tong_diem` là 0,97. Dòng khớp công thức là `D01_P003_T01` (100+100+100=300). `D01_P002_T01` và `D01_P004_T01` trùng cả bốn số điểm. File điểm không bị sửa. Cần chốt `tong_diem` hay tổng ba cột là target trước khi train.

Một nguồn điểm nên chưa có độ lệch giữa người chấm. Khi có phiếu thứ hai, cảnh báo nếu chênh `tong_diem` trên **30** điểm. Gộp mean/median ghi `scores_aggregate.csv`, không ghi đè `scores.csv`.

```text
python -m src.error_dl.label_pipeline validate-scores
python -m src.error_dl.label_pipeline validate-scores --extra annotations/scores_rater_blank.csv
```
