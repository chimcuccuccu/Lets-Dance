# `src/error_dl/` — Person 3, tuần 4–5

Phát hiện **loại lỗi theo đoạn thời gian** (multi-label). Đây là nhánh Deep Learning của Person 3.

DTW chỉ báo “đoạn này khác thường”, **không** biết là lỗi gì. Con người gán nhãn; model học pattern tinh hơn rule.

## Việc phải làm

**Tuần 1–3 (trước khi train)** — xem `docs/` và `annotations/`: taxonomy, form 0–300, gán nhãn theo cờ DTW, sanity scatter (nhiều lỗi → điểm thấp).

**Tuần 4**
- Loader: `diff_sequence` cửa sổ trượt + nhãn multi-label.
- Shared encoder (MLP/CNN nhỏ) + **một head sigmoid / loại lỗi**, BCE từng head.

**Tuần 5**
- Precision / Recall / F1 **từng loại lỗi** (không chỉ overall accuracy — class lệch).
- So với rule-based (`rule_engine.py`, threshold DTW).
- Nén `error_probabilities` theo đoạn → `error_embedding` cả video (trung bình có trọng số hoặc max-pooling theo severity).

## File

| File | Vai trò |
|---|---|
| `dataset.py` | Cửa sổ + multi-label |
| `error_model.py` | Encoder + heads |
| `train.py` | Vòng train |
| `rule_engine.py` | Baseline, **không phải DL** |

**Deliverable:** API `error_model(diff_sequence_windowed) -> error_probabilities, error_embedding`. Tuần 7 Person 3 chủ trì prompt LLM vì hiểu cấu trúc lỗi rõ nhất.
