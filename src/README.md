# `src/` — code pipeline

Toàn bộ mã nguồn. Mỗi package tương ứng một khối trong kiến trúc. **Không** nhét notebook khảo sát vào đây (để ở `notebooks/`).

## Thứ tự đọc / implement

```
Tuần 2    pose  →  (poses/*.npy)
Tuần 3    features/dtw + preprocessing  →  diff_sequence, alignment
Tuần 4–5  spatial_dl / temporal_dl / error_dl  (3 người song song)
Tuần 6    features/fusion + scoring
Tuần 7    retrieval + feedback
```

## Package

| Package | Người | Việc |
|---|---|---|
| [`pose/`](pose/README.md) | P1 | MediaPipe → `(T, 33, 4)` |
| [`preprocessing/`](preprocessing/README.md) | P1 (+ P2 align) | Chuẩn hoá pose, `diff_sequence` |
| [`spatial_dl/`](spatial_dl/README.md) | P1 | DL hình dạng / tư thế |
| [`features/`](features/README.md) | P2 (+ fusion chung) | Geometry, DTW, fusion |
| [`temporal_dl/`](temporal_dl/README.md) | P2 | DL nhịp / timing |
| [`error_dl/`](error_dl/README.md) | P3 | DL loại lỗi + rule baseline |
| [`scoring/`](scoring/README.md) | Chung tuần 6 | XGBoost 0–300 |
| [`retrieval/`](retrieval/README.md) | Chung tuần 7 | FAISS theo `dance_id` |
| [`feedback/`](feedback/README.md) | P3 chủ trì tuần 7 | Prompt LLM |

Mỗi người **bắt buộc** có 1 neural net riêng, tự train, tự eval. DTW / XGBoost / rule không thay thế được phần DL.
