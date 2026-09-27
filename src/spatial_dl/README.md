# `src/spatial_dl/` — Person 1, tuần 4–5

Học **hình dạng / tư thế lệch** so với reference từ `diff_sequence`. Đây là nhánh Deep Learning của Person 1.

## Việc phải làm

**Tuần 4**
- Dataset: `diff_sequence` + `tong_diem` (sanity-check, chưa chắc là target chính).
- Model V3 bản đầu: **MLP hoặc 1D-CNN nhỏ** trên `diff_sequence` đã pooling theo thời gian (chưa cần Transformer/GNN).
- Split **theo `person_id`**, không random theo `dance_id`.
- Train vài epoch, loss giảm, có checkpoint.

**Tuần 5**
- Chống overfit: jitter, mirror trái-phải, giảm size nếu cần.
- Xuất `spatial_embedding`, PCA/t-SNE: điểm cao vs điểm thấp có tách cụm không.
- Ablation nhỏ: embedding vs raw geometry → XGBoost tạm.

## File

| File | Vai trò |
|---|---|
| `dataset.py` | Loader |
| `model_v3.py` | Kiến trúc |
| `train.py` | Vòng train |

**Deliverable:** API `spatial_model(diff_sequence) -> embedding` sẵn cho fusion tuần 6.
