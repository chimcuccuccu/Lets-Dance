# `src/spatial_dl/` — Person 1, tuần 4–5

Học **hình dạng / tư thế lệch** so với reference từ `diff_sequence`. Đây là nhánh Deep Learning của Person 1.

## Việc phải làm

**Tuần 4**
- Dataset: `diff_sequence` + `tong_diem` (sanity-check, chưa chắc là target chính).
- Model V3 bản đầu: **1D-CNN nhỏ** trên `diff_sequence` (avg+max pool → embedding L2-norm).
- Split **theo `person_id`** (`dataset.split_indices_by_person`) — không random theo sample.
- Train vài epoch, log MSE/MAE, checkpoint best/last vào `experiments/spatial_dl/`.
- Build diff trước: `python -m src.preprocessing.build_diffs --scores annotations/scores.csv`

**Tuần 5**
- Chống overfit: jitter, mirror trái-phải, giảm size nếu cần.
- Xuất `spatial_embedding`, PCA/t-SNE: điểm cao vs điểm thấp có tách cụm không.
- Ablation nhỏ: embedding vs raw geometry → XGBoost tạm.

## File

| File | Vai trò |
|---|---|
| `dataset.py` | Loader + person split + aug nhẹ |
| `model_v3.py` | 1D-CNN → embedding + score |
| `train.py` | Train / checkpoint / history CSV |

## Chạy tuần 4

```bash
# 1) Diff + DTW path
python -m src.preprocessing.build_diffs --scores annotations/scores.csv --poses poses --out poses/diffs

# 2) Train
python -m src.spatial_dl.train --official --epochs 40

# 3) Eval
python scripts/eval_spatial.py
```

**Deliverable:** API `spatial_model(diff_sequence) -> embedding` sẵn cho fusion tuần 6.
Checkpoint: `experiments/spatial_dl/spatial_model_v3_best.pth`
