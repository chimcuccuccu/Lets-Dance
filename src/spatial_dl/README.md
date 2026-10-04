# `src/spatial_dl/` — Person 1, tuần 4–5 (+ AIST pretrain tuỳ chọn)

Học **hình dạng / tư thế lệch** so với reference từ `diff_sequence`.

## File

| File | Vai trò |
|---|---|
| `dataset.py` | Loader diff + person split + aug (jitter, mirror, shift…) |
| `model_v3.py` | `SpatialEncoder` / `SpatialModelV3` / `SpatialAutoEncoder` |
| `train.py` | Fine-tune + freeze→unfreeze + early stopping |
| `infer.py` | **API fusion:** `encode_diff` / `export_spatial_embeddings` |
| `joints_convert.py` | COCO-17 / SMPL-24 → MediaPipe-33 |
| `aist_dataset.py` | AIST++ load + cache + synthetic diff |
| `pretrain.py` | Pretrain encoder trên AIST++ |

## Tuần 4 — fine-tune

```powershell
python -m src.preprocessing.build_diffs --scores annotations/scores.csv --poses poses --out poses/diffs
python -m src.spatial_dl.train --official --epochs 40
python scripts/eval_spatial.py
```

Checkpoint: `experiments/spatial_dl/spatial_model_v3_best.pth`

## AIST++ pretrain + improved fine-tune

Chi tiết: [`docs/aist_pretrain.md`](../../docs/aist_pretrain.md)

```powershell
python -m src.spatial_dl.pretrain --improved
python -m src.spatial_dl.train --improved --official --epochs 40
python scripts/eval_spatial.py --ckpt experiments/spatial_dl_improved/spatial_model_v3_best.pth
```

## Tuần 5 — embedding, visualize, ablation (+ cải tiến)

Cải tiến Person 1 (không đụng P2/P3):
- **Dual embedding:** raw → score head; L2-norm → fusion
- **Aux loss:** dự đoán mean `|diff|` theo face/arms/torso/legs (limbs nặng hơn)
- **TTA:** trung bình dự đoán gốc + mirror lúc infer

```powershell
python -m src.spatial_dl.train --improved --official --epochs 40
python scripts/eval_spatial.py --ckpt experiments/spatial_dl_improved/spatial_model_v3_best.pth
python scripts/visualize_spatial_embeddings.py
python scripts/eval_spatial_xgb.py
```

API:
```python
from src.spatial_dl import load_spatial_checkpoint, encode_diff, predict_score
model, meta = load_spatial_checkpoint("experiments/spatial_dl_improved/spatial_model_v3_best.pth")
emb = encode_diff(diff, model, use_tta=True)          # (256,) fusion
score = predict_score(diff, model, use_tta=True)      # điểm khop_dong_tac
```

Không đụng code Person 2/3 (chỉ *đọc* geometry/DTW khi ablation).
