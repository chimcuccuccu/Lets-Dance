# AIST++ pretrain — Spatial encoder (Person 1, tuỳ chọn)

Mục tiêu: học không gian biểu diễn chuyển động trên dataset công khai trước khi fine-tune trên diff tự quay (dataset nhỏ → giảm overfit).

**Không bắt buộc** để pipeline chạy end-to-end. Bỏ qua nếu chưa kịp; quay lại khi dư thời gian (ablation “có / không pretrain” là điểm cộng báo cáo).

## Dữ liệu

| Asset | Lý do |
|---|---|
| `keypoints3d/*.pkl` | COCO-17 3D đã tối ưu thời gian — đủ để map sang MediaPipe, **không** cần video hay SMPL `.pkl` (license) |
| `ignore_list.txt` | Loại sequence lỗi chính thức |

### Nguồn dữ liệu

| Source | Đường dẫn | Format |
|---|---|---|
| `keypoints3d` (mặc định) | `data/aistpp/keypoints3d/*.pkl` | COCO-17 3D |
| `shared` | `dance_coach_project/.../shared_data/dataset/Basic Dance/*.npz` | MediaPipe `(xy,vis)` — ~4464 clip AIST đa camera |

Download AIST++ pkl (nếu dùng `keypoints3d`):

```powershell
python scripts/download_aistpp.py --out data/aistpp
```

Pretrain từ shared_data (đã tiền xử lý sẵn):

```powershell
python -m src.spatial_dl.pretrain --source shared --epochs 20 --camera all
# Chỉ 1 góc camera:
python -m src.spatial_dl.pretrain --source shared --camera c01 --epochs 20
```

Layout sau khi tải:

```
data/aistpp/
  keypoints3d/*.pkl
  ignore_list.txt
  mediapipe_cache/          # tự tạo khi pretrain
```

## Convert skeleton

AIST++ annotate **COCO-17** (output của pipeline fit SMPL của họ). Code map joint chung → MediaPipe-33:

- File: `src/spatial_dl/joints_convert.py`
- `coco17_to_mediapipe` — đường chính
- `smpl24_to_mediapipe` — nếu có sẵn joint *positions* SMPL-24 (không nhận axis-angle)

Joint không map được (ngón tay chi tiết, mắt phụ…) được ước lượng heuristic và gắn `visibility=0` để preprocess nội suy / giảm trọng số.

## Proxy task

Mặc định **`synthetic_diff`**: lấy hai cửa sổ (ưu tiên cùng genre) → `diff = pose_a − pose_b` → autoencoder tái tạo diff.

Lý do: fine-tune dùng `performer − reference`, cùng domain đầu vào hơn AE trên absolute pose.

Chế độ phụ: `--mode pose_ae` (tái tạo pose tuyệt đối).

## Chạy

```powershell
# Pretrain (checkpoint deliverable)
python -m src.spatial_dl.pretrain --epochs 20 --checkpoint checkpoints/spatial_pretrained.pt

# Smoke nhanh (vài sequence)
python -m src.spatial_dl.pretrain --max-sequences 32 --epochs 2

# Fine-tune từ pretrain
python -m src.spatial_dl.train --official --epochs 40 --pretrained checkpoints/spatial_pretrained.pt
```

So sánh ablation (ghi vào báo cáo):

1. Train from scratch (không `--pretrained`)
2. Fine-tune từ `spatial_pretrained.pt`

```powershell
python -m src.spatial_dl.train --official --epochs 40 --checkpoint-dir experiments/spatial_dl_scratch
python -m src.spatial_dl.train --official --epochs 40 --pretrained checkpoints/spatial_pretrained.pt --checkpoint-dir experiments/spatial_dl_pretrained
# So train_history.csv: epoch tới cùng val MAE / Pearson
```

### Kết quả pretrain đã chạy (CPU, 2026-10-03)

| | |
|---|---|
| Sequences | 1363 (sau `ignore_list`) |
| Split | train 1229 / val 134 (theo dancer) |
| Mode | `synthetic_diff` |
| Epochs | 15 |
| Best val recon loss | **0.088** (epoch 14) |
| Checkpoint | `checkpoints/spatial_pretrained.pt` |
| History | `checkpoints/spatial_pretrain_history.csv` |

Loss: 0.219 → 0.095 (train), 0.194 → 0.088 (val) — hội tụ ổn định.

## Checkpoint format

```python
{
  "encoder_state_dict": ...,   # SpatialEncoder — dùng khi fine-tune
  "model_state_dict": ...,     # full SpatialAutoEncoder (tuỳ chọn)
  "pretrain_mode": "synthetic_diff",
  "val_loss": float,
  ...
}
```
