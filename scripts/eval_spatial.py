"""Evaluate best SpatialModelV3 checkpoint on val split (có TTA)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.spatial_dl.dataset import SCORE_MAX, TARGET_SCALES, get_dataloaders
from src.spatial_dl.infer import encode_diff, load_spatial_checkpoint, predict_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/spatial_dl_improved/spatial_model_v3_best.pth")
    ap.add_argument("--data-path", default="poses/diffs")
    ap.add_argument("--csv-path", default="annotations/scores.csv")
    ap.add_argument("--train-csv", default="annotations/scores_train.csv")
    ap.add_argument("--val-csv", default="annotations/scores_test.csv")
    ap.add_argument("--tta", action="store_true", help="Bật test-time mirror augmentation")
    args = ap.parse_args()

    model, ckpt = load_spatial_checkpoint(args.ckpt)
    target_col = ckpt.get("target_col", "tong_diem")
    scale = float(ckpt.get("score_scale", TARGET_SCALES.get(target_col, SCORE_MAX)))
    use_tta = args.tta

    train_csv = args.train_csv if os.path.exists(args.train_csv) else ""
    val_csv = args.val_csv if os.path.exists(args.val_csv) else ""
    _, val_loader = get_dataloaders(
        batch_size=32,
        use_dummy=False,
        data_path=args.data_path,
        csv_path=args.csv_path,
        train_csv=train_csv,
        val_csv=val_csv,
        normalize_score=True,
        augment=False,
        target_col=target_col,
        score_max=scale,
    )

    preds, tgts, embeds = [], [], []
    for x, y in val_loader:
        p = predict_score(x, model, score_scale=scale, use_tta=use_tta, return_numpy=True)
        e = encode_diff(x, model, use_tta=use_tta, normalize=True, return_numpy=True)
        preds.append(np.asarray(p).reshape(-1))
        tgts.append(y.numpy().ravel() * scale)
        embeds.append(np.asarray(e).reshape(x.shape[0], -1))

    preds = np.concatenate(preds).ravel()
    tgts = np.concatenate(tgts).ravel()
    embeds = np.concatenate(embeds, axis=0)
    mae = float(np.mean(np.abs(preds - tgts)))
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    r = float(np.corrcoef(preds, tgts)[0, 1]) if len(preds) >= 3 else float("nan")

    print("=== Spatial DL eval (val) ===")
    print(
        f"checkpoint: {args.ckpt} (epoch {ckpt.get('epoch')}) "
        f"target={target_col} tta={use_tta}"
    )
    print(f"n={len(preds)}  MAE={mae:.2f}  RMSE={rmse:.2f}  Pearson={r:.3f}")
    print(f"embedding shape={embeds.shape}  L2_mean={np.linalg.norm(embeds, axis=1).mean():.4f}")

    sample = next(iter(val_loader))[0][:1]
    emb = encode_diff(sample.squeeze(0), model, use_tta=use_tta)
    print(f"encode_diff() → {tuple(np.asarray(emb).shape)} L2={np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()
