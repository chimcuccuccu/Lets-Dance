"""Evaluate best SpatialModelV3 checkpoint on val split."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.spatial_dl.dataset import SCORE_MAX, get_dataloaders
from src.spatial_dl.model_v3 import SpatialModelV3, spatial_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/spatial_dl_demo_full/spatial_model_v3_best.pth")
    ap.add_argument("--data-path", default="poses/diffs")
    ap.add_argument("--csv-path", default="annotations/scores.csv")
    ap.add_argument("--train-csv", default="annotations/scores_train.csv")
    ap.add_argument("--val-csv", default="annotations/scores_test.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    scale = float(ckpt.get("score_scale", SCORE_MAX))
    model = SpatialModelV3()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader = get_dataloaders(
        batch_size=32,
        use_dummy=False,
        data_path=args.data_path,
        csv_path=args.csv_path,
        train_csv=args.train_csv if os.path.exists(args.train_csv) else "",
        val_csv=args.val_csv if os.path.exists(args.val_csv) else "",
        normalize_score=True,
        augment=False,
    )

    preds, tgts, embeds = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            p, e = model(x)
            preds.append(p.numpy() * scale)
            tgts.append(y.numpy() * scale)
            embeds.append(e.numpy())

    preds = np.concatenate(preds).ravel()
    tgts = np.concatenate(tgts).ravel()
    embeds = np.concatenate(embeds, axis=0)
    mae = float(np.mean(np.abs(preds - tgts)))
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    r = float(np.corrcoef(preds, tgts)[0, 1]) if len(preds) >= 3 else float("nan")

    print("=== Spatial DL eval (val, person-split) ===")
    print(f"checkpoint: {args.ckpt} (epoch {ckpt.get('epoch')})")
    print(f"n={len(preds)}  MAE={mae:.2f}  RMSE={rmse:.2f}  Pearson={r:.3f}")
    print(f"embedding shape={embeds.shape}  L2_mean={np.linalg.norm(embeds, axis=1).mean():.4f}")

    sample = next(iter(val_loader))[0][:1]
    emb = spatial_model(sample.squeeze(0), model)
    print(f"spatial_model() → {tuple(emb.shape)} L2={emb.norm().item():.4f}")
    print("PASS" if mae < 40 and r > 0.2 else "WARN (metrics yếu nhưng pipeline chạy)")


if __name__ == "__main__":
    main()
