"""Person 1 — Training loop Spatial DL (tuần 4)."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.spatial_dl.dataset import SCORE_MAX, get_dataloaders
from src.spatial_dl.model_v3 import SpatialModelV3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_metrics(model, loader, criterion, device, train: bool, optimizer=None, score_scale: float = SCORE_MAX):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0
    preds_all = []
    targets_all = []
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            # MAE trên thang 0-300
            total_mae += (torch.abs(preds.detach() - batch_y).sum().item() * score_scale)
            preds_all.append(preds.detach().cpu().numpy() * score_scale)
            targets_all.append(batch_y.detach().cpu().numpy() * score_scale)
            n += bs

    n = max(n, 1)
    preds_cat = np.concatenate(preds_all, axis=0).ravel() if preds_all else np.array([])
    targets_cat = np.concatenate(targets_all, axis=0).ravel() if targets_all else np.array([])
    pearson = float("nan")
    if len(preds_cat) >= 3 and np.std(preds_cat) > 1e-8 and np.std(targets_cat) > 1e-8:
        pearson = float(np.corrcoef(preds_cat, targets_cat)[0, 1])
    return total_loss / n, total_mae / n, pearson


def train_model(
    num_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    use_dummy: bool = False,
    use_old_data: bool = False,
    use_demo_data: bool = False,
    data_path: str = "poses/diffs",
    csv_path: str = "annotations/scores.csv",
    train_csv: str = "",
    val_csv: str = "",
    checkpoint_dir: str = "experiments/spatial_dl",
    seed: int = 42,
    target_frames: int = 150,
    alignment_dir: str = "annotations/dtw_paths",
    normalize_score: bool = True,
    augment: bool = True,
) -> SpatialModelV3:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        use_dummy=use_dummy,
        use_old_data=use_old_data,
        use_demo_data=use_demo_data,
        data_path=data_path,
        csv_path=csv_path,
        train_csv=train_csv,
        val_csv=val_csv,
        seed=seed,
        target_frames=target_frames,
        alignment_dir=alignment_dir,
        normalize_score=normalize_score,
        augment=augment,
    )

    score_scale = SCORE_MAX if normalize_score else 1.0
    model = SpatialModelV3().to(device)
    # SmoothL1 ổn định hơn MSE trên điểm hồi quy
    criterion = nn.SmoothL1Loss(beta=0.05 if normalize_score else 10.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "spatial_model_v3_best.pth")
    last_path = os.path.join(checkpoint_dir, "spatial_model_v3_last.pth")
    hist_path = os.path.join(checkpoint_dir, "train_history.csv")
    best_val = float("inf")

    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_mae", "train_pearson", "val_loss", "val_mae", "val_pearson", "lr"]
        )

    logger.info("Bắt đầu huấn luyện (%d epochs)...", num_epochs)
    for epoch in range(num_epochs):
        train_loss, train_mae, train_r = _epoch_metrics(
            model, train_loader, criterion, device, True, optimizer, score_scale
        )
        val_loss, val_mae, val_r = _epoch_metrics(
            model, val_loader, criterion, device, False, None, score_scale
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch [%d/%d] | Train loss=%.4f MAE=%.2f r=%.3f | Val loss=%.4f MAE=%.2f r=%.3f | lr=%.2e",
            epoch + 1,
            num_epochs,
            train_loss,
            train_mae,
            train_r,
            val_loss,
            val_mae,
            val_r,
            lr,
        )

        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch + 1, train_loss, train_mae, train_r, val_loss, val_mae, val_r, lr]
            )

        payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_pearson": val_r,
            "normalize_score": normalize_score,
            "score_scale": score_scale,
        }
        torch.save(payload, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, best_path)
            logger.info("  ↳ Saved best → %s (MAE=%.2f r=%.3f)", best_path, val_mae, val_r)

    logger.info("Xong. Best Val loss=%.4f | history=%s", best_val, hist_path)
    return model


def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Train SpatialModelV3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-path", type=str, default="poses/diffs")
    p.add_argument("--csv-path", type=str, default="annotations/scores.csv")
    p.add_argument("--train-csv", type=str, default="", help="Fixed train split CSV")
    p.add_argument("--val-csv", type=str, default="", help="Fixed val/test split CSV")
    p.add_argument("--alignment-dir", type=str, default="annotations/dtw_paths")
    p.add_argument("--checkpoint-dir", type=str, default="experiments/spatial_dl")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--raw-score", action="store_true", help="Không chia tong_diem/300")
    p.add_argument(
        "--demo-full",
        action="store_true",
        help="Train toàn bộ TikTok Demo với split official 183/34",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dummy", action="store_true")
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--old-data", action="store_true")
    mode.add_argument("--official", action="store_true", default=False)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    if args.demo_full:
        train_csv = train_csv or "annotations/scores_train.csv"
        val_csv = val_csv or "annotations/scores_test.csv"
        args.checkpoint_dir = (
            args.checkpoint_dir
            if args.checkpoint_dir != "experiments/spatial_dl"
            else "experiments/spatial_dl_demo_full"
        )
        logger.info(
            "Demo-full: train=%s val=%s → %s",
            train_csv,
            val_csv,
            args.checkpoint_dir,
        )

    # Mặc định official nếu có diffs; ngược lại dummy
    use_official = args.official or args.demo_full or (
        not args.dummy and not args.demo and not args.old_data
        and os.path.isdir(args.data_path)
        and (os.path.exists(args.csv_path) or (train_csv and val_csv))
    )
    if use_official and not args.official and not args.demo_full:
        logger.info("Phát hiện poses/diffs + scores.csv → train --official")

    train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_dummy=args.dummy or not (use_official or args.demo or args.old_data),
        use_old_data=args.old_data,
        use_demo_data=args.demo,
        data_path=args.data_path,
        csv_path=args.csv_path,
        train_csv=train_csv,
        val_csv=val_csv,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        alignment_dir=args.alignment_dir,
        normalize_score=not args.raw_score,
        augment=not args.no_aug,
    )
