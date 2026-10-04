"""Person 1 — Training loop Spatial DL (freeze→unfreeze, aux joint-group loss)."""
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
import torch.nn.functional as F
import torch.optim as optim

from src.spatial_dl.dataset import (
    JOINT_GROUP_LOSS_WEIGHTS,
    SCORE_MAX,
    TARGET_SCALES,
    get_dataloaders,
    joint_group_abs_means,
)
from src.spatial_dl.model_v3 import DEFAULT_EMBED_SIZE, SpatialModelV3, load_pretrained_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_GROUP_ORDER = ("face", "arms", "torso", "legs")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_weight_tensor(device: torch.device) -> torch.Tensor:
    w = torch.tensor(
        [JOINT_GROUP_LOSS_WEIGHTS[g] for g in _GROUP_ORDER],
        dtype=torch.float32,
        device=device,
    )
    return w / w.sum()


def _combined_loss(
    model: SpatialModelV3,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    score_criterion: nn.Module,
    *,
    aux_weight: float,
    group_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score SmoothL1 + weighted aux trên mean |diff| theo nhóm khớp."""
    if aux_weight > 0 and model.aux_fc is not None:
        preds, _, aux = model(batch_x, return_aux=True)
        score_loss = score_criterion(preds, batch_y)
        target_aux = joint_group_abs_means(batch_x)
        # per-group SmoothL1, rồi trọng số face/arms/torso/legs
        per = F.smooth_l1_loss(aux, target_aux, beta=0.05, reduction="none")
        aux_loss = (per * group_w.unsqueeze(0)).sum(dim=1).mean()
        return score_loss + aux_weight * aux_loss, preds
    preds, _ = model(batch_x)
    return score_criterion(preds, batch_y), preds


def _epoch_metrics(
    model,
    loader,
    criterion,
    device,
    train: bool,
    optimizer=None,
    score_scale: float = SCORE_MAX,
    aux_weight: float = 0.0,
    group_w: Optional[torch.Tensor] = None,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0
    preds_all = []
    targets_all = []
    n = 0
    if group_w is None:
        group_w = _group_weight_tensor(device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            loss, preds = _combined_loss(
                model,
                batch_x,
                batch_y,
                criterion,
                aux_weight=aux_weight if train else 0.0,
                group_w=group_w,
            )
            # Val: chỉ score loss để early-stop so sánh được
            if not train:
                loss = criterion(preds, batch_y)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=5.0
                )
                optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_mae += torch.abs(preds.detach() - batch_y).sum().item() * score_scale
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


def _build_optimizer(
    model: SpatialModelV3,
    lr_head: float,
    lr_encoder: float,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    if model.aux_fc is not None:
        head_params = head_params + [p for p in model.aux_fc.parameters() if p.requires_grad]
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_encoder})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    if not groups:
        groups = [{"params": model.parameters(), "lr": lr_head}]
    return optim.AdamW(groups, weight_decay=weight_decay)


def train_model(
    num_epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    encoder_lr: float = 1e-4,
    freeze_epochs: int = 5,
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
    pretrained_path: str = "",
    target_col: str = "khop_dong_tac",
    embed_size: int = DEFAULT_EMBED_SIZE,
    use_lstm: bool = True,
    early_stopping_patience: int = 10,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    aux_weight: float = 0.2,
) -> SpatialModelV3:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_max = TARGET_SCALES.get(target_col, SCORE_MAX)
    logger.info(
        "Device: %s | target=%s scale=%.0f | freeze_epochs=%d | aux_weight=%.2f",
        device,
        target_col,
        score_max,
        freeze_epochs,
        aux_weight,
    )

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
        target_col=target_col,
        score_max=score_max,
    )

    score_scale = score_max if normalize_score else 1.0
    model = SpatialModelV3(
        embed_size=embed_size,
        use_lstm=use_lstm,
        dropout=dropout,
        use_aux_head=aux_weight > 0,
    ).to(device)
    if pretrained_path:
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained checkpoint không tồn tại: {pretrained_path}"
            )
        load_pretrained_encoder(model, pretrained_path, map_location=device, strict=False)
        logger.info("Fine-tune từ pretrain: %s", pretrained_path)

    criterion = nn.SmoothL1Loss(beta=0.05 if normalize_score else max(1.0, score_max * 0.05))
    group_w = _group_weight_tensor(device)

    frozen = bool(pretrained_path and freeze_epochs > 0)
    if frozen:
        model.freeze_encoder()
        logger.info("Phase A: freeze encoder %d epochs | head lr=%.2e", freeze_epochs, learning_rate)
    optimizer = _build_optimizer(
        model,
        lr_head=learning_rate,
        lr_encoder=encoder_lr if not frozen else 0.0,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "spatial_model_v3_best.pth")
    last_path = os.path.join(checkpoint_dir, "spatial_model_v3_last.pth")
    hist_path = os.path.join(checkpoint_dir, "train_history.csv")
    best_val = float("inf")
    epochs_without_improve = 0

    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "train_loss",
                "train_mae",
                "train_pearson",
                "val_loss",
                "val_mae",
                "val_pearson",
                "lr_head",
                "phase",
            ]
        )

    logger.info("Bắt đầu huấn luyện (%d epochs)...", num_epochs)
    for epoch in range(num_epochs):
        if frozen and epoch == freeze_epochs:
            model.unfreeze_encoder()
            frozen = False
            optimizer = _build_optimizer(
                model,
                lr_head=learning_rate,
                lr_encoder=encoder_lr,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
            epochs_without_improve = 0
            logger.info(
                "Phase B: unfreeze encoder | head_lr=%.2e enc_lr=%.2e",
                learning_rate,
                encoder_lr,
            )

        phase = "freeze" if frozen else "full"
        train_loss, train_mae, train_r = _epoch_metrics(
            model,
            train_loader,
            criterion,
            device,
            True,
            optimizer,
            score_scale,
            aux_weight=aux_weight,
            group_w=group_w,
        )
        val_loss, val_mae, val_r = _epoch_metrics(
            model,
            val_loader,
            criterion,
            device,
            False,
            None,
            score_scale,
            aux_weight=0.0,
            group_w=group_w,
        )
        scheduler.step(val_loss)
        lr_head = optimizer.param_groups[-1]["lr"]

        logger.info(
            "Epoch [%d/%d] %s | Train loss=%.4f MAE=%.2f r=%.3f | Val loss=%.4f MAE=%.2f r=%.3f | lr=%.2e",
            epoch + 1,
            num_epochs,
            phase,
            train_loss,
            train_mae,
            train_r,
            val_loss,
            val_mae,
            val_r,
            lr_head,
        )

        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_mae,
                    train_r,
                    val_loss,
                    val_mae,
                    val_r,
                    lr_head,
                    phase,
                ]
            )

        payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_pearson": val_r,
            "normalize_score": normalize_score,
            "score_scale": score_scale,
            "target_col": target_col,
            "embed_size": embed_size,
            "use_lstm": use_lstm,
            "pretrained_path": pretrained_path or "",
            "aux_weight": aux_weight,
            "dual_embedding": True,
        }
        torch.save(payload, last_path)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_without_improve = 0
            torch.save(payload, best_path)
            logger.info("  ↳ Saved best → %s (MAE=%.2f r=%.3f)", best_path, val_mae, val_r)
        else:
            epochs_without_improve += 1
            if (
                early_stopping_patience > 0
                and epochs_without_improve >= early_stopping_patience
            ):
                logger.info(
                    "Early stop @ epoch %d (patience=%d, best_val=%.4f)",
                    epoch + 1,
                    early_stopping_patience,
                    best_val,
                )
                break

    logger.info("Xong. Best Val loss=%.4f | history=%s", best_val, hist_path)
    return model


def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Train SpatialModelV3 (improved)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="LR cho score head")
    p.add_argument("--encoder-lr", type=float, default=1e-4, help="LR encoder sau unfreeze")
    p.add_argument("--freeze-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-path", type=str, default="poses/diffs")
    p.add_argument("--csv-path", type=str, default="annotations/scores.csv")
    p.add_argument("--train-csv", type=str, default="")
    p.add_argument("--val-csv", type=str, default="")
    p.add_argument("--alignment-dir", type=str, default="annotations/dtw_paths")
    p.add_argument("--checkpoint-dir", type=str, default="experiments/spatial_dl")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--raw-score", action="store_true")
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument(
        "--target",
        type=str,
        default="khop_dong_tac",
        choices=list(TARGET_SCALES.keys()),
        help="Cột điểm target (Spatial nên dùng khop_dong_tac)",
    )
    p.add_argument("--embed-size", type=int, default=DEFAULT_EMBED_SIZE)
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--early-stop", type=int, default=10, help="Patience early stopping (0=tắt)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument(
        "--aux-weight",
        type=float,
        default=0.2,
        help="Trọng số aux loss joint-group (0=tắt)",
    )
    p.add_argument("--improved", action="store_true", help="Preset: freeze5 + khop_dong_tac + pretrained")
    p.add_argument("--demo-full", action="store_true")
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
    if args.improved:
        args.pretrained = args.pretrained or "checkpoints/spatial_pretrained.pt"
        args.freeze_epochs = max(args.freeze_epochs, 5)
        args.target = args.target or "khop_dong_tac"
        if args.weight_decay == 1e-4:
            args.weight_decay = 5e-4
        if args.dropout == 0.3:
            args.dropout = 0.4
        if args.checkpoint_dir == "experiments/spatial_dl":
            args.checkpoint_dir = "experiments/spatial_dl_improved"
        if not train_csv and os.path.exists("annotations/scores_train.csv"):
            train_csv = "annotations/scores_train.csv"
        if not val_csv and os.path.exists("annotations/scores_test.csv"):
            val_csv = "annotations/scores_test.csv"
        logger.info("Improved preset → ckpt_dir=%s pretrained=%s", args.checkpoint_dir, args.pretrained)

    if args.demo_full:
        train_csv = train_csv or "annotations/scores_train.csv"
        val_csv = val_csv or "annotations/scores_test.csv"
        if args.checkpoint_dir == "experiments/spatial_dl":
            args.checkpoint_dir = "experiments/spatial_dl_demo_full"

    use_official = args.official or args.demo_full or args.improved or (
        not args.dummy
        and not args.demo
        and not args.old_data
        and os.path.isdir(args.data_path)
        and (os.path.exists(args.csv_path) or (train_csv and val_csv))
    )

    train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        encoder_lr=args.encoder_lr,
        freeze_epochs=args.freeze_epochs,
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
        pretrained_path=args.pretrained,
        target_col=args.target,
        embed_size=args.embed_size,
        use_lstm=not args.no_lstm,
        early_stopping_patience=args.early_stop,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        aux_weight=args.aux_weight,
    )
