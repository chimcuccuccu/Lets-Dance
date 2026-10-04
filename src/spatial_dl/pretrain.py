"""
Person 1 — AIST++ pretrain for SpatialEncoder (proxy autoencoder).

Default proxy: reconstruct synthetic same-genre diffs so the encoder matches
the fine-tune input domain (performer − reference).

Checkpoint (deliverable)::

    checkpoints/spatial_pretrained.pt
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.spatial_dl.aist_dataset import get_aist_dataloaders
from src.spatial_dl.model_v3 import DEFAULT_EMBED_SIZE, SpatialAutoEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CKPT = "checkpoints/spatial_pretrained.pt"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    joint_weight: Optional[torch.Tensor] = None,
    beta: float = 0.05,
) -> torch.Tensor:
    """Smooth L1 on XYZ; optional per-joint weights (prefer COCO-mapped joints)."""
    err = nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    if joint_weight is None:
        return err.mean()
    w = joint_weight.view(1, 1, -1, 1).to(pred.device, pred.dtype)
    denom = w.sum() * pred.shape[0] * pred.shape[1] * pred.shape[3] + 1e-8
    return (err * w).sum() / denom


def _epoch(
    model: SpatialAutoEncoder,
    loader,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    joint_weight: Optional[torch.Tensor],
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            recon, _ = model(x)
            loss = _masked_recon_loss(recon, y, joint_weight)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = x.size(0)
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


def pretrain(
    aist_root: str = "data/aistpp",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    mode: str = "synthetic_diff",
    target_frames: int = 150,
    window_frames: int = 180,
    seed: int = 42,
    max_sequences: int = 0,
    checkpoint_path: str = DEFAULT_CKPT,
    history_path: str = "",
    weight_mapped_joints: bool = True,
    source: str = "keypoints3d",
    shared_dir: str = "",
    camera: str = "all",
    embed_size: int = DEFAULT_EMBED_SIZE,
    use_lstm: bool = True,
) -> SpatialAutoEncoder:
    from src.spatial_dl.aist_dataset import DEFAULT_SHARED_BASIC
    from src.spatial_dl.joints_convert import mapped_joint_mask

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_dir = shared_dir or DEFAULT_SHARED_BASIC
    logger.info(
        "AIST pretrain | device=%s mode=%s source=%s camera=%s embed=%d lstm=%s",
        device,
        mode,
        source,
        camera,
        embed_size,
        use_lstm,
    )

    train_loader, val_loader = get_aist_dataloaders(
        aist_root=aist_root,
        batch_size=batch_size,
        mode=mode,
        target_frames=target_frames,
        window_frames=window_frames,
        seed=seed,
        max_sequences=max_sequences,
        build_cache=True,
        source=source,
        shared_dir=shared_dir,
        camera=camera,
    )

    model = SpatialAutoEncoder(embed_size=embed_size, use_lstm=use_lstm).to(device)
    joint_w = None
    # COCO→MP heuristic weights only for pure keypoints3d (mix/shared = full MP / mixed)
    use_joint_w = weight_mapped_joints and source in ("keypoints3d", "kpt3d")
    if use_joint_w:
        mask = mapped_joint_mask("coco17").astype(np.float32)
        # Mapped joints weight 1.0; heuristics 0.25 (still train, less noise)
        joint_w = torch.from_numpy(np.where(mask, 1.0, 0.25).astype(np.float32))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    hist = history_path or os.path.join(
        os.path.dirname(checkpoint_path) or "checkpoints", "spatial_pretrain_history.csv"
    )
    with open(hist, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        train_loss = _epoch(model, train_loader, optimizer, device, joint_w)
        val_loss = _epoch(model, val_loader, None, device, joint_w)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - ep_t0
        logger.info(
            "Epoch [%d/%d] train=%.5f val=%.5f lr=%.2e (%.1fs)",
            epoch,
            epochs,
            train_loss,
            val_loss,
            lr_now,
            elapsed,
        )
        with open(hist, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, lr_now, elapsed])

        payload = {
            "epoch": epoch,
            "encoder_state_dict": model.encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "train_loss": train_loss,
            "pretrain_mode": mode,
            "aist_root": aist_root,
            "source": source,
            "shared_dir": shared_dir if "shared" in source or source == "mix" else "",
            "camera": camera,
            "embed_size": embed_size,
            "use_lstm": use_lstm,
            "target_frames": target_frames,
            "window_frames": window_frames,
            "seed": seed,
        }
        torch.save(payload, checkpoint_path.replace(".pt", "_last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, checkpoint_path)
            logger.info("  ↳ best → %s (val=%.5f)", checkpoint_path, val_loss)

    logger.info(
        "Pretrain done in %.1f min | best_val=%.5f | ckpt=%s",
        (time.time() - t0) / 60.0,
        best_val,
        checkpoint_path,
    )
    return model


def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="AIST++ Spatial encoder pretrain")
    p.add_argument("--aist-root", default="data/aistpp")
    p.add_argument(
        "--source",
        choices=("keypoints3d", "shared", "mix"),
        default="keypoints3d",
        help="keypoints3d | shared | mix (shared+keypoints3d)",
    )
    p.add_argument("--shared-dir", default="")
    p.add_argument("--camera", default="all", help="shared/mix: all | c01 | ...")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mode", choices=("synthetic_diff", "pose_ae"), default="synthetic_diff")
    p.add_argument("--target-frames", type=int, default=150)
    p.add_argument("--window-frames", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sequences", type=int, default=0)
    p.add_argument("--embed-size", type=int, default=DEFAULT_EMBED_SIZE)
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument(
        "--improved",
        action="store_true",
        help="Preset: mix + camera c01 + 40 epochs + embed 256 + LSTM",
    )
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--build-cache-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.improved:
        args.source = "mix"
        args.camera = "c01" if args.camera == "all" else args.camera
        args.epochs = max(args.epochs, 40)
        args.checkpoint = args.checkpoint or DEFAULT_CKPT
        logger.info(
            "Improved pretrain preset: source=%s camera=%s epochs=%d",
            args.source,
            args.camera,
            args.epochs,
        )

    if args.build_cache_only:
        from src.spatial_dl.aist_dataset import get_aist_dataloaders

        get_aist_dataloaders(
            aist_root=args.aist_root,
            source=args.source,
            shared_dir=args.shared_dir or "",
            camera=args.camera,
            max_sequences=args.max_sequences,
            build_cache=True,
            batch_size=1,
        )
        logger.info("Cache build done (source=%s)", args.source)
    else:
        pretrain(
            aist_root=args.aist_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            mode=args.mode,
            target_frames=args.target_frames,
            window_frames=args.window_frames,
            seed=args.seed,
            max_sequences=args.max_sequences,
            checkpoint_path=args.checkpoint,
            source=args.source,
            shared_dir=args.shared_dir,
            camera=args.camera,
            embed_size=args.embed_size,
            use_lstm=not args.no_lstm,
        )
