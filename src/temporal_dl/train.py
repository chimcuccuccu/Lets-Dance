"""
Person 2 — Training loops cho Temporal DL (Tuần 3-4).

Hai mode:
  python -m src.temporal_dl.train pretrain   → train genre classifier trên AIST++
  python -m src.temporal_dl.train train      → train BiLSTM chính trên dataset tự quay

Outputs:
  checkpoints/temporal_pretrained.pt               ← best pretrain checkpoint
  experiments/temporal_dl/pretrain_history.csv     ← lịch sử loss/acc pretrain
  experiments/temporal_dl/temporal_best.pth        ← best main model
  experiments/temporal_dl/train_history.csv        ← lịch sử loss/mae main training
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.temporal_dl.dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DTW_CSV,
    DEFAULT_POSES_DIR,
    get_aistpp_loaders,
    get_temporal_loaders,
)
from src.temporal_dl.lstm import (
    AISTPPPretrainModel,
    TemporalRegressionModel,
    load_pretrained_backbone,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "temporal_dl"
PRETRAIN_CKPT   = CHECKPOINTS_DIR / "temporal_pretrained.pt"
MAIN_CKPT       = EXPERIMENTS_DIR / "temporal_best.pth"


def _ensure_dirs() -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# PRETRAIN — Genre Classification trên AIST++
# ──────────────────────────────────────────────────────────────────────────────

def pretrain_aistpp(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    window_frames: int = 60,
    val_ratio: float = 0.15,
    max_files: int = 0,
    hidden_size: int = 64,
    embed_dim: int = 32,
    seed: int = 42,
) -> Path:
    """
    Pretrain TemporalBiLSTM trên tập AIST++ với proxy task phân loại genre.

    Returns đường dẫn đến checkpoint đã lưu.
    """
    _ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Pretrain device: %s", device)

    # Data
    train_loader, val_loader = get_aistpp_loaders(
        cache_dir=cache_dir,
        batch_size=batch_size,
        window_frames=window_frames,
        val_ratio=val_ratio,
        max_files=max_files,
        seed=seed,
    )

    # Model
    model = AISTPPPretrainModel(
        hidden_size=hidden_size,
        embed_dim=embed_dim,
    ).to(device)
    logger.info(
        "AISTPPPretrainModel params: %d",
        sum(p.numel() for p in model.parameters()),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    history_path = EXPERIMENTS_DIR / "pretrain_history.csv"
    best_val_acc = 0.0
    best_epoch   = 0

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ---- Train ----
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss   += loss.item() * x.size(0)
                preds         = logits.argmax(dim=1)
                train_correct += (preds == y).sum().item()
                train_total   += x.size(0)

            # ---- Val ----
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits    = model(x)
                    loss      = criterion(logits, y)
                    val_loss  += loss.item() * x.size(0)
                    preds      = logits.argmax(dim=1)
                    val_correct += (preds == y).sum().item()
                    val_total   += x.size(0)

            t_loss  = train_loss / max(train_total, 1)
            t_acc   = train_correct / max(train_total, 1)
            v_loss  = val_loss / max(val_total, 1)
            v_acc   = val_correct / max(val_total, 1)
            cur_lr  = optimizer.param_groups[0]["lr"]

            writer.writerow([epoch, f"{t_loss:.6f}", f"{t_acc:.4f}",
                             f"{v_loss:.6f}", f"{v_acc:.4f}", cur_lr])
            f.flush()

            logger.info(
                "Pretrain [%02d/%02d] %.1fs | loss %.4f→%.4f | acc %.2f%%→%.2f%%",
                epoch, epochs, time.time() - t0,
                t_loss, v_loss, t_acc * 100, v_acc * 100,
            )

            # Save best
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_epoch   = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": v_acc,
                        "config": {
                            "hidden_size": hidden_size,
                            "embed_dim": embed_dim,
                            "window_frames": window_frames,
                        },
                    },
                    PRETRAIN_CKPT,
                )

            scheduler.step()

    logger.info(
        "✅ Pretrain xong — best val_acc=%.2f%% (epoch %d) → %s",
        best_val_acc * 100, best_epoch, PRETRAIN_CKPT,
    )
    logger.info("History → %s", history_path)
    return PRETRAIN_CKPT


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN CHÍNH — BiLSTM trên dataset tự quay
# ──────────────────────────────────────────────────────────────────────────────

def train_temporal(
    dtw_csv: str | Path = DEFAULT_DTW_CSV,
    poses_dir: str | Path = DEFAULT_POSES_DIR,
    scores_csv: Optional[str | Path] = None,
    pretrain_ckpt: Optional[str | Path] = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    window_frames: int = 60,
    val_ratio: float = 0.15,
    hidden_size: int = 64,
    embed_dim: int = 32,
    seed: int = 42,
) -> Path:
    """
    Train TemporalRegressionModel trên dataset tự quay.

    Nếu có pretrain_ckpt, load backbone làm warm-start.
    Returns đường dẫn đến best checkpoint.
    """
    _ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Train device: %s", device)

    # Data
    train_loader, val_loader = get_temporal_loaders(
        dtw_csv=dtw_csv,
        poses_dir=poses_dir,
        scores_csv=scores_csv,
        batch_size=batch_size,
        window_frames=window_frames,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Model
    model = TemporalRegressionModel(
        hidden_size=hidden_size,
        embed_dim=embed_dim,
    ).to(device)

    if pretrain_ckpt is None and PRETRAIN_CKPT.is_file():
        pretrain_ckpt = PRETRAIN_CKPT
        logger.info("Auto-load pretrain checkpoint: %s", pretrain_ckpt)

    if pretrain_ckpt and Path(pretrain_ckpt).is_file():
        model = load_pretrained_backbone(model, pretrain_ckpt).to(device)

    logger.info(
        "TemporalRegressionModel params: %d",
        sum(p.numel() for p in model.parameters()),
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history_path = EXPERIMENTS_DIR / "train_history.csv"
    best_val_mae = float("inf")
    best_epoch   = 0

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_mae", "val_loss", "val_mae", "lr"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ---- Train ----
            model.train()
            train_loss = 0.0
            train_mae  = 0.0
            n_train    = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_mae  += (pred.detach() - y).abs().sum().item()
                n_train    += x.size(0)

            # ---- Val ----
            model.eval()
            val_loss = 0.0
            val_mae  = 0.0
            n_val    = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred     = model(x)
                    val_loss += criterion(pred, y).item() * x.size(0)
                    val_mae  += (pred - y).abs().sum().item()
                    n_val    += x.size(0)

            t_loss = train_loss / max(n_train, 1)
            t_mae  = train_mae  / max(n_train, 1)
            v_loss = val_loss   / max(n_val, 1)
            v_mae  = val_mae    / max(n_val, 1)
            cur_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([epoch, f"{t_loss:.6f}", f"{t_mae:.6f}",
                             f"{v_loss:.6f}", f"{v_mae:.6f}", cur_lr])
            f.flush()

            logger.info(
                "Train [%02d/%02d] %.1fs | loss %.4f→%.4f | mae %.4f→%.4f",
                epoch, epochs, time.time() - t0,
                t_loss, v_loss, t_mae, v_mae,
            )

            if v_mae < best_val_mae:
                best_val_mae = v_mae
                best_epoch   = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_mae": v_mae,
                        "config": {
                            "hidden_size": hidden_size,
                            "embed_dim": embed_dim,
                            "window_frames": window_frames,
                            "input_dim": 17,
                        },
                    },
                    MAIN_CKPT,
                )

            scheduler.step()

    logger.info(
        "✅ Training xong — best val_mae=%.4f (epoch %d) → %s",
        best_val_mae, best_epoch, MAIN_CKPT,
    )
    logger.info("History → %s", history_path)
    return MAIN_CKPT


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Person 2 — Temporal DL Training"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- pretrain ---
    pp = sub.add_parser("pretrain", help="Pretrain trên AIST++ (genre classification)")
    pp.add_argument("--cache-dir",  default=str(DEFAULT_CACHE_DIR))
    pp.add_argument("--epochs",     type=int,   default=20)
    pp.add_argument("--batch-size", type=int,   default=32)
    pp.add_argument("--lr",         type=float, default=1e-3)
    pp.add_argument("--window",     type=int,   default=60,
                    help="Số frame mỗi cửa sổ")
    pp.add_argument("--max-files",  type=int,   default=0,
                    help="Giới hạn số file AIST++ (0 = tất cả)")
    pp.add_argument("--hidden",     type=int,   default=64)
    pp.add_argument("--embed-dim",  type=int,   default=32)

    # --- train ---
    tp = sub.add_parser("train", help="Train BiLSTM chính trên dataset tự quay")
    tp.add_argument("--dtw-csv",        default=str(DEFAULT_DTW_CSV))
    tp.add_argument("--poses-dir",      default=str(DEFAULT_POSES_DIR))
    tp.add_argument("--scores-csv",     default=None)
    tp.add_argument("--pretrain-ckpt",  default=None)
    tp.add_argument("--epochs",         type=int,   default=30)
    tp.add_argument("--batch-size",     type=int,   default=16)
    tp.add_argument("--lr",             type=float, default=5e-4)
    tp.add_argument("--window",         type=int,   default=60)
    tp.add_argument("--hidden",         type=int,   default=64)
    tp.add_argument("--embed-dim",      type=int,   default=32)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "pretrain":
        pretrain_aistpp(
            cache_dir=args.cache_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            window_frames=args.window,
            max_files=args.max_files,
            hidden_size=args.hidden,
            embed_dim=args.embed_dim,
        )
    elif args.command == "train":
        train_temporal(
            dtw_csv=args.dtw_csv,
            poses_dir=args.poses_dir,
            scores_csv=args.scores_csv,
            pretrain_ckpt=args.pretrain_ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            window_frames=args.window,
            hidden_size=args.hidden,
            embed_dim=args.embed_dim,
        )


if __name__ == "__main__":
    main()
