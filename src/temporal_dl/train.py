"""
Person 2 — Training loops cho Temporal DL.

Hai mode:
  python -m src.temporal_dl.train pretrain   → train genre classifier trên AIST++
  python -m src.temporal_dl.train train      → train BiLSTM chính trên dataset tự quay

Outputs:
  checkpoints/temporal_pretrained.pt               ← best pretrain checkpoint
  experiments/temporal_dl/pretrain_history.csv     ← lịch sử loss/acc pretrain
  experiments/temporal_dl/temporal_{run_name}.pth  ← best main model
  experiments/temporal_dl/train_history_{run_name}.csv

Không có --run-name thì ghi ra tên mặc định (`temporal_best.pth`,
`train_history.csv`). LUÔN dùng --run-name cho các thí nghiệm mới để không ghi
đè lên kết quả đã commit.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Console Windows mặc định cp1252 → print() tiếng Việt sẽ UnicodeEncodeError.
# Ép UTF-8 để chạy được bằng `python -m ...` mà không cần set PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from src.temporal_dl.dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DTW_CSV,
    DEFAULT_POSES_DIR,
    DEFAULT_SCORES_CSV,
    DEFAULT_TRAIN_SCORES_CSV,
    DEFAULT_VAL_SCORES_CSV,
    SELF_REFERENTIAL_TARGETS,
    get_aistpp_loaders,
    get_temporal_loaders,
    get_temporal_loaders_grouped,
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

def _video_level_mae(
    idx_list: np.ndarray,
    preds: np.ndarray,
    ds,
    target_range: Tuple[float, float],
) -> float:
    """
    MAE ở **cấp video**, thang điểm gốc.

    Gộp prediction của các cửa sổ thuộc cùng một video (trung bình), de-normalize,
    rồi so với nhãn thật của video. Đây là con số duy nhất so sánh được với
    baseline global-mean / dance-mean — `val_mae` theo cửa sổ và đã normalize thì
    không.
    """
    tmin, tmax = target_range
    span = (tmax - tmin) or 1.0
    by_video: Dict[int, list] = defaultdict(list)
    truth: Dict[int, float] = {}
    for i, p in zip(idx_list, preds):
        m = ds.meta[int(i)]
        by_video[m.video_index].append(float(p))
        truth[m.video_index] = m.target_raw
    if not by_video:
        return float("nan")
    errs = [
        abs((float(np.mean(ps)) * span + tmin) - truth[v])
        for v, ps in by_video.items()
    ]
    return float(np.mean(errs))


def train_temporal(
    dtw_csv: str | Path = DEFAULT_DTW_CSV,
    poses_dir: str | Path = DEFAULT_POSES_DIR,
    scores_csv: Optional[str | Path] = DEFAULT_SCORES_CSV,
    pretrain_ckpt: Optional[str | Path] = None,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 5e-4,
    window_frames: int = 60,
    val_ratio: float = 0.2,
    hidden_size: int = 64,
    embed_dim: int = 32,
    seed: int = 42,
    *,
    target_col: str = "khop_nhip",
    split: str = "official",
    train_csv: str | Path = DEFAULT_TRAIN_SCORES_CSV,
    val_csv: str | Path = DEFAULT_VAL_SCORES_CSV,
    fold: Optional[int] = None,
    n_folds: int = 5,
    use_dtw: bool = True,
    hop_frames: Optional[int] = None,
    augment: bool = True,
    early_stop_patience: int = 12,
    run_name: Optional[str] = None,
    allow_dtw_target: bool = False,
) -> Path:
    """
    Train TemporalRegressionModel trên dataset tự quay.

    Mặc định (Tuần 5): target = `khop_nhip`, split official 183/34 rời nhau theo
    person_id. Đường legacy cần `split="random"` +
    `target_col="dtw_distance_total"` + `allow_dtw_target=True`.

    Nếu có pretrain_ckpt, load backbone làm warm-start.
    Returns đường dẫn đến best checkpoint.
    """
    _ensure_dirs()

    # ---- Guard target tự tham chiếu (xem docs/temporal_eval.md §1.2) ----
    if target_col in SELF_REFERENTIAL_TARGETS and use_dtw and not allow_dtw_target:
        raise SystemExit(
            f"Target {target_col!r} cũng nằm trong input (kênh cuối, index 16) "
            f"→ model tự tham chiếu, val_mae vô nghĩa.\n"
            f"Dùng --target-col khop_nhip, hoặc --allow-dtw-target nếu cố ý "
            f"đối chiếu lại cách cũ."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Train device: %s", device)

    # ---- Data ----
    if split == "random":
        # Đường legacy — có leakage cấp window, chỉ để đối chiếu.
        train_loader, val_loader = get_temporal_loaders(
            dtw_csv=dtw_csv, poses_dir=poses_dir, scores_csv=scores_csv,
            batch_size=batch_size, window_frames=window_frames,
            val_ratio=val_ratio, seed=seed,
        )
        meta: Dict[str, Any] = {
            "split": "random", "target_col": target_col, "input_dim": 17,
            "use_dtw": use_dtw, "window_frames": window_frames,
            "hop_frames": window_frames, "seed": seed,
            "target_range": list(train_loader.dataset.dataset.target_range),
        }
    else:
        train_loader, val_loader, meta = get_temporal_loaders_grouped(
            dtw_csv=dtw_csv, poses_dir=poses_dir, scores_csv=scores_csv,
            target_col=target_col, split=split,
            train_csv=train_csv, val_csv=val_csv, fold=fold, n_folds=n_folds,
            val_ratio=val_ratio, batch_size=batch_size,
            window_frames=window_frames, hop_frames=hop_frames,
            use_dtw=use_dtw, augment=augment, seed=seed,
        )

    ds = val_loader.dataset.dataset          # TemporalSeqDataset gốc
    ds.return_index = True                   # cần idx để gộp theo video
    target_range = tuple(meta["target_range"])
    span = (target_range[1] - target_range[0]) or 1.0

    # ---- Model ----
    model = TemporalRegressionModel(
        input_dim=int(meta["input_dim"]),
        hidden_size=hidden_size,
        embed_dim=embed_dim,
    ).to(device)

    if pretrain_ckpt is None and PRETRAIN_CKPT.is_file():
        pretrain_ckpt = PRETRAIN_CKPT
        logger.info("Auto-load pretrain checkpoint: %s", pretrain_ckpt)

    if pretrain_ckpt and Path(pretrain_ckpt).is_file():
        model = load_pretrained_backbone(model, pretrain_ckpt).to(device)

    logger.info(
        "TemporalRegressionModel params: %d | input_dim=%d",
        sum(p.numel() for p in model.parameters()), model.input_dim,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    tag = run_name or ""
    ckpt_path = (EXPERIMENTS_DIR / f"temporal_{tag}.pth") if tag else MAIN_CKPT
    history_path = (
        EXPERIMENTS_DIR / f"train_history_{tag}.csv" if tag
        else EXPERIMENTS_DIR / "train_history.csv"
    )

    best_metric = float("inf")
    best_epoch = 0
    best_row: Dict[str, float] = {}
    epochs_since_best = 0

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_mae", "val_loss", "val_mae",
            "val_mae_raw", "val_mae_video_raw", "lr",
        ])

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ---- Train ----
            model.train()
            train_loss = train_mae = 0.0
            n_train = 0
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_mae += (pred.detach() - y).abs().sum().item()
                n_train += x.size(0)

            # ---- Val ----
            model.eval()
            val_loss = val_mae = 0.0
            n_val = 0
            all_idx, all_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    pred = model(x)
                    val_loss += criterion(pred, y).item() * x.size(0)
                    val_mae += (pred - y).abs().sum().item()
                    n_val += x.size(0)
                    if len(batch) > 2:
                        all_idx.append(batch[2].cpu().numpy())
                        all_pred.append(pred.cpu().numpy())

            t_loss = train_loss / max(n_train, 1)
            t_mae = train_mae / max(n_train, 1)
            v_loss = val_loss / max(n_val, 1)
            v_mae = val_mae / max(n_val, 1)
            v_mae_raw = v_mae * span
            v_mae_video = (
                _video_level_mae(
                    np.concatenate(all_idx), np.concatenate(all_pred), ds, target_range
                )
                if all_idx else float("nan")
            )
            cur_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch, f"{t_loss:.6f}", f"{t_mae:.6f}", f"{v_loss:.6f}",
                f"{v_mae:.6f}", f"{v_mae_raw:.4f}", f"{v_mae_video:.4f}", cur_lr,
            ])
            f.flush()

            logger.info(
                "Train [%02d/%02d] %.1fs | loss %.4f→%.4f | val_mae %.4f "
                "(=%.2f điểm) | val_mae_video %.3f điểm",
                epoch, epochs, time.time() - t0, t_loss, v_loss,
                v_mae, v_mae_raw, v_mae_video,
            )

            # Early stop + save theo MAE cấp video (số có nghĩa), fallback window
            metric = v_mae_video if np.isfinite(v_mae_video) else v_mae
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                epochs_since_best = 0
                best_row = {
                    "val_mae": v_mae, "val_mae_raw": v_mae_raw,
                    "val_mae_video_raw": v_mae_video,
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_mae": v_mae,
                        "val_mae_raw": v_mae_raw,
                        "val_mae_video_raw": v_mae_video,
                        "config": {
                            "hidden_size": hidden_size,
                            "embed_dim": embed_dim,
                            **meta,
                        },
                    },
                    ckpt_path,
                )
            else:
                epochs_since_best += 1
                if early_stop_patience and epochs_since_best >= early_stop_patience:
                    logger.info(
                        "Early stop ở epoch %d (không cải thiện %d epoch)",
                        epoch, epochs_since_best,
                    )
                    break

            scheduler.step()

    logger.info(
        "✅ Training xong — best epoch %d | val_mae=%.4f (=%.2f điểm) | "
        "val_mae_video_raw=%.3f điểm",
        best_epoch, best_row.get("val_mae", float("nan")),
        best_row.get("val_mae_raw", float("nan")),
        best_row.get("val_mae_video_raw", float("nan")),
    )
    logger.info("Đã ghi: %s", ckpt_path)
    logger.info("Đã ghi: %s", history_path)
    return ckpt_path


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
    tp.add_argument("--scores-csv",     default=str(DEFAULT_SCORES_CSV))
    tp.add_argument("--no-scores",      action="store_true",
                    help="Không dùng file nhãn (đường legacy). Dùng cờ này "
                         "thay cho --scores-csv \"\" vì PowerShell nuốt chuỗi rỗng")
    tp.add_argument("--pretrain-ckpt",  default=None)
    tp.add_argument("--epochs",         type=int,   default=60)
    tp.add_argument("--batch-size",     type=int,   default=16)
    tp.add_argument("--lr",             type=float, default=5e-4)
    tp.add_argument("--window",         type=int,   default=60)
    tp.add_argument("--hidden",         type=int,   default=64)
    tp.add_argument("--embed-dim",      type=int,   default=32)
    tp.add_argument("--seed",           type=int,   default=42)
    # --- Tuần 5 ---
    tp.add_argument("--target-col",     default="khop_nhip",
                    help="Cột nhãn trong scores.csv (mặc định khop_nhip)")
    tp.add_argument("--split",          default="official",
                    choices=["official", "person", "groupkfold", "random"],
                    help="Cách chia train/val. 'random' = legacy (có leakage)")
    tp.add_argument("--train-csv",      default=str(DEFAULT_TRAIN_SCORES_CSV))
    tp.add_argument("--val-csv",        default=str(DEFAULT_VAL_SCORES_CSV))
    tp.add_argument("--fold",           type=int,   default=None,
                    help="Fold thứ mấy (bắt buộc với --split groupkfold)")
    tp.add_argument("--n-folds",        type=int,   default=5)
    tp.add_argument("--val-ratio",      type=float, default=0.2)
    tp.add_argument("--no-dtw",         action="store_true",
                    help="Bỏ kênh dtw → input 16 chiều (kịch bản (b) LSTM-only)")
    tp.add_argument("--hop",            type=int,   default=None,
                    help="Hop giữa các cửa sổ lúc train (mặc định = --window)")
    tp.add_argument("--no-aug",         action="store_true")
    tp.add_argument("--patience",       type=int,   default=12,
                    help="Early stop theo val_mae_video_raw (0 = tắt)")
    tp.add_argument("--run-name",       default=None,
                    help="Hậu tố tên output. Bỏ trống = ghi vào tên mặc định")
    tp.add_argument("--allow-dtw-target", action="store_true",
                    help="Cho phép target dtw_distance_total (tự tham chiếu)")

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
            scores_csv=None if args.no_scores else (args.scores_csv or None),
            pretrain_ckpt=args.pretrain_ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            window_frames=args.window,
            val_ratio=args.val_ratio,
            hidden_size=args.hidden,
            embed_dim=args.embed_dim,
            seed=args.seed,
            target_col=args.target_col,
            split=args.split,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            fold=args.fold,
            n_folds=args.n_folds,
            use_dtw=not args.no_dtw,
            hop_frames=args.hop,
            augment=not args.no_aug,
            early_stop_patience=args.patience,
            run_name=args.run_name,
            allow_dtw_target=args.allow_dtw_target,
        )


if __name__ == "__main__":
    main()
