"""Person 3 — vòng train Error DL (multi-label BCE từng head)."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from src.error_dl.dataset import (
    ERROR_TYPES,
    get_error_dataloaders,
    positive_weights,
    resolve_project_path,
)
from src.error_dl.error_model import ErrorModel, binary_cross_entropy_per_head

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_f1(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """F1 từng lớp. Lớp không có mẫu dương trong batch đánh giá → nan."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    tp = (pred * target).sum(axis=0)
    fp = (pred * (1.0 - target)).sum(axis=0)
    fn = ((1.0 - pred) * target).sum(axis=0)
    support = target.sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1e-8)
    recall = tp / np.maximum(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-8)
    f1 = np.where(support > 0, f1, np.nan)
    return f1.astype(np.float64)


def _run_epoch(model, loader, device, optimizer, pos_weight, train: bool):
    model.train(train)
    total_loss = 0.0
    n = 0
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, _ = model(batch_x)
            loss = binary_cross_entropy_per_head(logits, batch_y, pos_weight=pos_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = batch_x.shape[0]
            total_loss += float(loss.item()) * bs
            n += bs
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_all.append(batch_y.detach().cpu().numpy())
    if n == 0:
        raise RuntimeError("DataLoader rỗng")
    probs = np.concatenate(probs_all, axis=0)
    target = np.concatenate(y_all, axis=0)
    f1 = _binary_f1((probs >= 0.5).astype(np.float32), target)
    valid = f1[~np.isnan(f1)]
    macro = float(valid.mean()) if len(valid) else 0.0
    return total_loss / n, macro, f1


def _fmt_f1(f1: np.ndarray) -> str:
    parts = []
    for name, value in zip(ERROR_TYPES, f1):
        shown = "n/a" if np.isnan(value) else f"{value:.3f}"
        parts.append(f"{name}={shown}")
    return " ".join(parts)


def train_model(
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    diffs_dir: str = "poses/diffs",
    labels_csv: str = "annotations/error_labels.csv",
    windows_csv: str = "annotations/suggestions.csv",
    scores_csv: str = "annotations/scores.csv",
    poses_dir: str = "poses",
    checkpoint_dir: str = "experiments/error_dl",
    window_frames: int = 32,
    embed_size: int = 64,
    encoder: str = "cnn",
    dropout: float = 0.1,
    val_ratio: float = 0.2,
    use_pos_weight: bool = True,
    synthetic: bool = False,
    patience: int = 8,
) -> ErrorModel:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, dataset, train_idx, _val_idx = get_error_dataloaders(
        diffs_dir=diffs_dir,
        labels_csv=labels_csv,
        windows_csv=windows_csv,
        scores_csv=scores_csv,
        poses_dir=poses_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        window_frames=window_frames,
        synthetic=synthetic,
    )
    model = ErrorModel(
        embed_size=embed_size,
        window_frames=window_frames,
        encoder=encoder,
        dropout=dropout,
    ).to(device)
    sample_x, sample_y = next(iter(train_loader))
    with torch.no_grad():
        logits, emb = model(sample_x.to(device))
    logger.info(
        "Batch x=%s y=%s logits=%s emb=%s | device=%s encoder=%s",
        tuple(sample_x.shape),
        tuple(sample_y.shape),
        tuple(logits.shape),
        tuple(emb.shape),
        device,
        encoder,
    )

    pos_weight = None
    if use_pos_weight:
        weights = positive_weights(dataset.targets[train_idx])
        pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        logger.info(
            "BCE pos_weight: %s",
            ", ".join(f"{name}={weights[i]:.2f}" for i, name in enumerate(ERROR_TYPES)),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    out_dir = resolve_project_path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "train_history.csv"
    best_path = out_dir / "error_model_best.pt"
    last_path = out_dir / "error_model_last.pt"
    with hist_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            ["epoch", "train_loss", "val_loss", "train_macro_f1", "val_macro_f1", *_f1_cols("train"), *_f1_cols("val")]
        )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    best_val_loss = None
    best_per_class: Optional[np.ndarray] = None
    stale = 0
    history: List[Dict] = []

    for epoch in range(num_epochs):
        train_loss, train_f1, train_per = _run_epoch(
            model, train_loader, device, optimizer, pos_weight, train=True
        )
        val_loss, val_f1, val_per = _run_epoch(
            model, val_loader, device, optimizer, pos_weight, train=False
        )
        logger.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%.4f train_f1=%.3f val_f1=%.3f | %s",
            epoch + 1,
            num_epochs,
            train_loss,
            val_loss,
            train_f1,
            val_f1,
            _fmt_f1(val_per),
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_macro_f1": train_f1,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        with hist_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    train_loss,
                    val_loss,
                    train_f1,
                    val_f1,
                    *[_csv_f1(v) for v in train_per],
                    *[_csv_f1(v) for v in val_per],
                ]
            )
        payload = _checkpoint_payload(
            model,
            epoch=epoch + 1,
            val_loss=val_loss,
            val_macro_f1=val_f1,
            per_class_f1=val_per,
            window_frames=window_frames,
            embed_size=embed_size,
            encoder=encoder,
            dropout=dropout,
            synthetic=synthetic,
        )
        torch.save(payload, last_path)
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_per_class = val_per.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(payload, best_path)
            logger.info("  Saved best → %s (val_macro_f1=%.3f)", best_path, val_f1)
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                logger.info("Early stop @ epoch %d (patience=%d)", epoch + 1, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_loss": best_val_loss,
        "error_types": list(ERROR_TYPES),
        "per_class_f1": {
            name: None if best_per_class is None or np.isnan(best_per_class[i]) else float(best_per_class[i])
            for i, name in enumerate(ERROR_TYPES)
        },
        "n_train": len(train_idx),
        "n_val": len(dataset) - len(train_idx),
        "n_windows": len(dataset),
        "synthetic": synthetic,
        "encoder": encoder,
        "window_frames": window_frames,
        "embed_size": embed_size,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(
        "Xong. Best epoch=%d val_macro_f1=%.3f | %s",
        best_epoch,
        best_f1,
        best_path,
    )
    return model


def _f1_cols(split: str) -> List[str]:
    return [f"{split}_f1_{name}" for name in ERROR_TYPES]


def _csv_f1(value: float) -> str:
    if np.isnan(value):
        return ""
    return f"{float(value):.6f}"


def _checkpoint_payload(model, **meta):
    per = meta.pop("per_class_f1")
    return {
        "model_state_dict": model.state_dict(),
        "error_types": list(ERROR_TYPES),
        "per_class_f1": {
            name: None if np.isnan(per[i]) else float(per[i]) for i, name in enumerate(ERROR_TYPES)
        },
        **meta,
    }


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train Error DL multi-label (BCE từng head)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffs", default="poses/diffs")
    parser.add_argument("--labels", default="annotations/error_labels.csv")
    parser.add_argument("--windows", default="annotations/suggestions.csv")
    parser.add_argument("--scores", default="annotations/scores.csv")
    parser.add_argument("--poses", default="poses")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--encoder", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Không cần file .npy: sinh diff giả có tín hiệu theo nhãn để kiểm tra vòng train",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)
    epochs = args.epochs if args.epochs is not None else (8 if args.smoke else 30)
    ckpt = args.checkpoint_dir or ("experiments/error_dl_smoke" if args.smoke else "experiments/error_dl")
    train_model(
        num_epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        diffs_dir=args.diffs,
        labels_csv=args.labels,
        windows_csv=args.windows,
        scores_csv=args.scores,
        poses_dir=args.poses,
        checkpoint_dir=ckpt,
        window_frames=args.window_frames,
        embed_size=args.embed_size,
        encoder=args.encoder,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        use_pos_weight=not args.no_pos_weight,
        synthetic=args.smoke,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
