"""Person 3 — Precision/Recall/F1 từng loại lỗi, so với rule ngưỡng DTW.

Accuracy không phải số chính: đa số cửa sổ không có lỗi, đoán toàn 0 vẫn đúng phần lớn.
Rule chỉ có một bit (DTW > ngưỡng bài). Bit đó được chấm cho từng loại để thấy
rule không tách được off_beat / wrong_move / ...
DL có một head sigmoid cho mỗi loại.
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.error_dl.dataset import ERROR_TYPES, get_error_dataloaders, resolve_project_path
from src.error_dl.error_model import (
    ErrorModel,
    aggregate_error_embedding,
    error_embedding_names,
    error_model,
)
from src.error_dl.rule_engine import rule_predictions
from src.error_dl.train import train_model

logger = logging.getLogger(__name__)

DECISION_THRESHOLD = 0.5


def diffs_ready(diffs_dir: str) -> bool:
    root = resolve_project_path(diffs_dir)
    if not root.is_dir():
        return False
    return next(root.rglob("*_diff.npy"), None) is not None


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true = y_true.astype(np.float64)
    pred = y_pred.astype(np.float64)
    tp = (pred * true).sum(axis=0)
    fp = (pred * (1.0 - true)).sum(axis=0)
    fn = ((1.0 - pred) * true).sum(axis=0)
    tn = ((1.0 - pred) * (1.0 - true)).sum(axis=0)
    return tp, fp, fn, tn


def _prf(tp: float, fp: float, fn: float, tn: float) -> tuple[float, float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    return precision, recall, f1, accuracy


def per_error_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    error_types: Sequence[str] = ERROR_TYPES,
) -> List[dict]:
    """Một dòng mỗi loại, cộng macro / micro / bài toán 'có lỗi hay không'."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} khác y_pred {y_pred.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != len(error_types):
        raise ValueError(f"Expected (N, {len(error_types)}), nhận {y_true.shape}")
    binary_pred = (y_pred >= DECISION_THRESHOLD).astype(np.float64)
    binary_true = (y_true >= DECISION_THRESHOLD).astype(np.float64)
    tp, fp, fn, tn = _counts(binary_true, binary_pred)

    rows: List[dict] = []
    precisions, recalls, f1s, accuracies = [], [], [], []
    for i, name in enumerate(error_types):
        precision, recall, f1, accuracy = _prf(float(tp[i]), float(fp[i]), float(fn[i]), float(tn[i]))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
        rows.append(
            _row(model_name, name, precision, recall, f1, float(binary_true[:, i].sum()), accuracy, tp[i], fp[i], fn[i])
        )

    def _nanmean(values: List[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    rows.append(
        _row(
            model_name,
            "__macro__",
            _nanmean(precisions),
            _nanmean(recalls),
            _nanmean(f1s),
            float(binary_true.sum()),
            _nanmean(accuracies),
            float(tp.sum()),
            float(fp.sum()),
            float(fn.sum()),
        )
    )
    precision, recall, f1, accuracy = _prf(float(tp.sum()), float(fp.sum()), float(fn.sum()), float(tn.sum()))
    rows.append(
        _row(model_name, "__micro__", precision, recall, f1, float(binary_true.sum()), accuracy, tp.sum(), fp.sum(), fn.sum())
    )

    any_true = (binary_true.max(axis=1) > 0).astype(np.float64)
    any_pred = (binary_pred.max(axis=1) > 0).astype(np.float64)
    atp = float((any_pred * any_true).sum())
    afp = float((any_pred * (1.0 - any_true)).sum())
    afn = float(((1.0 - any_pred) * any_true).sum())
    atn = float(((1.0 - any_pred) * (1.0 - any_true)).sum())
    precision, recall, f1, accuracy = _prf(atp, afp, afn, atn)
    rows.append(_row(model_name, "__any_error__", precision, recall, f1, float(any_true.sum()), accuracy, atp, afp, afn))
    return rows


def _row(model_name, error_type, precision, recall, f1, support, accuracy, tp, fp, fn) -> dict:
    return {
        "model": model_name,
        "error_type": error_type,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def comparison_table(
    y_true: np.ndarray,
    dl_probs: np.ndarray,
    rule_pred: np.ndarray,
    data_tag: str,
) -> pd.DataFrame:
    rows = per_error_rows(y_true, dl_probs, "dl") + per_error_rows(y_true, rule_pred, "rule_dtw")
    table = pd.DataFrame(rows)
    table.insert(0, "data", data_tag)
    return table


def predict_probabilities(
    model: ErrorModel,
    dataset,
    indices: Sequence[int],
    batch_size: int = 64,
) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("Không có cửa sổ để chấm")
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch_x, _batch_y in loader:
            probs, _emb = model.probabilities(batch_x.to(device))
            chunks.append(probs.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def video_error_embeddings(dataset, probabilities: np.ndarray) -> pd.DataFrame:
    """Một dòng mỗi video. probabilities đã xếp đúng thứ tự dataset."""
    if len(probabilities) != len(dataset):
        raise ValueError("probabilities phải phủ hết dataset, đúng thứ tự cửa sổ")
    meta = dataset.meta.reset_index(drop=True)
    names = error_embedding_names()
    rows = []
    for (dance_id, video_id), group in meta.groupby(["dance_id", "video_id"], sort=False):
        emb = aggregate_error_embedding(probabilities[group.index.to_numpy()])
        if torch.is_tensor(emb):
            emb = emb.cpu().numpy()
        row = {"dance_id": dance_id, "video_id": video_id, "n_windows": int(len(group))}
        row.update({name: float(value) for name, value in zip(names, emb)})
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "  n/a"
    return f"{value:6.3f}"


def log_table(table: pd.DataFrame) -> None:
    show = table.loc[table["error_type"] != "__micro__"].copy()
    logger.info("Ngưỡng quyết định DL = %.2f. Accuracy cao vẫn có thể đi với F1 thấp.", DECISION_THRESHOLD)
    for rec in show.itertuples(index=False):
        logger.info(
            "%-8s %-16s  P=%s  R=%s  F1=%s  support=%4.0f  acc=%s",
            rec.model,
            rec.error_type,
            _fmt(rec.precision),
            _fmt(rec.recall),
            _fmt(rec.f1),
            rec.support,
            _fmt(rec.accuracy),
        )


def _macro_f1(table: pd.DataFrame, model_name: str) -> float:
    hit = table.loc[(table["model"] == model_name) & (table["error_type"] == "__macro__"), "f1"]
    return float(hit.iloc[0])


def evaluate(
    synthetic: bool,
    epochs: int,
    batch_size: int,
    seed: int,
    checkpoint_dir: str,
    diffs_dir: str = "poses/diffs",
    labels_csv: str = "annotations/error_labels.csv",
    windows_csv: str = "annotations/suggestions.csv",
    scores_csv: str = "annotations/scores.csv",
    poses_dir: str = "poses",
    encoder: str = "cnn",
    window_frames: int = 32,
    embed_size: int = 64,
    patience: int = 8,
) -> pd.DataFrame:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    data_tag = "synthetic" if synthetic else "diff_sequence"
    model = train_model(
        num_epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        diffs_dir=diffs_dir,
        labels_csv=labels_csv,
        windows_csv=windows_csv,
        scores_csv=scores_csv,
        poses_dir=poses_dir,
        checkpoint_dir=checkpoint_dir,
        window_frames=window_frames,
        embed_size=embed_size,
        encoder=encoder,
        synthetic=synthetic,
        patience=patience,
    )
    _train_loader, _val_loader, dataset, _train_idx, val_idx = get_error_dataloaders(
        diffs_dir=diffs_dir,
        labels_csv=labels_csv,
        windows_csv=windows_csv,
        scores_csv=scores_csv,
        poses_dir=poses_dir,
        batch_size=batch_size,
        seed=seed,
        window_frames=window_frames,
        synthetic=synthetic,
    )
    order = list(range(len(dataset)))
    probs_all = predict_probabilities(model, dataset, order, batch_size=batch_size)
    y_val = dataset.targets[val_idx]
    dl_val = probs_all[val_idx]
    rule_val = rule_predictions(dataset.meta.iloc[val_idx], n_classes=len(ERROR_TYPES), suggestions_csv=windows_csv)
    table = comparison_table(y_val, dl_val, rule_val, data_tag=data_tag)
    out_dir = resolve_project_path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "per_error_metrics.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    embeddings = video_error_embeddings(dataset, probs_all)
    emb_path = out_dir / "error_embeddings.csv"
    embeddings.to_csv(emb_path, index=False, encoding="utf-8-sig")

    sample = dataset.meta.iloc[val_idx[0]]
    video_idx = dataset.meta.index[
        (dataset.meta["dance_id"] == sample["dance_id"]) & (dataset.meta["video_id"] == sample["video_id"])
    ].tolist()
    windows = torch.stack([dataset[i][0] for i in video_idx])
    probs, video_emb = error_model(windows, model)
    logger.info(
        "API error_model(%s/%s): probabilities=%s error_embedding=%s",
        sample["dance_id"],
        sample["video_id"],
        tuple(probs.shape),
        tuple(video_emb.shape),
    )
    log_table(table)
    dl_f1 = _macro_f1(table, "dl")
    rule_f1 = _macro_f1(table, "rule_dtw")
    logger.info(
        "Macro F1 từng loại — DL %.3f | rule DTW %.3f | bảng %s | embedding %s",
        dl_f1,
        rule_f1,
        table_path,
        emb_path,
    )
    zero = np.zeros_like(y_val)
    zero_acc = float((zero == (y_val >= DECISION_THRESHOLD)).mean())
    logger.info(
        "Accuracy nếu đoán mọi cửa sổ không có lỗi: %.3f. Vì vậy bảng dùng Precision/Recall/F1 từng loại.",
        zero_acc,
    )
    return table


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Chấm Error DL theo từng loại lỗi và so với ngưỡng DTW")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Chưa có diff .npy: sinh diff giả có pattern riêng từng loại lỗi",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)
    ready = diffs_ready(args.diffs)
    if not ready and not args.smoke:
        raise SystemExit(
            "Không có diff_sequence .npy trong poses/diffs. "
            "Train trên diff thật: python -m src.preprocessing.build_diffs && python -m src.error_dl.eval. "
            "Kiểm tra bảng khi chưa có pose: python -m src.error_dl.eval --smoke"
        )
    synthetic = args.smoke or not ready
    epochs = args.epochs if args.epochs is not None else (6 if synthetic else 30)
    ckpt = args.checkpoint_dir or ("experiments/error_dl_smoke" if synthetic else "experiments/error_dl")
    evaluate(
        synthetic=synthetic,
        epochs=epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_dir=ckpt,
        diffs_dir=args.diffs,
        labels_csv=args.labels,
        windows_csv=args.windows,
        scores_csv=args.scores,
        poses_dir=args.poses,
        encoder=args.encoder,
        window_frames=args.window_frames,
        embed_size=args.embed_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
