"""Person 3 — baseline rule: ngưỡng DTW, không phân loại được loại lỗi.

Ngưỡng là cột `threshold` trong `suggestions.csv` (phân vị 90 theo từng dance_id).
Một cửa sổ vượt ngưỡng được coi là có lỗi. Rule không có head theo loại, nên khi
chấm cùng bảng multi-label với DL, cờ đó được gán cho mọi error_type.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd
    return ROOT / p


def apply_dtw_threshold(distance: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """1 nếu DTW của cửa sổ lớn hơn ngưỡng của bài, ngược lại 0."""
    dist = np.asarray(distance, dtype=np.float64)
    limit = np.asarray(threshold, dtype=np.float64)
    flagged = np.isfinite(dist) & np.isfinite(limit) & (dist > limit)
    return flagged.astype(np.float32)


def broadcast_flag(flags: np.ndarray, n_classes: int) -> np.ndarray:
    """Cùng một cờ cho mọi loại lỗi. Shape (N,) → (N, n_classes)."""
    if n_classes < 1:
        raise ValueError("n_classes phải >= 1")
    column = np.asarray(flags, dtype=np.float32).reshape(-1, 1)
    return np.repeat(column, n_classes, axis=1)


def align_dtw_threshold(
    windows: pd.DataFrame,
    suggestions_csv: str | Path = "annotations/suggestions.csv",
) -> tuple[np.ndarray, np.ndarray]:
    """Khớp DTW + ngưỡng theo (dance_id, video_id, window_id), giữ đúng thứ tự `windows`."""
    path = resolve_path(suggestions_csv)
    suggestions = pd.read_csv(path, encoding="utf-8-sig")
    need = {"dance_id", "video_id", "window_id", "dtw_distance_windowed", "threshold"}
    missing = need - set(suggestions.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột {sorted(missing)}")
    suggestions = suggestions.copy()
    suggestions["dance_id"] = suggestions["dance_id"].astype(str)
    suggestions["video_id"] = suggestions["video_id"].astype(str)
    suggestions["window_id"] = pd.to_numeric(suggestions["window_id"], errors="coerce").astype("Int64")
    suggestions = suggestions.drop_duplicates(["dance_id", "video_id", "window_id"], keep="first")
    lookup = suggestions.set_index(["dance_id", "video_id", "window_id"])

    keys = windows.copy()
    keys["dance_id"] = keys["dance_id"].astype(str)
    keys["video_id"] = keys["video_id"].astype(str)
    keys["window_id"] = pd.to_numeric(keys["window_id"], errors="coerce").astype("Int64")
    index = pd.MultiIndex.from_frame(keys[["dance_id", "video_id", "window_id"]])
    aligned = lookup.reindex(index)
    distance = aligned["dtw_distance_windowed"].to_numpy(dtype=np.float64)
    threshold = aligned["threshold"].to_numpy(dtype=np.float64)
    return distance, threshold


def rule_predictions(
    windows: pd.DataFrame,
    n_classes: int,
    suggestions_csv: str | Path = "annotations/suggestions.csv",
) -> np.ndarray:
    """(N, n_classes) — 1 ở mọi loại nếu cửa sổ vượt ngưỡng DTW."""
    distance, threshold = align_dtw_threshold(windows, suggestions_csv)
    return broadcast_flag(apply_dtw_threshold(distance, threshold), n_classes)
