"""
Person 2 — API fusion-ready cho Temporal DL (Tuần 5).

Trách nhiệm duy nhất: load checkpoint và biến một video thành embedding bằng
**đúng** preprocessing của lúc train. Không có CLI.

Quy tắc chống lệch train/infer: mọi hàm ở đây cắt cửa sổ bằng `_chunk_sequence`
của `dataset.py` và dựng input bằng `build_temporal_sequence` của `dataset.py` —
không hàm nào tự cắt/tự ghép lại.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.temporal_dl.dataset import (
    DEFAULT_DTW_CSV,
    DEFAULT_POSES_DIR,
    DEFAULT_SCORES_CSV,
    _chunk_sequence,
    build_temporal_sequence,
    parse_person_id,
    sorted_seg_columns,
)
from src.temporal_dl.lstm import TemporalRegressionModel

logger = logging.getLogger(__name__)

AGGREGATE_MODES = ("mean", "mean_std", "mean_std_minmax", "none")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def load_temporal_checkpoint(
    path: str | Path,
    *,
    device: Optional[str | torch.device] = None,
) -> Tuple[TemporalRegressionModel, Dict[str, Any]]:
    """
    Load `TemporalRegressionModel` + metadata từ file .pth.

    `input_dim` lấy từ `ckpt["config"]`, không mặc định 17 — bắt buộc, vì encoder
    của kịch bản (b) có input_dim=16 và sẽ không load được nếu dựng sai.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint không tồn tại: {path}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint không hợp lệ (thiếu model_state_dict): {path}")

    cfg = ckpt.get("config", {})
    model = TemporalRegressionModel(
        input_dim=int(cfg.get("input_dim", 17)),
        hidden_size=int(cfg.get("hidden_size", 64)),
        embed_dim=int(cfg.get("embed_dim", 32)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    logger.info(
        "Loaded temporal ckpt %s | input_dim=%d target=%s split=%s epoch=%s",
        path.name, model.input_dim, cfg.get("target_col", "?"),
        cfg.get("split", "?"), ckpt.get("epoch", "?"),
    )
    return model, ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Encode
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_windows(embs: np.ndarray, mode: str = "mean_std") -> np.ndarray:
    """
    Gộp embedding của các cửa sổ thành một vector cho cả video.

    embs: (n_window, embed_dim) → (embed_dim × k,)

    `mean_std` là mặc định: `std` giữa các cửa sổ mã hoá *mức đồng đều* của
    timing trong cả bài — người đúng nhịp nửa đầu rồi trôi khác hẳn người lệch
    đều nhẹ, dù hai người có thể cùng `dtw_distance_total`. Chỉ lấy mean là ném
    thông tin đó đi.
    """
    if mode not in AGGREGATE_MODES:
        raise ValueError(f"aggregate phải thuộc {AGGREGATE_MODES}, nhận {mode!r}")
    if mode == "none":
        return embs.astype(np.float32)
    parts = [embs.mean(axis=0)]
    if mode in ("mean_std", "mean_std_minmax"):
        parts.append(embs.std(axis=0))
    if mode == "mean_std_minmax":
        parts.extend([embs.min(axis=0), embs.max(axis=0)])
    return np.concatenate(parts).astype(np.float32)


@torch.no_grad()
def encode_windows(
    windows: np.ndarray | Sequence[np.ndarray],
    model: TemporalRegressionModel,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode các cửa sổ → (embeddings (n, embed_dim), scores (n,)).

    `scores` là output của regressor, còn ở thang [0,1] đã normalize.
    """
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"windows phải là (n, T, C), nhận {arr.shape}")
    if arr.shape[-1] != model.input_dim:
        raise ValueError(
            f"input_dim lệch: model mong {model.input_dim}, windows có {arr.shape[-1]}. "
            f"Nhớ dựng sequence với use_dtw={model.input_dim == 17}."
        )

    device = next(model.parameters()).device
    model.eval()
    embs: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for start in range(0, arr.shape[0], batch_size):
        x = torch.from_numpy(arr[start : start + batch_size]).to(device)
        embs.append(model.get_embedding(x).cpu().numpy())
        scores.append(model(x).cpu().numpy())
    return (
        np.concatenate(embs, axis=0).astype(np.float32),
        np.concatenate(scores, axis=0).astype(np.float32),
    )


def encode_video(
    seq: np.ndarray,
    model: TemporalRegressionModel,
    *,
    window_frames: int = 60,
    hop_frames: Optional[int] = None,
    aggregate: str = "mean_std",
    batch_size: int = 64,
    return_scores: bool = False,
):
    """
    Encode cả một video: (T, C) → embedding đã gộp.

    Cắt cửa sổ bằng `_chunk_sequence` của dataset.py để biên cửa sổ khớp y hệt
    lúc train. `hop_frames` nhỏ hơn `window_frames` (vd. 30) chỉ làm tăng số mẫu
    đứng sau mean/std → bớt nhiễu gộp; encoder vốn chỉ thấy từng cửa sổ nên
    không bị ảnh hưởng, và các cửa sổ chồng nhau đều thuộc cùng một video nên
    không tạo leakage.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"encode_video cần (T, C), nhận {seq.shape}")
    hop = hop_frames or window_frames
    chunks = _chunk_sequence(seq, window_frames, hop)
    embs, scores = encode_windows(np.stack(chunks, axis=0), model, batch_size=batch_size)
    out = aggregate_windows(embs, aggregate)
    if return_scores:
        return out, scores
    return out


def predict_video_score(
    seq: np.ndarray,
    model: TemporalRegressionModel,
    *,
    target_range: Optional[Tuple[float, float]] = None,
    window_frames: int = 60,
    hop_frames: Optional[int] = None,
) -> float:
    """Dự đoán điểm cho cả video (trung bình các cửa sổ), de-normalize nếu có range."""
    _, scores = encode_video(
        seq, model, window_frames=window_frames, hop_frames=hop_frames,
        aggregate="mean", return_scores=True,
    )
    s = float(np.mean(scores))
    if target_range:
        tmin, tmax = target_range
        return s * (tmax - tmin) + tmin
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Export embeddings cho toàn dataset
# ──────────────────────────────────────────────────────────────────────────────

def export_temporal_embeddings(
    model: TemporalRegressionModel,
    ckpt: Optional[Dict[str, Any]] = None,
    *,
    scores_csv: str | Path = DEFAULT_SCORES_CSV,
    dtw_csv: str | Path = DEFAULT_DTW_CSV,
    poses_dir: str | Path = DEFAULT_POSES_DIR,
    aggregate: str = "mean_std",
    window_frames: int = 60,
    hop_frames: Optional[int] = 30,
    out_csv: Optional[str | Path] = None,
    split_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> pd.DataFrame:
    """
    Xuất embedding per-video cho toàn bộ dataset → DataFrame (và CSV nếu có out_csv).

    Cột: dance_id, video_id, person_id, các sub-score, split, n_windows,
    window_frames, hop_frames, aggregate, emb_dim, pred_score, pred_target,
    use_dtw, ckpt_epoch, ckpt_target, rồi emb_000..emb_{D-1}.
    """
    cfg = (ckpt or {}).get("config", {})
    use_dtw = bool(cfg.get("use_dtw", model.input_dim == 17))
    target_range = cfg.get("target_range")
    target_range = (float(target_range[0]), float(target_range[1])) if target_range else None

    scores = pd.read_csv(scores_csv)
    scores.columns = [str(c).strip().strip('"').lstrip("﻿") for c in scores.columns]
    dtw = pd.read_csv(dtw_csv)
    df = scores.merge(dtw, on=["dance_id", "video_id"], how="inner")
    seg_cols = sorted_seg_columns(dtw)
    logger.info("export_temporal_embeddings: %d video sau inner-join", len(df))

    rows: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []
    for _, r in df.iterrows():
        dance_id, video_id = str(r["dance_id"]), str(r["video_id"])
        seq = build_temporal_sequence(
            dance_id, video_id, r, poses_dir=poses_dir,
            use_dtw=use_dtw, seg_cols=seg_cols,
        )
        if seq is None:
            logger.warning("Bỏ %s/%s — thiếu pose", dance_id, video_id)
            continue
        emb, win_scores = encode_video(
            seq, model, window_frames=window_frames, hop_frames=hop_frames,
            aggregate=aggregate, return_scores=True,
        )
        pred = float(np.mean(win_scores))
        rows.append(
            {
                "dance_id": dance_id,
                "video_id": video_id,
                "person_id": str(r.get("person_id", "") or parse_person_id(video_id)),
                "khop_dong_tac": float(r.get("khop_dong_tac", np.nan)),
                "khop_nhip": float(r.get("khop_nhip", np.nan)),
                "nang_luong": float(r.get("nang_luong", np.nan)),
                "tong_diem": float(r.get("tong_diem", np.nan)),
                "split": (split_map or {}).get((dance_id, video_id), ""),
                "n_windows": int(len(win_scores)),
                "window_frames": window_frames,
                "hop_frames": hop_frames or window_frames,
                "aggregate": aggregate,
                "emb_dim": int(emb.shape[0]),
                "pred_score": pred,
                "pred_target": (
                    pred * (target_range[1] - target_range[0]) + target_range[0]
                    if target_range else np.nan
                ),
                "emb_l2norm": float(np.linalg.norm(emb)),
                "use_dtw": use_dtw,
                "ckpt_epoch": (ckpt or {}).get("epoch", ""),
                "ckpt_target": cfg.get("target_col", ""),
            }
        )
        vectors.append(emb)

    if not rows:
        raise RuntimeError("export_temporal_embeddings: không encode được video nào")

    meta = pd.DataFrame(rows)
    emb_all = np.stack(vectors, axis=0)
    emb_df = pd.DataFrame(
        emb_all, columns=[f"emb_{i:03d}" for i in range(emb_all.shape[1])]
    )
    out = pd.concat([meta.reset_index(drop=True), emb_df], axis=1)

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        logger.info("Đã ghi: %s  (%d video × %d chiều)", out_csv, *emb_all.shape)
    return out
