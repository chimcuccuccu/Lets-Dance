"""
Person 2 — Temporal DL Model.

Hai class chính:
1. TemporalBiLSTM  — backbone encoder cho sequence geometry features.
2. AISTPPPretrainModel — backbone + genre classification head (pretrain proxy task).

API deliverable cho fusion:
    temporal_model(seq) -> (embedding, dtw_features)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Input dimensions
GEOM_DIM   = 8    # geometry features (8 góc khớp)
PRETRAIN_INPUT_DIM  = GEOM_DIM          # AIST++ pretrain: chỉ geometry 1 chuỗi
TEMPORAL_INPUT_DIM  = GEOM_DIM * 2 + 1  # main training: perf + ref + dtw = 17
NODTW_INPUT_DIM     = GEOM_DIM * 2      # kịch bản (b) LSTM-only: bỏ kênh dtw = 16

NUM_GENRES = 10   # genres trong AIST++ (đồng bộ dataset.py)


# ──────────────────────────────────────────────────────────────────────────────
# TemporalBiLSTM — Backbone
# ──────────────────────────────────────────────────────────────────────────────

class TemporalBiLSTM(nn.Module):
    """
    Bidirectional LSTM encoder nhỏ cho chuỗi geometry features.

    Input : (B, T, input_dim)
    Output: embedding (B, embed_dim)   ←  dùng cho fusion

    input_dim:
      - 8  khi pretrain trên AIST++ (geometry 1 chuỗi)
      - 17 khi train chính trên dataset tự quay
    """

    def __init__(
        self,
        input_dim: int = TEMPORAL_INPUT_DIM,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_size = hidden_size
        self.embed_dim   = embed_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM → hidden_size * 2 (vì bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        returns: (B, embed_dim)
        """
        out, (h_n, _) = self.lstm(x)   # out: (B, T, hidden*2)
        # Lấy hidden state cuối cùng từ cả 2 hướng (forward + backward)
        # h_n: (num_layers * 2, B, hidden_size)
        h_fwd = h_n[-2]   # (B, hidden_size) — forward
        h_bwd = h_n[-1]   # (B, hidden_size) — backward
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden*2)
        h_cat = self.dropout(h_cat)
        embedding = self.fc(h_cat)   # (B, embed_dim)
        return embedding


# ──────────────────────────────────────────────────────────────────────────────
# AISTPPPretrainModel — Backbone + Genre Classification Head
# ──────────────────────────────────────────────────────────────────────────────

class AISTPPPretrainModel(nn.Module):
    """
    Pretrain model: TemporalBiLSTM + head phân loại thể loại nhảy (genre).

    Proxy task: phân loại AIST++ genre (gBR, gHO, gJB...) → model học representation
    chuyển động tốt trước khi fine-tune trên dataset tự quay.

    Input : (B, T, 8)   ← AIST++ geometry features
    Output: (B, NUM_GENRES)  ← logits
    """

    def __init__(
        self,
        num_genres: int = NUM_GENRES,
        hidden_size: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = TemporalBiLSTM(
            input_dim=PRETRAIN_INPUT_DIM,
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, num_genres)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 8)
        returns logits: (B, num_genres)
        """
        emb = self.backbone(x)      # (B, embed_dim)
        return self.head(emb)       # (B, num_genres)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Lấy embedding (không qua classification head) — dùng sau pretrain."""
        return self.backbone(x)


# ──────────────────────────────────────────────────────────────────────────────
# TemporalRegressionModel — Model chính
# ──────────────────────────────────────────────────────────────────────────────

class TemporalRegressionModel(nn.Module):
    """
    Model chính cho bài toán regression dự đoán khớp nhịp.

    Input : (B, T, input_dim)
            input_dim=17 → ghép (perf_geom, ref_geom, dtw_per_frame)  [kịch bản (c)]
            input_dim=16 → ghép (perf_geom, ref_geom), bỏ kênh dtw    [kịch bản (b)]
    Output: (B,)         ← dự đoán score (đã normalize)

    Nếu có pretrained checkpoint từ AISTPPPretrainModel, có thể load
    backbone.state_dict() để khởi tạo warm-start thay vì random init —
    `load_pretrained_backbone` lọc theo shape nên dùng được cho cả 16 và 17.
    """

    def __init__(
        self,
        input_dim: int = TEMPORAL_INPUT_DIM,
        hidden_size: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.backbone = TemporalBiLSTM(
            input_dim=input_dim,
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),   # output [0,1] — target đã normalize
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        returns: (B,) dự đoán score [0, 1]
        """
        emb = self.backbone(x)
        return self.regressor(emb).squeeze(1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Lấy embedding (B, embed_dim) để dùng cho fusion."""
        return self.backbone(x)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_pretrained_backbone(
    model: TemporalRegressionModel,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> TemporalRegressionModel:
    """
    Load backbone weights từ AISTPPPretrainModel checkpoint vào TemporalRegressionModel.

    Chỉ transfer các weights có shape khớp (bỏ qua LSTM input projection vì
    pretrain input_dim=8 khác main model input_dim=17). Các recurrent weights
    (h→h) và FC projection vẫn được transfer bình thường.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    # Lọc lấy chỉ backbone weights
    backbone_state = {
        k.removeprefix("backbone."): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }
    # Bỏ qua keys có shape không khớp (LSTM input weights: weight_ih_l0*)
    current_state = model.backbone.state_dict()
    compatible = {
        k: v for k, v in backbone_state.items()
        if k in current_state and current_state[k].shape == v.shape
    }
    skipped = [k for k in backbone_state if k not in compatible]
    if skipped:
        print(f"[load_pretrained] Bỏ qua {len(skipped)} keys do shape mismatch: {skipped}")
    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)
    n_loaded = len(compatible)
    print(f"[load_pretrained] Loaded {n_loaded} keys từ {checkpoint_path}")
    if missing:
        print(f"[load_pretrained] Missing (sẽ random init): {missing}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Public inference API (Tuần 5+)
# ──────────────────────────────────────────────────────────────────────────────

def _dtw_summary_from_row(dtw_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Thống kê DTW từ một dòng `annotations/dtw_features.csv`."""
    from src.temporal_dl.dataset import sorted_seg_columns

    seg_cols = sorted_seg_columns(dtw_row)
    segs = np.array(
        [float(dtw_row[c]) for c in seg_cols
         if c in dtw_row and dtw_row[c] is not None and np.isfinite(float(dtw_row[c]))],
        dtype=np.float64,
    )
    flag_cols = sorted_seg_columns(dtw_row, prefix="flag_seg_")
    flags = np.array(
        [float(dtw_row[c]) for c in flag_cols
         if c in dtw_row and dtw_row[c] is not None and np.isfinite(float(dtw_row[c]))],
        dtype=np.float64,
    )
    n_seg = int(segs.size)
    n_flagged = int(flags.sum()) if flags.size else 0
    total = float(dtw_row.get("dtw_distance_total", 0.0) or 0.0)
    return {
        "dtw_distance_total": total,
        "dtw_seg_mean": float(segs.mean()) if n_seg else 0.0,
        "dtw_seg_std": float(segs.std()) if n_seg else 0.0,
        "dtw_seg_max": float(segs.max()) if n_seg else 0.0,
        "dtw_seg_min": float(segs.min()) if n_seg else 0.0,
        "n_segments": n_seg,
        "n_flagged": n_flagged,
        "flag_ratio": n_flagged / max(n_seg, 1),
        "source": "dtw_row",
    }


def _dtw_summary_from_channel(seq: torch.Tensor, window_frames: int, hop: int) -> Dict[str, Any]:
    """
    Fallback: suy thống kê DTW từ kênh cuối của input (chỉ khi input_dim=17).

    Dùng trung bình theo từng cửa sổ làm "segment", vì kênh dtw per-frame vốn
    được tile ra từ các giá trị `dtw_seg_*`.
    """
    ch = seq[..., -1]
    if ch.dim() == 2:          # (B, T) → lấy chuỗi đầu
        ch = ch[0]
        logger.warning(
            "temporal_model: dtw_features suy từ input channel cho batch > 1 — "
            "chỉ dùng chuỗi đầu tiên. Truyền dtw_row để có số chính xác."
        )
    arr = ch.detach().cpu().numpy().astype(np.float64)
    T = arr.shape[0]
    segs = np.array(
        [arr[s : s + window_frames].mean() for s in range(0, max(T - window_frames + 1, 1), hop)],
        dtype=np.float64,
    )
    if segs.size == 0:
        segs = np.array([arr.mean()], dtype=np.float64)
    return {
        "dtw_distance_total": float(arr.mean()),
        "dtw_seg_mean": float(segs.mean()),
        "dtw_seg_std": float(segs.std()),
        "dtw_seg_max": float(segs.max()),
        "dtw_seg_min": float(segs.min()),
        "n_segments": int(segs.size),
        "n_flagged": 0,
        "flag_ratio": 0.0,
        "source": "input_channel",
    }


def temporal_model(
    seq: torch.Tensor | np.ndarray,
    model: Optional[TemporalRegressionModel] = None,
    checkpoint_path: Optional[str | Path] = None,
    *,
    dtw_row: Optional[Mapping[str, Any]] = None,
    window_frames: int = 60,
    hop_frames: Optional[int] = None,
    aggregate: str = "mean_std",
    target_range: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    API deliverable Tuần 5:  ``temporal_model(seq) -> (embedding, dtw_features)``

    Args:
        seq: (T, C) hoặc (B, T, C) — C=17 (có kênh dtw) hoặc 16 (không).
            Chuỗi dài hơn `window_frames` sẽ được cắt cửa sổ rồi gộp lại, nên
            truyền cả video được.
        model / checkpoint_path: truyền 1 trong 2. Với checkpoint_path,
            `input_dim` lấy từ config của checkpoint (bắt buộc cho model 16 chiều).
        dtw_row: một dòng của `annotations/dtw_features.csv`. Có thì
            `dtw_features` chính xác; không có thì suy từ kênh cuối của input.
        aggregate: "mean" | "mean_std" | "mean_std_minmax" | "none".
        target_range: (min, max) để de-normalize prediction về thang điểm gốc.
            Thiếu thì lấy từ config của checkpoint nếu có.

    Returns:
        embedding: (D,) với input (T,C), hoặc (B, D) với input (B,T,C).
            D = embed_dim × {1, 2, 4} tuỳ `aggregate`; aggregate="none" trả
            (n_window, embed_dim).
        dtw_features: dict — xem src/temporal_dl/README.md. Khoá `vector` là
            np.ndarray (4,) = [total, seg_mean, seg_std, seg_max], **đúng thứ tự**
            `_dtw_row_features` của Person 1 để fusion Tuần 6 concat trực tiếp.
    """
    from src.temporal_dl.infer import encode_video, load_temporal_checkpoint

    ckpt_meta: Dict[str, Any] = {}
    if model is None:
        if checkpoint_path is None:
            raise ValueError("temporal_model cần `model` hoặc `checkpoint_path`")
        model, ckpt_meta = load_temporal_checkpoint(checkpoint_path)
    model.eval()

    if target_range is None:
        tr = ckpt_meta.get("config", {}).get("target_range") if ckpt_meta else None
        if tr:
            target_range = (float(tr[0]), float(tr[1]))

    if isinstance(seq, np.ndarray):
        seq = torch.from_numpy(seq.astype(np.float32))
    elif not torch.is_floating_point(seq):
        seq = seq.float()

    hop = hop_frames or window_frames
    batched = seq.dim() == 3
    seqs = [seq[i] for i in range(seq.shape[0])] if batched else [seq]

    embs, scores, n_windows = [], [], []
    for s in seqs:
        emb, win_scores = encode_video(
            s.detach().cpu().numpy(), model,
            window_frames=window_frames, hop_frames=hop,
            aggregate=aggregate, return_scores=True,
        )
        embs.append(torch.from_numpy(emb))
        scores.append(float(np.mean(win_scores)))
        n_windows.append(int(len(win_scores)))

    embedding = torch.stack(embs, dim=0) if batched else embs[0]

    # ---- dtw_features ----
    if dtw_row is not None:
        summary = _dtw_summary_from_row(dtw_row)
    elif seq.shape[-1] == TEMPORAL_INPUT_DIM:
        summary = _dtw_summary_from_channel(seq, window_frames, hop)
    else:
        logger.warning(
            "temporal_model: không có dtw_row và input_dim=%d (không có kênh dtw) "
            "→ dtw_features rỗng.", seq.shape[-1],
        )
        summary = {
            "dtw_distance_total": 0.0, "dtw_seg_mean": 0.0, "dtw_seg_std": 0.0,
            "dtw_seg_max": 0.0, "dtw_seg_min": 0.0, "n_segments": 0,
            "n_flagged": 0, "flag_ratio": 0.0, "source": "unavailable",
        }

    predicted_score = scores[0] if not batched else float(np.mean(scores))
    summary["vector"] = np.array(
        [summary["dtw_distance_total"], summary["dtw_seg_mean"],
         summary["dtw_seg_std"], summary["dtw_seg_max"]],
        dtype=np.float32,
    )
    summary["predicted_score"] = predicted_score
    summary["predicted_target"] = (
        predicted_score * (target_range[1] - target_range[0]) + target_range[0]
        if target_range else None
    )
    summary["n_windows"] = n_windows[0] if not batched else n_windows
    summary["window_frames"] = window_frames
    summary["hop_frames"] = hop
    summary["aggregate"] = aggregate
    summary["input_dim"] = int(seq.shape[-1])

    return embedding, summary
