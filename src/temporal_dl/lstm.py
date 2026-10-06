"""
Person 2 — Temporal DL Model (Tuần 3-4).

Hai class chính:
1. TemporalBiLSTM  — backbone encoder cho sequence geometry features.
2. AISTPPPretrainModel — backbone + genre classification head (pretrain proxy task).

API cho fusion (Tuần 5+):
    temporal_model(seq) -> (embedding, dtw_summary)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Input dimensions
GEOM_DIM   = 8    # geometry features (8 góc khớp)
PRETRAIN_INPUT_DIM  = GEOM_DIM          # AIST++ pretrain: chỉ geometry 1 chuỗi
TEMPORAL_INPUT_DIM  = GEOM_DIM * 2 + 1  # main training: perf + ref + dtw = 17

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
# TemporalRegressionModel — Model chính (Tuần 4)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalRegressionModel(nn.Module):
    """
    Model chính cho bài toán regression dự đoán khớp nhịp.

    Input : (B, T, 17)  ← ghép (perf_geom, ref_geom, dtw_per_frame)
    Output: (B,)         ← dự đoán score (đã normalize)

    Nếu có pretrained checkpoint từ AISTPPPretrainModel, có thể load
    backbone.state_dict() để khởi tạo warm-start thay vì random init.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = TemporalBiLSTM(
            input_dim=TEMPORAL_INPUT_DIM,
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
        x: (B, T, 17)
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

def temporal_model(
    seq: torch.Tensor,
    model: Optional[TemporalRegressionModel] = None,
    checkpoint_path: Optional[str | Path] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Inference API: temporal_model(seq) -> (embedding, score)

    seq: (T, 17) hoặc (B, T, 17) tensor
    Returns:
        embedding: (embed_dim,) hoặc (B, embed_dim)
        score: float — dự đoán score (nếu model là TemporalRegressionModel)

    Dùng cho Tuần 5 Fusion.
    """
    if model is None:
        model = TemporalRegressionModel()
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)

    model.eval()
    with torch.no_grad():
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)  # (1, T, 17)
        emb = model.get_embedding(seq)   # (B, embed_dim)
        score = model(seq).mean().item()

    return emb.squeeze(0), score
