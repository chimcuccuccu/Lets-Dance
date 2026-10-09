"""Person 3 — multi-label Error DL: shared encoder + một head sigmoid / loại lỗi.

`forward` trả logit để train bằng BCE-with-logits (đúng BCE sau sigmoid, ổn định hơn).
API `error_model(...)` trả xác suất từng cửa sổ và error_embedding đã nén cho cả video.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.error_dl.dataset import DEFAULT_WINDOW_FRAMES, ERROR_TYPES, NUM_DIMS, NUM_JOINTS

ArrayLike = Union[np.ndarray, torch.Tensor]


class ErrorEncoder(nn.Module):
    """Encoder nhỏ dùng chung: CNN 1D hoặc MLP trên cửa sổ (B, T, 33, 3)."""

    def __init__(
        self,
        embed_size: int = 64,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        encoder: str = "cnn",
        dropout: float = 0.1,
        num_joints: int = NUM_JOINTS,
        num_dims: int = NUM_DIMS,
    ):
        super().__init__()
        if encoder not in ("cnn", "mlp"):
            raise ValueError("encoder phải là 'cnn' hoặc 'mlp'")
        self.encoder = encoder
        self.embed_size = embed_size
        self.window_frames = window_frames
        self.num_joints = num_joints
        self.num_dims = num_dims
        in_ch = num_joints * num_dims

        if encoder == "cnn":
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Linear(64, embed_size)
        else:
            self.net = nn.Sequential(
                nn.Linear(window_frames * in_ch, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, embed_size),
            )
            self.proj = nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (B, T, J, D), nhận {tuple(x.shape)}")
        b, t, j, d = x.shape
        if j != self.num_joints or d != self.num_dims:
            raise ValueError(f"Expected J={self.num_joints} D={self.num_dims}, nhận J={j} D={d}")
        if self.encoder == "cnn":
            h = x.reshape(b, t, j * d).permute(0, 2, 1).contiguous()
            h = self.net(h).squeeze(-1)
            return self.drop(self.proj(h))
        if t != self.window_frames:
            raise ValueError(
                f"MLP encoder cần T={self.window_frames}, nhận T={t}. "
                "Đổi --window-frames cho khớp checkpoint, hoặc dùng --encoder cnn."
            )
        h = self.net(x.reshape(b, -1))
        return self.drop(h)


class ErrorModel(nn.Module):
    """Shared encoder + một Linear head cho mỗi error_type.

    forward → (logits [B, C], embedding [B, E])
    Xác suất = sigmoid(logits), từng head độc lập.
    """

    def __init__(
        self,
        embed_size: int = 64,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        encoder: str = "cnn",
        dropout: float = 0.1,
        num_joints: int = NUM_JOINTS,
        num_dims: int = NUM_DIMS,
    ):
        super().__init__()
        self.error_types = ERROR_TYPES
        self.embed_size = embed_size
        self.window_frames = window_frames
        self.encoder_name = encoder
        self.encoder = ErrorEncoder(
            embed_size=embed_size,
            window_frames=window_frames,
            encoder=encoder,
            dropout=dropout,
            num_joints=num_joints,
            num_dims=num_dims,
        )
        self.heads = nn.ModuleList([nn.Linear(embed_size, 1) for _ in ERROR_TYPES])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        logits = torch.cat([head(embedding) for head in self.heads], dim=1)
        return logits, embedding

    def probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, embedding = self.forward(x)
        return torch.sigmoid(logits), embedding


def binary_cross_entropy_per_head(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE từng head (sigmoid bên trong BCE-with-logits), rồi lấy trung bình các head."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits {tuple(logits.shape)} khác target {tuple(targets.shape)}")
    if logits.dim() != 2:
        raise ValueError(f"Expected (B, C), nhận {tuple(logits.shape)}")
    losses = []
    for i in range(logits.shape[1]):
        weight = None if pos_weight is None else pos_weight[i]
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[:, i],
                targets[:, i],
                pos_weight=weight,
            )
        )
    return torch.stack(losses).mean()


def error_embedding_names(error_types: Tuple[str, ...] = ERROR_TYPES) -> list[str]:
    """Tên cột fusion: mean theo loại, rồi max theo loại."""
    names = list(error_types)
    return [f"mean_{name}" for name in names] + [f"max_{name}" for name in names]


def aggregate_error_embedding(
    error_probabilities: ArrayLike,
    severity: Optional[ArrayLike] = None,
) -> ArrayLike:
    """Nén xác suất các cửa sổ của một video → một vector cho fusion.

    Input (N, C) hoặc (C,). Output (2C,):
      [0:C]  trung bình có trọng số severity (không có severity thì trọng số đều)
      [C:2C] max-pool theo loại. Có severity thì mỗi loại lấy xác suất của cửa sổ
             mà (xác suất × severity) lớn nhất, giá trị lưu vẫn là xác suất.

    Mọi cửa sổ đưa vào phải thuộc cùng một video.
    """
    is_tensor = torch.is_tensor(error_probabilities)
    device = error_probabilities.device if is_tensor else None
    dtype = error_probabilities.dtype if is_tensor else None
    if is_tensor:
        probs = error_probabilities.detach().float().cpu().numpy()
    else:
        probs = np.asarray(error_probabilities, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] == 0:
        raise ValueError(f"error_probabilities phải là (N, C), nhận {probs.shape}")

    if severity is None:
        weights = np.ones(probs.shape[0], dtype=np.float64)
    else:
        if is_tensor and torch.is_tensor(severity):
            weights = severity.detach().float().cpu().numpy().reshape(-1)
        else:
            weights = np.asarray(severity, dtype=np.float64).reshape(-1)
        if weights.shape[0] != probs.shape[0]:
            raise ValueError(
                f"severity có {weights.shape[0]} phần tử, probabilities có {probs.shape[0]} cửa sổ"
            )
        weights = np.clip(weights, 0.0, None)
        if float(weights.sum()) <= 0:
            weights = np.ones(probs.shape[0], dtype=np.float64)

    weighted = (probs * weights[:, None]).sum(axis=0) / weights.sum()
    if severity is None:
        pooled_max = probs.max(axis=0)
    else:
        score = probs * weights[:, None]
        chosen = score.argmax(axis=0)
        pooled_max = probs[chosen, np.arange(probs.shape[1])]
    out = np.concatenate([weighted, pooled_max]).astype(np.float32)
    if is_tensor:
        return torch.tensor(out, device=device, dtype=dtype if dtype is not None else torch.float32)
    return out


def error_model(
    diff_sequence_windowed: ArrayLike,
    model: ErrorModel,
    severity: Optional[ArrayLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """API: các cửa sổ của một video → (error_probabilities [N, 5], error_embedding [10]).

    `diff_sequence_windowed` là (T, 33, 3) một cửa sổ hoặc (N, T, 33, 3) mọi cửa sổ
    của cùng một video. Embedding là bản nén để đưa vào fusion, không phải hidden
    của encoder. Hidden từng cửa sổ nằm ở `model.probabilities`.
    """
    model.eval()
    if isinstance(diff_sequence_windowed, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(diff_sequence_windowed, dtype=np.float32))
    elif torch.is_tensor(diff_sequence_windowed):
        x = diff_sequence_windowed.float()
    else:
        x = torch.as_tensor(diff_sequence_windowed, dtype=torch.float32)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected (T,33,3) hoặc (N,T,33,3), nhận {tuple(x.shape)}")
    device = next(model.parameters()).device
    with torch.no_grad():
        probs, _window_embedding = model.probabilities(x.to(device))
        video_embedding = aggregate_error_embedding(probs, severity=severity)
        if torch.is_tensor(video_embedding):
            video_embedding = video_embedding.to(device=probs.device, dtype=probs.dtype)
        return probs, video_embedding


def load_error_model(path: str, device: Optional[torch.device] = None) -> ErrorModel:
    map_location = device or torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    model = ErrorModel(
        embed_size=int(ckpt.get("embed_size", 64)),
        window_frames=int(ckpt.get("window_frames", DEFAULT_WINDOW_FRAMES)),
        encoder=str(ckpt.get("encoder", "cnn")),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model


if __name__ == "__main__":
    net = ErrorModel()
    dummy = torch.randn(4, DEFAULT_WINDOW_FRAMES, NUM_JOINTS, NUM_DIMS)
    logits, window_emb = net(dummy)
    probs, video_emb = error_model(dummy, net)
    print(
        f"logits={tuple(logits.shape)} window_emb={tuple(window_emb.shape)} "
        f"probs={tuple(probs.shape)} error_embedding={tuple(video_emb.shape)}"
    )
