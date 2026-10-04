"""
Person 1 — SpatialModelV3: 1D-CNN (+ optional BiLSTM) trên diff_sequence.

Dual embedding:
  - raw embedding → score head (giữ biên độ)
  - L2-normalized embedding → fusion / FAISS
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_EMBED_SIZE = 256
NUM_JOINT_GROUPS = 4  # face, arms, torso, legs


class SpatialEncoder(nn.Module):
    """1D-CNN (+ optional BiLSTM) → embedding (raw; normalize tuỳ chọn)."""

    def __init__(
        self,
        num_joints: int = 33,
        num_dims: int = 3,
        embed_size: int = DEFAULT_EMBED_SIZE,
        normalize_embedding: bool = False,
        use_lstm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_dims = num_dims
        self.embed_size = embed_size
        self.input_size = num_joints * num_dims
        self.normalize_embedding = normalize_embedding
        self.use_lstm = use_lstm

        self.conv1 = nn.Conv1d(self.input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(192)
        self.conv4 = nn.Conv1d(192, embed_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(embed_size)
        self.drop = nn.Dropout(dropout)

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=embed_size,
                hidden_size=embed_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.lstm = None

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.embed_proj = nn.Linear(embed_size * 2, embed_size)

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        normalize: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected (B,T,J,D), got {tuple(x.shape)}")
        b, t, j, d = x.shape
        if j * d != self.input_size:
            raise ValueError(f"Expected J*D={self.input_size}, got J={j} D={d}")

        h = x.reshape(b, t, j * d).permute(0, 2, 1)  # (B, 99, T)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.drop(h)

        if self.lstm is not None:
            seq = h.permute(0, 2, 1)  # (B, T, E)
            seq, _ = self.lstm(seq)
            features = seq.permute(0, 2, 1)  # (B, E, T)
        else:
            features = h

        avg = self.avg_pool(features).squeeze(-1)
        mx = self.max_pool(features).squeeze(-1)
        embedding = self.embed_proj(torch.cat([avg, mx], dim=1))
        do_norm = self.normalize_embedding if normalize is None else normalize
        if do_norm:
            embedding = F.normalize(embedding, p=2, dim=1)
        return features, embedding

    def forward(self, x: torch.Tensor, *, normalize: Optional[bool] = None) -> torch.Tensor:
        _, embedding = self.forward_features(x, normalize=normalize)
        return embedding


class SpatialModelV3(nn.Module):
    """diff_sequence → score + (normalized) embedding; optional joint-group aux."""

    def __init__(
        self,
        num_joints: int = 33,
        num_dims: int = 3,
        embed_size: int = DEFAULT_EMBED_SIZE,
        dropout: float = 0.3,
        normalize_embedding: bool = False,
        use_lstm: bool = True,
        use_aux_head: bool = True,
    ):
        super().__init__()
        self.use_aux_head = use_aux_head
        # Encoder giữ raw embedding; normalize chỉ khi encode(..., normalize=True)
        self.encoder = SpatialEncoder(
            num_joints=num_joints,
            num_dims=num_dims,
            embed_size=embed_size,
            normalize_embedding=normalize_embedding,
            use_lstm=use_lstm,
            dropout=min(dropout, 0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
        # Aux: dự đoán mean |diff| theo nhóm khớp (face/arms/torso/legs)
        self.aux_fc = (
            nn.Sequential(
                nn.Linear(embed_size, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, NUM_JOINT_GROUPS),
                nn.Softplus(),
            )
            if use_aux_head
            else None
        )

    @property
    def normalize_embedding(self) -> bool:
        return self.encoder.normalize_embedding

    def encode(self, x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """Fusion API: mặc định L2-normalize. Score head dùng normalize=False."""
        return self.encoder(x, normalize=normalize)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        emb_raw = self.encode(x, normalize=False)
        score = self.fc(emb_raw)
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        if return_aux and self.aux_fc is not None:
            aux = self.aux_fc(emb_raw)
            return score, emb_norm, aux
        return score, emb_norm

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        if state_dict and not any(k.startswith("encoder.") for k in state_dict):
            if any(k.startswith(("conv1.", "bn1.", "embed_proj.")) for k in state_dict):
                migrated = {}
                for k, v in state_dict.items():
                    if k.startswith("fc.") or k.startswith("aux_fc."):
                        migrated[k] = v
                    elif k.startswith("decoder."):
                        continue
                    else:
                        migrated[f"encoder.{k}"] = v
                state_dict = migrated
        return super().load_state_dict(state_dict, strict=strict)


class SpatialAutoEncoder(nn.Module):
    """Reconstruct (B,T,33,3) via shared SpatialEncoder + conv decoder."""

    def __init__(
        self,
        num_joints: int = 33,
        num_dims: int = 3,
        embed_size: int = DEFAULT_EMBED_SIZE,
        use_lstm: bool = True,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_dims = num_dims
        self.encoder = SpatialEncoder(
            num_joints=num_joints,
            num_dims=num_dims,
            embed_size=embed_size,
            normalize_embedding=False,
            use_lstm=use_lstm,
        )
        out_ch = num_joints * num_dims
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_size, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features, embedding = self.encoder.forward_features(x, normalize=False)
        recon_flat = self.decoder(features)
        b, _, t = recon_flat.shape
        recon = recon_flat.permute(0, 2, 1).reshape(
            b, t, self.num_joints, self.num_dims
        )
        return recon, embedding


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def extract_encoder_state(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "encoder_state_dict" in state:
        raw = state["encoder_state_dict"]
    elif "model_state_dict" in state:
        raw = state["model_state_dict"]
    else:
        raw = state

    if any(k.startswith("encoder.") for k in raw):
        return _strip_prefix(raw, "encoder.")

    legacy_keys = ("conv1.", "bn1.", "conv2.", "bn2.", "conv3.", "bn3.", "embed_proj.")
    if any(k.startswith(legacy_keys) for k in raw):
        return {
            k: v
            for k, v in raw.items()
            if not k.startswith("fc.")
            and not k.startswith("decoder.")
            and not k.startswith("aux_fc.")
        }
    return dict(raw)


def load_pretrained_encoder(
    model: Union[SpatialModelV3, SpatialEncoder],
    checkpoint_path: str,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> Tuple[int, int]:
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    enc_state = extract_encoder_state(ckpt if isinstance(ckpt, dict) else {})
    target = model.encoder if isinstance(model, SpatialModelV3) else model
    missing, unexpected = target.load_state_dict(enc_state, strict=strict)
    n_loaded = sum(1 for k in enc_state if k not in unexpected)
    logger.info(
        "Loaded encoder from %s | tensors≈%d missing=%d unexpected=%d",
        checkpoint_path,
        n_loaded,
        len(missing),
        len(unexpected),
    )
    if missing:
        logger.warning("Missing keys (arch mismatch?): %s", list(missing)[:8])
    return n_loaded, len(missing)


def spatial_model(
    diff_sequence: torch.Tensor,
    model: SpatialModelV3,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """
    API fusion: ``spatial_model(diff_sequence) -> embedding`` (L2-norm mặc định).

    Prefer ``src.spatial_dl.infer.encode_diff`` khi input là NumPy / cần TTA.
    """
    model.eval()
    with torch.no_grad():
        if not torch.is_tensor(diff_sequence):
            diff_sequence = torch.as_tensor(diff_sequence, dtype=torch.float32)
        if diff_sequence.dim() == 3:
            diff_sequence = diff_sequence.unsqueeze(0)
        device = next(model.parameters()).device
        return model.encode(diff_sequence.to(device), normalize=normalize)


if __name__ == "__main__":
    model = SpatialModelV3()
    dummy = torch.randn(4, 150, 33, 3)
    score, embed, aux = model(dummy, return_aux=True)
    ae = SpatialAutoEncoder()
    recon, _ = ae(dummy)
    print(
        f"Score: {score.shape} | Emb: {embed.shape} | Aux: {aux.shape} | Recon: {recon.shape}"
    )
