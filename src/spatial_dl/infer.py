"""
Person 1 — API fusion-ready: spatial_model(diff_sequence) → embedding.

Hỗ trợ TTA (mirror) và dual embedding (norm cho fusion / raw cho score).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.preprocessing.preprocess import normalize_length
from src.spatial_dl.dataset import DEFAULT_TARGET_FRAMES, mirror_diff_torch
from src.spatial_dl.model_v3 import DEFAULT_EMBED_SIZE, SpatialModelV3

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]


def load_spatial_checkpoint(
    checkpoint_path: str,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[SpatialModelV3, Dict]:
    """Load SpatialModelV3 + metadata từ file .pth."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint không hợp lệ (thiếu model_state_dict): {checkpoint_path}")

    embed_size = int(ckpt.get("embed_size", DEFAULT_EMBED_SIZE))
    use_lstm = bool(ckpt.get("use_lstm", True))
    # aux_fc có thể thiếu ở ckpt cũ → strict=False
    model = SpatialModelV3(embed_size=embed_size, use_lstm=use_lstm, use_aux_head=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, ckpt


def _to_batch_tensor(
    diff_sequence: ArrayLike,
    *,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    device: torch.device,
) -> torch.Tensor:
    """Chuẩn hoá input về (B, T, 33, 3) float32 trên device."""
    if isinstance(diff_sequence, np.ndarray):
        x = diff_sequence.astype(np.float32)
        if x.ndim == 3:
            if target_frames > 0 and x.shape[0] != target_frames:
                x = normalize_length(x, target_frames)
            x = torch.from_numpy(x).unsqueeze(0)
        elif x.ndim == 4:
            fixed = []
            for i in range(x.shape[0]):
                seq = x[i]
                if target_frames > 0 and seq.shape[0] != target_frames:
                    seq = normalize_length(seq, target_frames)
                fixed.append(seq)
            x = torch.from_numpy(np.stack(fixed, axis=0))
        else:
            raise ValueError(f"Expected (T,33,3) or (B,T,33,3), got {x.shape}")
    else:
        x = diff_sequence
        if not torch.is_floating_point(x):
            x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f"Expected (T,33,3) or (B,T,33,3), got {tuple(x.shape)}")
        if target_frames > 0 and x.shape[1] != target_frames:
            arr = x.detach().cpu().numpy()
            fixed = [normalize_length(arr[i], target_frames) for i in range(arr.shape[0])]
            x = torch.from_numpy(np.stack(fixed, axis=0))

    if x.shape[-2:] != (33, 3):
        raise ValueError(f"Expected last dims (33, 3), got {tuple(x.shape)}")
    return x.to(device)


@torch.no_grad()
def encode_diff(
    diff_sequence: ArrayLike,
    model: SpatialModelV3,
    *,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    return_numpy: bool = True,
    normalize: bool = True,
    use_tta: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    API chính tuần 5 / fusion:
      embedding = encode_diff(diff_sequence, model)

    Args:
        normalize: True → L2 (fusion); False → raw (phù hợp score head)
        use_tta: trung bình embedding gốc + mirror (tuỳ chọn; mặc định tắt)
    """
    device = next(model.parameters()).device
    model.eval()
    x = _to_batch_tensor(diff_sequence, target_frames=target_frames, device=device)
    emb = model.encode(x, normalize=False)
    if use_tta:
        emb_m = model.encode(mirror_diff_torch(x), normalize=False)
        emb = 0.5 * (emb + emb_m)
    if normalize:
        emb = F.normalize(emb, p=2, dim=1)
    if emb.shape[0] == 1:
        emb = emb.squeeze(0)
    if return_numpy:
        return emb.detach().cpu().numpy()
    return emb


@torch.no_grad()
def predict_score(
    diff_sequence: ArrayLike,
    model: SpatialModelV3,
    *,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    score_scale: float = 100.0,
    use_tta: bool = False,
    return_numpy: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Dự đoán điểm (đã nhân score_scale). TTA mirror tắt mặc định (bật nếu cần)."""
    device = next(model.parameters()).device
    model.eval()
    x = _to_batch_tensor(diff_sequence, target_frames=target_frames, device=device)
    score, _ = model(x)
    if use_tta:
        score_m, _ = model(mirror_diff_torch(x))
        score = 0.5 * (score + score_m)
    score = score * score_scale
    if score.shape[0] == 1:
        score = score.squeeze(0)
    if return_numpy:
        return score.detach().cpu().numpy()
    return score


def _diff_path(data_dir: str, dance_id: str, video_id: str) -> Optional[str]:
    for p in (
        os.path.join(data_dir, dance_id, f"{video_id}_diff.npy"),
        os.path.join(data_dir, f"{dance_id}_{video_id}_diff.npy"),
        os.path.join(data_dir, dance_id, f"{video_id}.npy"),
    ):
        if os.path.isfile(p):
            return p
    return None


@torch.no_grad()
def export_spatial_embeddings(
    scores_csv: str,
    diffs_dir: str,
    checkpoint_path: str,
    *,
    out_npy: str = "experiments/spatial_dl_improved/spatial_embeddings.npy",
    out_meta: str = "experiments/spatial_dl_improved/spatial_embeddings_meta.csv",
    target_frames: int = DEFAULT_TARGET_FRAMES,
    batch_size: int = 32,
    use_tta: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Xuất embedding (N, E) + CSV metadata."""
    model, ckpt = load_spatial_checkpoint(checkpoint_path)
    df = pd.read_csv(scores_csv)

    rows: List[dict] = []
    diffs: List[np.ndarray] = []
    for _, row in df.iterrows():
        dance_id = str(row["dance_id"])
        video_id = str(row["video_id"])
        path = _diff_path(diffs_dir, dance_id, video_id)
        if not path:
            continue
        diff = np.load(path).astype(np.float32)
        if target_frames > 0 and diff.shape[0] != target_frames:
            diff = normalize_length(diff, target_frames)
        diffs.append(diff)
        rows.append(
            {
                "dance_id": dance_id,
                "video_id": video_id,
                "person_id": str(row.get("person_id", video_id)),
                "khop_dong_tac": float(row["khop_dong_tac"]) if "khop_dong_tac" in row else np.nan,
                "khop_nhip": float(row["khop_nhip"]) if "khop_nhip" in row else np.nan,
                "nang_luong": float(row["nang_luong"]) if "nang_luong" in row else np.nan,
                "tong_diem": float(row["tong_diem"]) if "tong_diem" in row else np.nan,
                "diff_path": path,
            }
        )

    if not diffs:
        raise FileNotFoundError(f"Không tìm thấy diff nào trong {diffs_dir}")

    emb_all = encode_diffs_batch(
        diffs, model, target_frames=target_frames, batch_size=batch_size, use_tta=use_tta
    )
    meta = pd.DataFrame(rows)
    meta["embed_dim"] = emb_all.shape[1]
    meta["ckpt_epoch"] = ckpt.get("epoch")
    meta["ckpt_target"] = ckpt.get("target_col", "")
    meta["use_tta"] = use_tta

    os.makedirs(os.path.dirname(out_npy) or ".", exist_ok=True)
    np.save(out_npy, emb_all)
    meta.to_csv(out_meta, index=False)
    logger.info(
        "Exported embeddings %s → %s | meta=%s | tta=%s",
        emb_all.shape,
        out_npy,
        out_meta,
        use_tta,
    )
    return emb_all, meta


def encode_diffs_batch(
    diffs: Sequence[np.ndarray],
    model: SpatialModelV3,
    *,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    batch_size: int = 32,
    use_tta: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """Encode list of (T,33,3) → (N, E)."""
    if not diffs:
        return np.zeros((0, model.encoder.embed_size), dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(diffs), batch_size):
            chunk = []
            for d in diffs[start : start + batch_size]:
                arr = np.asarray(d, dtype=np.float32)
                if target_frames > 0 and arr.shape[0] != target_frames:
                    arr = normalize_length(arr, target_frames)
                chunk.append(arr)
            x = torch.from_numpy(np.stack(chunk, axis=0)).to(device)
            emb = model.encode(x, normalize=False)
            if use_tta:
                emb = 0.5 * (emb + model.encode(mirror_diff_torch(x), normalize=False))
            if normalize:
                emb = F.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
