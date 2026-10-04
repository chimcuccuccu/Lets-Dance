"""
Person 1 — Dataset loader cho Spatial DL.
Đọc diff_sequence + nhãn tong_diem; split theo person_id.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.preprocessing.preprocess import build_diff_sequence, normalize_length

logger = logging.getLogger(__name__)

DEFAULT_TARGET_FRAMES = 150
SCORE_MAX = 300.0
# Per-criterion scales (Spatial branch nên ưu tiên khop_dong_tac)
TARGET_SCALES = {
    "tong_diem": 300.0,
    "khop_dong_tac": 100.0,
    "khop_nhip": 100.0,
    "nang_luong": 100.0,
}

# MediaPipe left↔right pairs (mirror augmentation)
_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def mirror_pose_diff(diff: np.ndarray) -> np.ndarray:
    """Mirror trái-phải trên diff (T, 33, 3): đảo x và swap cặp khớp."""
    out = diff.copy()
    out[:, :, 0] *= -1.0
    for a, b in _MIRROR_PAIRS:
        tmp = out[:, a].copy()
        out[:, a] = out[:, b]
        out[:, b] = tmp
    return out


def mirror_diff_torch(diff: torch.Tensor) -> torch.Tensor:
    """Mirror batch/tensor diff: (T,33,3) hoặc (B,T,33,3)."""
    if diff.dim() == 3:
        out = diff.unsqueeze(0).clone()
        squeeze = True
    elif diff.dim() == 4:
        out = diff.clone()
        squeeze = False
    else:
        raise ValueError(f"Expected (T,33,3) or (B,T,33,3), got {tuple(diff.shape)}")
    out[..., 0] *= -1.0
    for a, b in _MIRROR_PAIRS:
        tmp = out[:, :, a, :].clone()
        out[:, :, a, :] = out[:, :, b, :]
        out[:, :, b, :] = tmp
    return out.squeeze(0) if squeeze else out


# Nhóm khớp cho aux loss / TTA diagnostics (MediaPipe-33)
JOINT_GROUPS = {
    "face": list(range(0, 11)),
    "arms": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "torso": [11, 12, 23, 24],
    "legs": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
}
# face nhẹ, limbs nặng — khớp với chuyên môn Spatial (hình dạng)
JOINT_GROUP_LOSS_WEIGHTS = {
    "face": 0.25,
    "arms": 1.5,
    "torso": 1.0,
    "legs": 1.5,
}


def joint_group_abs_means(diff: torch.Tensor) -> torch.Tensor:
    """(B,T,33,3) → (B, 4) mean |diff| theo face/arms/torso/legs."""
    if diff.dim() != 4:
        raise ValueError(f"Expected (B,T,33,3), got {tuple(diff.shape)}")
    abs_d = diff.abs()
    outs = []
    for name in ("face", "arms", "torso", "legs"):
        idxs = JOINT_GROUPS[name]
        outs.append(abs_d[:, :, idxs, :].mean(dim=(1, 2, 3)))
    return torch.stack(outs, dim=1)


class SpatialDanceDataset(Dataset):
    """
    Modes:
      - default / official: load precomputed *_diff.npy
      - use_demo_data: build diff on-the-fly từ pose P+R
      - use_dummy: random tensors
      - use_old_data: legacy (không dùng cho báo cáo)
    """

    def __init__(
        self,
        data_dir: str = "",
        csv_file: str = "",
        is_train: bool = True,
        use_dummy: bool = False,
        use_old_data: bool = False,
        use_demo_data: bool = False,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        require_files: bool = True,
        alignment_dir: str = "",
        normalize_score: bool = True,
        augment: bool = False,
        target_col: str = "tong_diem",
        score_max: float = 0.0,
    ):
        self.use_dummy = use_dummy
        self.use_old_data = use_old_data
        self.use_demo_data = use_demo_data
        self.is_train = is_train
        self.target_frames = target_frames
        self.require_files = require_files
        self.alignment_dir = alignment_dir
        self.data_dir = data_dir
        self.normalize_score = normalize_score
        self.augment = augment and is_train
        self.target_col = target_col
        self.score_max = float(
            score_max
            or TARGET_SCALES.get(target_col, SCORE_MAX)
        )
        self.person_ids: List[str] = []

        if use_dummy:
            self.num_samples = 100
            rng = np.random.default_rng(42 if is_train else 43)
            raw = rng.uniform(0.2 * self.score_max, self.score_max, size=(self.num_samples,)).astype(np.float32)
            self.labels = (raw / self.score_max) if normalize_score else raw
            self.person_ids = [f"dummy_{i % 10}" for i in range(self.num_samples)]
            self.df = None
            return

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self.num_samples = len(self.df)

        if "person_id" in self.df.columns:
            self.person_ids = self.df["person_id"].astype(str).tolist()
        elif "video_id" in self.df.columns:
            self.person_ids = self.df["video_id"].astype(str).tolist()
        else:
            self.person_ids = [str(i) for i in range(self.num_samples)]

        if use_old_data:
            logger.warning("use_old_data=True — chỉ smoke test.")
            self.all_poses = np.load(data_dir, allow_pickle=True)
            if self.all_poses.ndim == 3 and self.all_poses.shape[-1] == 99:
                self.all_poses = self.all_poses.reshape(
                    self.all_poses.shape[0], self.all_poses.shape[1], 33, 3
                )
            score_col = "target_score" if "target_score" in self.df.columns else "tong_diem"
            provided = self.df[score_col].to_numpy(dtype=np.float32) if score_col in self.df.columns else np.zeros(len(self.df), dtype=np.float32)
            n = min(len(self.all_poses), len(provided))
            self.all_poses = self.all_poses[:n]
            labels = provided[:n]
            self.labels = (labels / self.score_max) if normalize_score else labels
            self.num_samples = n
            self.person_ids = self.person_ids[:n]
            return

        if use_demo_data:
            logger.info("Demo mode: build diff on-the-fly từ %s", data_dir)
            return

        if require_files:
            keep = []
            for i in range(len(self.df)):
                path = self._diff_path_for_row(self.df.iloc[i])
                if path and os.path.exists(path):
                    keep.append(i)
            if not keep:
                raise FileNotFoundError(
                    f"Không tìm thấy file diff trong {data_dir}. "
                    "Chạy: python -m src.preprocessing.build_diffs"
                )
            dropped = len(self.df) - len(keep)
            if dropped:
                logger.warning("Bỏ %d mẫu thiếu file diff.", dropped)
            self.df = self.df.iloc[keep].reset_index(drop=True)
            self.person_ids = [self.person_ids[i] for i in keep]
            self.num_samples = len(self.df)

    def _score_from_row(self, row) -> np.ndarray:
        col = self.target_col
        if col not in row.index:
            col = "tong_diem" if "tong_diem" in row.index else "score_total"
        val = float(row[col])
        if self.normalize_score:
            val = val / self.score_max
        return np.array([val], dtype=np.float32)

    def _diff_path_for_row(self, row) -> Optional[str]:
        dance_id = str(row["dance_id"])
        video_id = str(row["video_id"])
        candidates = [
            os.path.join(self.data_dir, f"{dance_id}_{video_id}_diff.npy"),
            os.path.join(self.data_dir, dance_id, f"{video_id}_diff.npy"),
            os.path.join(self.data_dir, dance_id, f"{video_id}.npy"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    def _load_alignment_path(self, dance_id: str, video_id: str) -> Optional[np.ndarray]:
        if not self.alignment_dir:
            return None
        candidates = [
            os.path.join(self.alignment_dir, f"{dance_id}_{video_id}_path.npy"),
            os.path.join(self.alignment_dir, dance_id, f"{video_id}_path.npy"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return np.load(p)
        return None

    def _ensure_length(self, seq: np.ndarray) -> np.ndarray:
        if seq.shape[0] == self.target_frames:
            return seq.astype(np.float32)
        return normalize_length(seq.astype(np.float32), self.target_frames)

    def _augment(self, diff: np.ndarray) -> np.ndarray:
        if not self.augment:
            return diff
        out = diff.astype(np.float32).copy()
        t = out.shape[0]

        if np.random.rand() < 0.5:
            out = mirror_pose_diff(out)

        # Temporal shift (circular) — mô phỏng lệch cửa sổ cắt
        if t > 8 and np.random.rand() < 0.5:
            shift = int(np.random.randint(-max(1, t // 10), max(2, t // 10)))
            out = np.roll(out, shift, axis=0)

        # Global scale jitter
        if np.random.rand() < 0.5:
            out = out * float(np.random.uniform(0.9, 1.1))

        # Additive noise
        if np.random.rand() < 0.6:
            sigma = float(np.random.uniform(0.005, 0.02))
            out = out + np.random.normal(0.0, sigma, size=out.shape).astype(np.float32)

        # Random joint dropout (zero a few joints for several frames)
        if np.random.rand() < 0.4:
            n_j = int(np.random.randint(1, 5))
            joints = np.random.choice(out.shape[1], size=n_j, replace=False)
            if t > 4:
                a = int(np.random.randint(0, t - 2))
                b = int(np.random.randint(a + 1, t))
                out[a:b, joints] = 0.0

        return out.astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.use_dummy:
            rng = np.random.default_rng(idx + (0 if self.is_train else 10_000))
            diff = rng.standard_normal((self.target_frames, 33, 3)).astype(np.float32)
            score = np.array([self.labels[idx]], dtype=np.float32)
            return torch.from_numpy(self._augment(diff)), torch.from_numpy(score)

        if self.use_old_data:
            diff = self._ensure_length(self.all_poses[idx].astype(np.float32))
            score = np.array([self.labels[idx]], dtype=np.float32)
            return torch.from_numpy(self._augment(diff)), torch.from_numpy(score)

        row = self.df.iloc[idx]

        if self.use_demo_data:
            video_id = str(row["video_id"])
            ref_id = str(row["ref_id"])
            score = self._score_from_row(row)
            p_path = os.path.join(self.data_dir, f"{video_id}.npy")
            r_path = os.path.join(self.data_dir, f"{ref_id}.npy")
            if not (os.path.exists(p_path) and os.path.exists(r_path)):
                raise FileNotFoundError(f"Thiếu pose demo: {p_path} hoặc {r_path}")
            dance_id = str(row["dance_id"]) if "dance_id" in row.index else "demo"
            alignment = self._load_alignment_path(dance_id, video_id)
            diff = build_diff_sequence(
                np.load(p_path),
                np.load(r_path),
                alignment_path=alignment,
                target_frames=self.target_frames,
                auto_dtw=alignment is None,
            )
            return torch.from_numpy(self._augment(diff)), torch.from_numpy(score)

        score = self._score_from_row(row)
        path = self._diff_path_for_row(row)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Thiếu diff: {path}")
        diff = self._ensure_length(np.load(path).astype(np.float32))
        return torch.from_numpy(self._augment(diff)), torch.from_numpy(score)


def split_indices_by_person(
    person_ids: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(person_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio))) if len(unique) > 1 else 0
    val_people = set(unique[:n_val])
    train_idx = [i for i, p in enumerate(person_ids) if p not in val_people]
    val_idx = [i for i, p in enumerate(person_ids) if p in val_people]
    if not train_idx or not val_idx:
        indices = np.arange(len(person_ids))
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * (1 - val_ratio)))
        train_idx = indices[:cut].tolist()
        val_idx = indices[cut:].tolist() or indices[-1:].tolist()
        if train_idx == val_idx and len(indices) > 1:
            train_idx = indices[:-1].tolist()
            val_idx = indices[-1:].tolist()
    return train_idx, val_idx


def get_dataloaders(
    batch_size: int = 16,
    use_dummy: bool = True,
    use_old_data: bool = False,
    use_demo_data: bool = False,
    data_path: str = "",
    csv_path: str = "",
    train_csv: str = "",
    val_csv: str = "",
    val_ratio: float = 0.2,
    seed: int = 42,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    alignment_dir: str = "",
    num_workers: int = 0,
    normalize_score: bool = True,
    augment: bool = True,
    target_col: str = "tong_diem",
    score_max: float = 0.0,
):
    """
    Nếu truyền train_csv + val_csv (vd. Demo official 183/34) → dùng fixed split.
    Ngược lại: một csv_path rồi split theo person_id.
    """
    if train_csv and val_csv and not use_dummy:
        common = dict(
            data_dir=data_path,
            use_dummy=False,
            use_old_data=use_old_data,
            use_demo_data=use_demo_data,
            target_frames=target_frames,
            alignment_dir=alignment_dir,
            normalize_score=normalize_score,
            target_col=target_col,
            score_max=score_max,
        )
        train_ds = SpatialDanceDataset(
            csv_file=train_csv, is_train=True, augment=augment, **common
        )
        val_ds = SpatialDanceDataset(
            csv_file=val_csv, is_train=False, augment=False, **common
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        logger.info(
            "Fixed split: train=%d val=%d (people train=%d val=%d) score_norm=%s",
            len(train_ds),
            len(val_ds),
            len(set(train_ds.person_ids)),
            len(set(val_ds.person_ids)),
            normalize_score,
        )
        return train_loader, val_loader

    base_kwargs = dict(
        data_dir=data_path,
        csv_file=csv_path,
        use_dummy=use_dummy,
        use_old_data=use_old_data,
        use_demo_data=use_demo_data,
        target_frames=target_frames,
        alignment_dir=alignment_dir,
        normalize_score=normalize_score,
        target_col=target_col,
        score_max=score_max,
    )

    # Một dataset nguồn để split index, rồi wrap train (aug) / val (no aug)
    source = SpatialDanceDataset(is_train=True, augment=False, **base_kwargs)
    train_idx, val_idx = split_indices_by_person(source.person_ids, val_ratio=val_ratio, seed=seed)

    train_ds = SpatialDanceDataset(is_train=True, augment=augment and not use_dummy, **base_kwargs)
    val_ds = SpatialDanceDataset(is_train=False, augment=False, **base_kwargs)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(
        "Split theo person_id: train=%d val=%d (unique people=%d) score_norm=%s",
        len(train_idx),
        len(val_idx),
        len(set(source.person_ids)),
        normalize_score,
    )
    return train_loader, val_loader
