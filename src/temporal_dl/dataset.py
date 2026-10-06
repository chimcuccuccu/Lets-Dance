"""
Person 2 — Temporal DL Datasets (Tuần 3-4).

Hai dataset:
1. AISTPPDataset  — load từ shared_basic_cache (.npy T×33×4), tính geometry
   features, chunk thành cửa sổ → cho pretrain genre classification.
2. TemporalSeqDataset — ghép (performer_geom, ref_geom, dtw_per_frame) theo
   thời gian → cho training BiLSTM chính, target = dtw_distance_total.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.features.geometry import compute_geometry_features

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Genre labels (đồng bộ với Person 1 src/spatial_dl/aist_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────
GENRE_TAGS: Tuple[str, ...] = (
    "gBR", "gPO", "gLO", "gMH", "gLH", "gHO", "gWA", "gKR", "gJS", "gJB",
)
GENRE_TO_ID: Dict[str, int] = {g: i for i, g in enumerate(GENRE_TAGS)}
NUM_GENRES = len(GENRE_TAGS)

_GENRE_RE = re.compile(r"^(?P<genre>g[A-Z]{2})_")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "aistpp" / "shared_basic_cache"
DEFAULT_POSES_DIR = PROJECT_ROOT / "poses"
DEFAULT_DTW_CSV   = PROJECT_ROOT / "annotations" / "dtw_features.csv"

# Geometry feature dimension (8 góc khớp — xem geometry.py)
GEOM_DIM = 8


def _parse_genre(stem: str) -> int:
    """Trả về genre_id từ tên file. Fallback = 0 nếu không tìm thấy."""
    m = _GENRE_RE.match(stem)
    if m:
        return GENRE_TO_ID.get(m.group("genre"), 0)
    return 0


def _chunk_sequence(seq: np.ndarray, window: int, hop: int) -> List[np.ndarray]:
    """Cắt chuỗi T thành các cửa sổ không chồng (hoặc chồng hop)."""
    T = seq.shape[0]
    chunks = []
    start = 0
    while start + window <= T:
        chunks.append(seq[start : start + window])
        start += hop
    if not chunks and T > 0:
        # Sequence ngắn hơn window — pad bằng 0
        pad = np.zeros((window - T, seq.shape[1]), dtype=np.float32)
        chunks.append(np.concatenate([seq, pad], axis=0))
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# 1. AISTPPDataset — pretrain genre classification
# ──────────────────────────────────────────────────────────────────────────────
class AISTPPDataset(Dataset):
    """
    Load pre-converted AIST++ .npy từ shared_basic_cache.

    Mỗi sample là một cửa sổ geometry features (window_frames, 8) kèm nhãn genre.
    """

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        window_frames: int = 60,
        hop_frames: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
        max_files: int = 0,
    ):
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Cache dir không tồn tại: {cache_dir}")

        self.window = window_frames
        self.hop = hop_frames or window_frames  # mặc định non-overlapping
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Thu thập tất cả .npy files
        all_files = sorted(cache_dir.glob("*.npy"))
        if max_files > 0:
            all_files = all_files[:max_files]

        if not all_files:
            raise FileNotFoundError(f"Không tìm thấy .npy trong {cache_dir}")

        # Tạo danh sách (chunk, genre_id)
        self._samples: List[Tuple[np.ndarray, int]] = []
        genre_counts: Dict[int, int] = {}

        for fpath in all_files:
            genre_id = _parse_genre(fpath.stem)
            try:
                pose = np.load(fpath).astype(np.float32)  # (T, 33, 4)
                if pose.ndim != 3 or pose.shape[1] != 33:
                    logger.warning("Bỏ qua %s — shape=%s", fpath.name, pose.shape)
                    continue
                geom = compute_geometry_features(pose)   # (T, 8)
                chunks = _chunk_sequence(geom, self.window, self.hop)
                for chunk in chunks:
                    self._samples.append((chunk.astype(np.float32), genre_id))
                genre_counts[genre_id] = genre_counts.get(genre_id, 0) + len(chunks)
            except Exception as exc:
                logger.warning("Lỗi load %s: %s", fpath.name, exc)

        if not self._samples:
            raise RuntimeError("AISTPPDataset: không có sample nào hợp lệ.")

        logger.info(
            "AISTPPDataset: %d samples từ %d files | genres=%s",
            len(self._samples), len(all_files),
            {GENRE_TAGS[k]: v for k, v in genre_counts.items()},
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        geom, genre_id = self._samples[idx]
        if self.augment:
            # Jitter nhẹ (±3°)
            if self.rng.random() < 0.5:
                geom = geom + self.rng.normal(0.0, 3.0, size=geom.shape).astype(np.float32)
            # Scale nhẹ
            if self.rng.random() < 0.5:
                geom = geom * float(self.rng.uniform(0.95, 1.05))
        x = torch.from_numpy(geom)                         # (window, 8)
        y = torch.tensor(genre_id, dtype=torch.long)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# 2. TemporalSeqDataset — training BiLSTM chính
# ──────────────────────────────────────────────────────────────────────────────

def _expand_dtw_to_frames(dtw_segs: List[float], T: int) -> np.ndarray:
    """
    Tạo mảng DTW per-frame bằng cách chia đều giá trị từng segment ra T frames.
    dtw_segs: [d0, d1, ...] đã tính theo window.
    Trả về (T,) float32.
    """
    n = len(dtw_segs)
    if n == 0:
        return np.zeros(T, dtype=np.float32)
    # Lặp/tile giá trị phân đoạn ra T frames
    seg_len = T // n
    arr = np.repeat(np.array(dtw_segs, dtype=np.float32), max(seg_len, 1))
    if len(arr) < T:
        arr = np.pad(arr, (0, T - len(arr)), mode="edge")
    else:
        arr = arr[:T]
    return arr


class TemporalSeqDataset(Dataset):
    """
    Dataset cho BiLSTM chính.

    Mỗi sample là một video:
    Input : (T, 8+8+1) = ghép(perf_geom, ref_geom, dtw_per_frame)
    Target: dtw_distance_total (proxy cho khop_nhip khi chưa có scores.csv)

    Khi scores_csv có cột 'video_id' & 'khop_nhip', sẽ dùng cột đó làm target.
    """

    INPUT_DIM = GEOM_DIM * 2 + 1   # 17

    def __init__(
        self,
        dtw_csv: str | Path = DEFAULT_DTW_CSV,
        poses_dir: str | Path = DEFAULT_POSES_DIR,
        scores_csv: Optional[str | Path] = None,
        window_frames: int = 60,
        hop_frames: Optional[int] = None,
        augment: bool = False,
        seed: int = 42,
        normalize_target: bool = True,
    ):
        self.poses_dir = Path(poses_dir)
        self.window = window_frames
        self.hop = hop_frames or window_frames
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Load dtw_features.csv
        dtw_csv = Path(dtw_csv)
        if not dtw_csv.is_file():
            raise FileNotFoundError(f"DTW CSV không tồn tại: {dtw_csv}")
        dtw_df = pd.read_csv(dtw_csv)

        # Load scores.csv nếu có
        score_map: Dict[str, float] = {}
        if scores_csv is not None:
            sc = pd.read_csv(scores_csv)
            if "video_id" in sc.columns and "khop_nhip" in sc.columns:
                score_map = dict(zip(sc["video_id"].astype(str), sc["khop_nhip"].astype(float)))
                logger.info("Loaded %d khop_nhip scores từ %s", len(score_map), scores_csv)

        # Nhận diện cột dtw_seg_*
        seg_cols = sorted(
            [c for c in dtw_df.columns if c.startswith("dtw_seg_")],
            key=lambda c: int(c.split("_")[-1]),
        )

        self._samples: List[Tuple[np.ndarray, float]] = []

        for _, row in dtw_df.iterrows():
            dance_id = str(row.get("dance_id", ""))
            video_id = str(row.get("video_id", ""))

            # Bỏ qua file reference
            if "ref" in video_id.lower():
                continue

            # Tìm performer và reference pose files
            dance_dir = self.poses_dir / dance_id
            if not dance_dir.is_dir():
                continue

            perf_path = dance_dir / f"{video_id}.npy"
            ref_candidates = list(dance_dir.glob(f"{dance_id}_ref.npy"))
            ref_candidates += list(dance_dir.glob("*ref*.npy"))

            if not perf_path.is_file() or not ref_candidates:
                continue
            ref_path = ref_candidates[0]

            try:
                perf_pose = np.load(perf_path).astype(np.float32)  # (T,33,4)
                ref_pose  = np.load(ref_path).astype(np.float32)   # (Tr,33,4)

                perf_geom = compute_geometry_features(perf_pose)   # (T,8)
                ref_geom  = compute_geometry_features(ref_pose)    # (Tr,8)

                # Align độ dài về min(T, Tr)
                T = min(perf_geom.shape[0], ref_geom.shape[0])
                perf_geom = perf_geom[:T]
                ref_geom  = ref_geom[:T]

                # DTW per-frame
                seg_vals = [float(row[c]) for c in seg_cols if pd.notna(row.get(c))]
                dtw_frame = _expand_dtw_to_frames(seg_vals, T)  # (T,)

                # Ghép input: (T, 17)
                inp = np.concatenate(
                    [perf_geom, ref_geom, dtw_frame[:, None]], axis=1
                ).astype(np.float32)

                # Target
                if video_id in score_map:
                    target = float(score_map[video_id])
                else:
                    target = float(row.get("dtw_distance_total", 0.0))

                # Chunk
                chunks = _chunk_sequence(inp, self.window, self.hop)
                for chunk in chunks:
                    self._samples.append((chunk, target))

            except Exception as exc:
                logger.warning("Bỏ qua %s/%s: %s", dance_id, video_id, exc)

        if not self._samples:
            raise RuntimeError("TemporalSeqDataset: không có sample nào hợp lệ.")

        # Normalize target về [0, 1] để train ổn định hơn
        if normalize_target:
            targets = np.array([t for _, t in self._samples], dtype=np.float32)
            tmin, tmax = targets.min(), targets.max()
            if tmax > tmin:
                self._samples = [
                    (x, float((t - tmin) / (tmax - tmin))) for x, t in self._samples
                ]
            self._target_range = (float(tmin), float(tmax))
        else:
            self._target_range = (0.0, 1.0)

        logger.info(
            "TemporalSeqDataset: %d samples | target_range=(%.3f, %.3f)",
            len(self._samples), *self._target_range,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        inp, target = self._samples[idx]
        if self.augment and self.rng.random() < 0.4:
            inp = inp + self.rng.normal(0.0, 1.0, size=inp.shape).astype(np.float32)
        x = torch.from_numpy(inp)
        y = torch.tensor(target, dtype=torch.float32)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────────────────────

def get_aistpp_loaders(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    batch_size: int = 32,
    window_frames: int = 60,
    val_ratio: float = 0.15,
    max_files: int = 0,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Tạo train/val DataLoader cho AIST++ pretrain."""
    full_ds = AISTPPDataset(
        cache_dir=cache_dir,
        window_frames=window_frames,
        augment=True,
        seed=seed,
        max_files=max_files,
    )
    n_val = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val
    rng_split = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=rng_split
    )
    # Tắt augment cho val
    val_ds.dataset.augment = False  # type: ignore

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=n_train >= batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    logger.info(
        "AIST++ loaders: train=%d val=%d batch=%d window=%d",
        n_train, n_val, batch_size, window_frames,
    )
    return train_loader, val_loader


def get_temporal_loaders(
    dtw_csv: str | Path = DEFAULT_DTW_CSV,
    poses_dir: str | Path = DEFAULT_POSES_DIR,
    scores_csv: Optional[str | Path] = None,
    batch_size: int = 16,
    window_frames: int = 60,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Tạo train/val DataLoader cho BiLSTM training chính."""
    full_ds = TemporalSeqDataset(
        dtw_csv=dtw_csv,
        poses_dir=poses_dir,
        scores_csv=scores_csv,
        window_frames=window_frames,
        augment=True,
        seed=seed,
    )
    n_val = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val
    rng_split = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=rng_split
    )
    val_ds.dataset.augment = False  # type: ignore

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=n_train >= batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    logger.info(
        "Temporal loaders: train=%d val=%d batch=%d window=%d",
        n_train, n_val, batch_size, window_frames,
    )
    return train_loader, val_loader
