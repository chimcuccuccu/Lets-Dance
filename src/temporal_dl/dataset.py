"""
Person 2 — Temporal DL Datasets (Tuần 3-5).

Hai dataset:
1. AISTPPDataset  — load từ shared_basic_cache (.npy T×33×4), tính geometry
   features, chunk thành cửa sổ → cho pretrain genre classification.
2. TemporalSeqDataset — ghép (performer_geom, ref_geom, dtw_per_frame) theo
   thời gian → cho training BiLSTM chính.

Tuần 5 bổ sung (xem docs/temporal_week5.md):
  - `build_temporal_sequence()` tách ra module-level để script eval/infer dùng
    *đúng* preprocessing của lúc train (chống lệch train/infer).
  - `use_dtw=False` → input 16 chiều (bỏ kênh dtw) cho kịch bản (b) LSTM-only.
  - `TemporalSampleMeta` + `video_keys`/`video_person` → cho phép split theo
    person_id ở **cấp video**, thay vì random_split cấp window (vốn làm window
    của cùng một video nằm cả train lẫn val).
  - `get_temporal_loaders_grouped()` — factory mới có split official/person/
    groupkfold. `get_temporal_loaders()` giữ nguyên hành vi cũ để Tuần 4
    reproduce được.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple,
)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

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
# D01_P001_T01 → P001
_PERSON_RE = re.compile(r"_(?P<person>P\d+)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "aistpp" / "shared_basic_cache"
DEFAULT_POSES_DIR = PROJECT_ROOT / "poses"
DEFAULT_DTW_CSV   = PROJECT_ROOT / "annotations" / "dtw_features.csv"
DEFAULT_SCORES_CSV      = PROJECT_ROOT / "annotations" / "scores.csv"
DEFAULT_TRAIN_SCORES_CSV = PROJECT_ROOT / "annotations" / "scores_train.csv"
DEFAULT_VAL_SCORES_CSV   = PROJECT_ROOT / "annotations" / "scores_test.csv"

# Geometry feature dimension (8 góc khớp — xem geometry.py)
GEOM_DIM = 8
# Input dims: perf_geom + ref_geom (+ dtw_per_frame nếu use_dtw)
NODTW_INPUT_DIM = GEOM_DIM * 2       # 16
DTW_INPUT_DIM   = GEOM_DIM * 2 + 1   # 17

# Cột target nằm trong input → train trên nó là tự tham chiếu (xem Tuần 4 erratum)
SELF_REFERENTIAL_TARGETS = ("dtw_distance_total",)

VideoKey = Tuple[str, str]


def _parse_genre(stem: str) -> int:
    """Trả về genre_id từ tên file. Fallback = 0 nếu không tìm thấy."""
    m = _GENRE_RE.match(stem)
    if m:
        return GENRE_TO_ID.get(m.group("genre"), 0)
    return 0


def parse_person_id(video_id: str) -> str:
    """Suy person_id từ video_id (D01_P001_T01 → P001). Fallback = video_id."""
    m = _PERSON_RE.search(video_id)
    return m.group("person") if m else video_id


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


def sorted_seg_columns(
    source: pd.DataFrame | pd.Series | Mapping[str, Any],
    prefix: str = "dtw_seg_",
) -> List[str]:
    """
    Cột `dtw_seg_*` / `flag_seg_*` sắp theo số nguyên ở cuối tên.

    `annotations/dtw_features.csv` có thứ tự cột **ragged** (pandas chèn
    `dtw_seg_9` sau `flag_seg_8` khi gặp video dài hơn), nên không được tin vào
    thứ tự cột trong file — phải sort theo index.
    """
    if isinstance(source, pd.DataFrame):
        names: Iterable[str] = source.columns
    elif isinstance(source, pd.Series):
        names = source.index
    else:
        names = source.keys()
    cols = [str(c) for c in names if str(c).startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def build_temporal_sequence(
    dance_id: str,
    video_id: str,
    dtw_row: Mapping[str, Any],
    poses_dir: str | Path = DEFAULT_POSES_DIR,
    *,
    use_dtw: bool = True,
    seg_cols: Optional[Sequence[str]] = None,
) -> Optional[np.ndarray]:
    """
    Dựng input sequence cho một video.

    Returns:
        (T, 17) nếu use_dtw=True  — ghép [perf_geom(8), ref_geom(8), dtw_frame(1)]
        (T, 16) nếu use_dtw=False — ghép [perf_geom(8), ref_geom(8)]
        None nếu thiếu pose performer hoặc reference.

    Đây là *nguồn chân lý duy nhất* cho cách dựng input: cả
    `TemporalSeqDataset` lúc train và `src/temporal_dl/infer.py` lúc eval đều
    gọi hàm này, nên không thể lệch preprocessing giữa train và infer.
    """
    poses_dir = Path(poses_dir)
    dance_dir = poses_dir / dance_id
    if not dance_dir.is_dir():
        return None

    perf_path = dance_dir / f"{video_id}.npy"
    ref_candidates = list(dance_dir.glob(f"{dance_id}_ref.npy"))
    ref_candidates += list(dance_dir.glob("*ref*.npy"))
    if not perf_path.is_file() or not ref_candidates:
        return None
    ref_path = ref_candidates[0]

    perf_pose = np.load(perf_path).astype(np.float32)  # (T,33,4)
    ref_pose  = np.load(ref_path).astype(np.float32)   # (Tr,33,4)

    perf_geom = compute_geometry_features(perf_pose)   # (T,8)
    ref_geom  = compute_geometry_features(ref_pose)    # (Tr,8)

    # Align độ dài về min(T, Tr)
    T = min(perf_geom.shape[0], ref_geom.shape[0])
    if T <= 0:
        return None
    perf_geom = perf_geom[:T]
    ref_geom  = ref_geom[:T]

    parts = [perf_geom, ref_geom]
    if use_dtw:
        if seg_cols is None:
            seg_cols = sorted_seg_columns(dtw_row)
        seg_vals = [
            float(dtw_row[c]) for c in seg_cols
            if c in dtw_row and pd.notna(dtw_row[c])
        ]
        dtw_frame = _expand_dtw_to_frames(seg_vals, T)  # (T,)
        parts.append(dtw_frame[:, None])

    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass(frozen=True)
class TemporalSampleMeta:
    """Danh tính của một window — cần cho split theo person và gộp theo video."""
    dance_id: str
    video_id: str
    person_id: str
    video_index: int
    window_index: int
    n_windows: int
    target_raw: float


class TemporalSeqDataset(Dataset):
    """
    Dataset cho BiLSTM chính.

    Mỗi sample là một *cửa sổ* của một video:
    Input : (window, 17) = ghép(perf_geom, ref_geom, dtw_per_frame)
            (window, 16) nếu use_dtw=False
    Target: giá trị cột `target_col` của video đó.

    Lưu ý quan trọng (Tuần 5): dataset phát sample theo **window**, nên không
    bao giờ được chia train/val bằng `random_split` trên dataset này — dùng
    `get_temporal_loaders_grouped()` để chia ở cấp video.
    """

    INPUT_DIM = DTW_INPUT_DIM   # 17 (giữ cho code cũ; xem self.input_dim)

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
        *,
        target_col: str = "khop_nhip",
        use_dtw: bool = True,
        allow_keys: Optional[Iterable[VideoKey]] = None,
        require_target: bool = False,
        target_range: Optional[Tuple[float, float]] = None,
    ):
        self.poses_dir = Path(poses_dir)
        self.window = window_frames
        self.hop = hop_frames or window_frames
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.target_col = target_col
        self.use_dtw = use_dtw
        self.input_dim = DTW_INPUT_DIM if use_dtw else NODTW_INPUT_DIM
        # Chỉ augment các index này (None = tất cả). Cho phép 1 dataset object
        # phục vụ cả train Subset (augment) và val Subset (sạch).
        self.augment_indices: Optional[Set[int]] = None
        self.return_index: bool = False

        # Load dtw_features.csv
        dtw_csv = Path(dtw_csv)
        if not dtw_csv.is_file():
            raise FileNotFoundError(f"DTW CSV không tồn tại: {dtw_csv}")
        dtw_df = pd.read_csv(dtw_csv)

        # Load scores.csv nếu có → map (dance_id, video_id) -> (target, person_id)
        score_map: Dict[VideoKey, Tuple[float, str]] = {}
        if scores_csv is not None and str(scores_csv) != "":
            sc = pd.read_csv(scores_csv)
            sc.columns = [str(c).strip().strip('"').lstrip("﻿") for c in sc.columns]
            missing = {"dance_id", "video_id", target_col} - set(sc.columns)
            if missing:
                raise ValueError(
                    f"{scores_csv} thiếu cột {sorted(missing)} (cần cho target_col="
                    f"{target_col!r})"
                )
            for _, r in sc.iterrows():
                key = (str(r["dance_id"]), str(r["video_id"]))
                person = str(r.get("person_id", "") or parse_person_id(key[1]))
                score_map[key] = (float(r[target_col]), person)
            logger.info("Loaded %d nhãn %s từ %s", len(score_map), target_col, scores_csv)
        elif target_col not in SELF_REFERENTIAL_TARGETS:
            # Không có scores_csv mà target là cột điểm → trước đây rơi về
            # dtw_distance_total một cách âm thầm (lỗi P2 của Tuần 4). Không
            # cho phép nữa.
            raise ValueError(
                f"target_col={target_col!r} cần scores_csv, nhưng scores_csv=None. "
                f"Truyền scores_csv=annotations/scores.csv, hoặc dùng "
                f"target_col='dtw_distance_total' (đường legacy Tuần 4)."
            )

        if target_col in SELF_REFERENTIAL_TARGETS and use_dtw:
            logger.warning(
                "⚠️  target_col=%r VÀ use_dtw=True: target này cũng nằm trong input "
                "(kênh %d) → model tự tham chiếu, val_mae KHÔNG phải số "
                "generalization. Đây là lỗi P2 của Tuần 4; chỉ dùng để reproduce.",
                target_col, DTW_INPUT_DIM - 1,
            )

        seg_cols = sorted_seg_columns(dtw_df)
        allow: Optional[Set[VideoKey]] = set(allow_keys) if allow_keys is not None else None

        self._samples: List[Tuple[np.ndarray, float]] = []
        self.meta: List[TemporalSampleMeta] = []
        self.video_keys: List[VideoKey] = []
        self.video_person: List[str] = []
        self.video_dance: List[str] = []
        self.dropped_videos: List[Tuple[VideoKey, str]] = []

        n_no_target = 0
        for _, row in dtw_df.iterrows():
            dance_id = str(row.get("dance_id", ""))
            video_id = str(row.get("video_id", ""))
            key = (dance_id, video_id)

            # Bỏ qua file reference
            if "ref" in video_id.lower():
                continue
            if allow is not None and key not in allow:
                continue

            # Target
            if key in score_map:
                target_raw, person_id = score_map[key]
            elif target_col in SELF_REFERENTIAL_TARGETS:
                target_raw = float(row.get(target_col, 0.0))
                person_id = parse_person_id(video_id)
            else:
                n_no_target += 1
                self.dropped_videos.append((key, f"thiếu nhãn {target_col}"))
                continue

            if require_target and key not in score_map:
                n_no_target += 1
                self.dropped_videos.append((key, f"thiếu nhãn {target_col}"))
                continue

            try:
                inp = build_temporal_sequence(
                    dance_id, video_id, row,
                    poses_dir=self.poses_dir,
                    use_dtw=use_dtw,
                    seg_cols=seg_cols,
                )
            except Exception as exc:
                logger.warning("Bỏ qua %s/%s: %s", dance_id, video_id, exc)
                self.dropped_videos.append((key, f"lỗi dựng sequence: {exc}"))
                continue
            if inp is None:
                self.dropped_videos.append((key, "thiếu pose performer hoặc reference"))
                continue

            chunks = _chunk_sequence(inp, self.window, self.hop)
            video_index = len(self.video_keys)
            self.video_keys.append(key)
            self.video_person.append(person_id)
            self.video_dance.append(dance_id)
            n_win = len(chunks)
            for w_idx, chunk in enumerate(chunks):
                self._samples.append((chunk, float(target_raw)))
                self.meta.append(
                    TemporalSampleMeta(
                        dance_id=dance_id,
                        video_id=video_id,
                        person_id=person_id,
                        video_index=video_index,
                        window_index=w_idx,
                        n_windows=n_win,
                        target_raw=float(target_raw),
                    )
                )

        if not self._samples:
            raise RuntimeError("TemporalSeqDataset: không có sample nào hợp lệ.")

        if n_no_target:
            logger.info("Bỏ %d video không có nhãn %s", n_no_target, target_col)

        # Index window theo video (dùng cho split cấp video)
        self._windows_by_video: Dict[int, List[int]] = {}
        for i, m in enumerate(self.meta):
            self._windows_by_video.setdefault(m.video_index, []).append(i)

        # Normalize target về [0, 1] để train ổn định hơn
        if target_range is not None:
            self.set_target_range(*target_range)
        elif normalize_target:
            targets = np.array([m.target_raw for m in self.meta], dtype=np.float32)
            self.set_target_range(float(targets.min()), float(targets.max()))
        else:
            self.target_range = (0.0, 1.0)
            self._target_range = self.target_range

        logger.info(
            "TemporalSeqDataset: %d window / %d video | input_dim=%d | target=%s "
            "| range=(%.3f, %.3f)",
            len(self._samples), len(self.video_keys), self.input_dim,
            target_col, *self.target_range,
        )

    # ---- helpers ----------------------------------------------------------

    def windows_of_video(self, video_index: int) -> List[int]:
        """Index các window thuộc một video."""
        return list(self._windows_by_video.get(video_index, ()))

    def set_target_range(self, tmin: float, tmax: float) -> None:
        """
        Normalize target về [0,1] theo range cho trước (luôn tính lại từ
        `target_raw`, nên gọi nhiều lần vẫn đúng).

        Dùng min/max của **tập train** để không rò nhãn val vào quá trình scale.
        """
        span = float(tmax) - float(tmin)
        if span <= 0:
            span = 1.0
        self._samples = [
            (x, float((m.target_raw - tmin) / span))
            for (x, _), m in zip(self._samples, self.meta)
        ]
        self.target_range = (float(tmin), float(tmax))
        self._target_range = self.target_range   # alias tương thích code cũ

    def denormalize(self, y):
        """Đưa prediction/target đã normalize về thang điểm gốc."""
        tmin, tmax = self.target_range
        span = tmax - tmin
        if span <= 0:
            span = 1.0
        if isinstance(y, np.ndarray):
            return y * span + tmin
        return float(y) * span + tmin

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        inp, target = self._samples[idx]
        do_aug = self.augment and (
            self.augment_indices is None or idx in self.augment_indices
        )
        if do_aug and self.rng.random() < 0.4:
            inp = inp + self.rng.normal(0.0, 1.0, size=inp.shape).astype(np.float32)
        x = torch.from_numpy(inp)
        y = torch.tensor(target, dtype=torch.float32)
        if self.return_index:
            return x, y, idx
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
    """
    Factory LEGACY của Tuần 4 — giữ nguyên hành vi để reproduce được.

    ⚠️  Chia train/val bằng `random_split` trên **window**, nên window của cùng
    một video nằm cả hai phía → val_mae từ factory này KHÔNG phải số
    generalization (lỗi P1, xem docs/temporal_week5.md). Dùng
    `get_temporal_loaders_grouped()` cho mọi việc mới.
    """
    logger.warning(
        "⚠️  get_temporal_loaders(): random_split cấp window → leakage giữa các "
        "window cùng video. Chỉ dùng để reproduce Tuần 4; việc mới dùng "
        "get_temporal_loaders_grouped()."
    )
    # Khi không có scores_csv, Tuần 4 train trên dtw_distance_total.
    target_col = "khop_nhip" if scores_csv else "dtw_distance_total"
    full_ds = TemporalSeqDataset(
        dtw_csv=dtw_csv,
        poses_dir=poses_dir,
        scores_csv=scores_csv,
        window_frames=window_frames,
        augment=True,
        seed=seed,
        target_col=target_col,
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


def _read_video_keys(csv_path: str | Path) -> List[VideoKey]:
    """Đọc (dance_id, video_id) từ một file scores*.csv."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().strip('"').lstrip("﻿") for c in df.columns]
    if not {"dance_id", "video_id"} <= set(df.columns):
        raise ValueError(f"{csv_path} thiếu cột dance_id/video_id")
    return [(str(d), str(v)) for d, v in zip(df["dance_id"], df["video_id"])]


def split_video_indices(
    ds: TemporalSeqDataset,
    *,
    split: str = "official",
    train_csv: str | Path = DEFAULT_TRAIN_SCORES_CSV,
    val_csv: str | Path = DEFAULT_VAL_SCORES_CSV,
    fold: Optional[int] = None,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Chia **video index** thành train/val. Không bao giờ chia ở cấp window.

    split:
      "official"   — theo annotations/scores_train.csv / scores_test.csv (183/34)
      "person"     — split_indices_by_person của Person 1 (read-only import)
      "groupkfold" — GroupKFold theo person_id, lấy fold thứ `fold`
    """
    n_video = len(ds.video_keys)
    key_to_idx = {k: i for i, k in enumerate(ds.video_keys)}

    if split == "official":
        train_keys = set(_read_video_keys(train_csv))
        val_keys = set(_read_video_keys(val_csv))
        overlap = train_keys & val_keys
        if overlap:
            raise ValueError(
                f"{train_csv} và {val_csv} có {len(overlap)} video trùng nhau"
            )
        train_idx = [key_to_idx[k] for k in ds.video_keys if k in train_keys]
        val_idx = [key_to_idx[k] for k in ds.video_keys if k in val_keys]
        train_idx = sorted(set(train_idx))
        val_idx = sorted(set(val_idx))
    elif split == "person":
        from src.spatial_dl.dataset import split_indices_by_person

        train_idx, val_idx = split_indices_by_person(
            ds.video_person, val_ratio=val_ratio, seed=seed
        )
    elif split == "groupkfold":
        from sklearn.model_selection import GroupKFold

        if fold is None:
            raise ValueError("split='groupkfold' cần tham số fold")
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold phải trong [0, {n_folds}), nhận {fold}")
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(np.arange(n_video), groups=ds.video_person))
        tr, va = splits[fold]
        train_idx, val_idx = sorted(tr.tolist()), sorted(va.tolist())
    else:
        raise ValueError(f"split không hợp lệ: {split!r}")

    if not train_idx or not val_idx:
        raise RuntimeError(
            f"split={split!r} cho tập rỗng: train={len(train_idx)} val={len(val_idx)}. "
            f"Dataset có {n_video} video."
        )

    # S2/S3 — assert rời nhau ở cả cấp video và cấp person
    if set(train_idx) & set(val_idx):
        raise RuntimeError("Video xuất hiện ở cả train và val")
    train_persons = {ds.video_person[i] for i in train_idx}
    val_persons = {ds.video_person[i] for i in val_idx}
    person_overlap = train_persons & val_persons
    if person_overlap:
        raise RuntimeError(
            f"Person xuất hiện ở cả train và val: {sorted(person_overlap)}"
        )
    return train_idx, val_idx


def get_temporal_loaders_grouped(
    dtw_csv: str | Path = DEFAULT_DTW_CSV,
    poses_dir: str | Path = DEFAULT_POSES_DIR,
    scores_csv: str | Path = DEFAULT_SCORES_CSV,
    *,
    target_col: str = "khop_nhip",
    split: str = "official",
    train_csv: str | Path = DEFAULT_TRAIN_SCORES_CSV,
    val_csv: str | Path = DEFAULT_VAL_SCORES_CSV,
    fold: Optional[int] = None,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    window_frames: int = 60,
    hop_frames: Optional[int] = None,
    use_dtw: bool = True,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Factory Tuần 5 — chia train/val ở **cấp video**, theo person_id.

    Khác `get_temporal_loaders()` ở 3 điểm quyết định tính hợp lệ của mọi số đo:
      1. Mọi window của một video nằm trọn một phía (hết leakage P1).
      2. Train/val rời nhau theo person_id, không chỉ theo video.
      3. Target normalize bằng min/max của **tập train**.

    Returns (train_loader, val_loader, meta). `meta["train_videos"]` đi vào
    checkpoint config để script eval assert được không có leakage encoder.
    """
    ds = TemporalSeqDataset(
        dtw_csv=dtw_csv,
        poses_dir=poses_dir,
        scores_csv=scores_csv,
        window_frames=window_frames,
        hop_frames=hop_frames,
        augment=augment,
        seed=seed,
        normalize_target=False,
        target_col=target_col,
        use_dtw=use_dtw,
        require_target=True,
    )

    train_v, val_v = split_video_indices(
        ds, split=split, train_csv=train_csv, val_csv=val_csv,
        fold=fold, n_folds=n_folds, val_ratio=val_ratio, seed=seed,
    )

    train_idx = [i for v in train_v for i in ds.windows_of_video(v)]
    val_idx = [i for v in val_v for i in ds.windows_of_video(v)]
    if set(train_idx) & set(val_idx):
        raise RuntimeError("Window xuất hiện ở cả train và val")

    # Target range chỉ từ window train
    train_targets = np.array([ds.meta[i].target_raw for i in train_idx], dtype=np.float64)
    ds.set_target_range(float(train_targets.min()), float(train_targets.max()))

    ds.augment_indices = set(train_idx) if augment else set()

    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=len(train_idx) >= batch_size,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    train_persons = sorted({ds.video_person[i] for i in train_v})
    val_persons = sorted({ds.video_person[i] for i in val_v})
    meta: Dict[str, Any] = {
        "split": split,
        "fold": fold,
        "n_folds": n_folds if split == "groupkfold" else None,
        "target_col": target_col,
        "input_dim": ds.input_dim,
        "use_dtw": use_dtw,
        "window_frames": window_frames,
        "hop_frames": ds.hop,
        "target_range": list(ds.target_range),
        "seed": seed,
        "train_videos": [list(ds.video_keys[i]) for i in train_v],
        "val_videos": [list(ds.video_keys[i]) for i in val_v],
        "train_persons": train_persons,
        "val_persons": val_persons,
        "n_train_videos": len(train_v),
        "n_val_videos": len(val_v),
        "n_train_windows": len(train_idx),
        "n_val_windows": len(val_idx),
        "dropped_videos": [[list(k), why] for k, why in ds.dropped_videos],
    }

    logger.info(
        "Grouped loaders [%s%s]: video %d/%d | window %d/%d | person %d/%d "
        "| key_overlap=0 person_overlap=0 | target_range=(%.2f, %.2f)",
        split, f" fold={fold}" if fold is not None else "",
        len(train_v), len(val_v), len(train_idx), len(val_idx),
        len(train_persons), len(val_persons), *ds.target_range,
    )
    logger.info("Val persons: %s", val_persons)
    return train_loader, val_loader, meta
