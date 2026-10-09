"""Person 3 — dataset loader: diff_sequence cửa sổ trượt + nhãn lỗi multi-label.

Lưới thời gian trùng `annotations/suggestions.csv` và `label_pipeline`
(60 frame, bước 30 frame, 30 fps → đoạn 2 giây, bước 1 giây).

`diff_sequence` lưu cả video, thường đã resample về T cố định (mặc định 150).
Mỗi cửa sổ được cắt theo tỉ lệ thời gian trên trục đó rồi resample về
`window_frames` để batch đồng nhất.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# Cùng lưới với src/error_dl/label_pipeline.py và src/features/run_dtw_all.py
FPS = 30
WINDOW_FRAMES_SRC = 60
HOP_FRAMES_SRC = 30
WINDOW_SEC = WINDOW_FRAMES_SRC / FPS
HOP_SEC = HOP_FRAMES_SRC / FPS

ERROR_TYPES: Tuple[str, ...] = (
    "off_beat",
    "wrong_move",
    "low_amplitude",
    "wrong_direction",
    "missed_move",
)
TYPE_INDEX = {name: i for i, name in enumerate(ERROR_TYPES)}

NUM_JOINTS = 33
NUM_DIMS = 3
DEFAULT_WINDOW_FRAMES = 32
SYNTH_DIFF_FRAMES = 150


def resolve_project_path(path: str | Path) -> Path:
    """Ưu tiên đường dẫn đang có (cwd hoặc tuyệt đối), không thì tính từ gốc repo."""
    p = Path(path)
    if p.is_absolute():
        return p
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd
    return ROOT / p


def _is_reference(video_id: str) -> bool:
    return "REF" in str(video_id).upper()


def _person_from_video(video_id: str) -> str:
    for part in str(video_id).split("_"):
        if len(part) > 1 and part[0] == "P" and part[1:].isdigit():
            return part
    return str(video_id)


def parse_error_types(value) -> List[int]:
    """`error_type` một tên, hoặc nhiều tên cách nhau bởi `|`."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    found: List[int] = []
    for part in str(value).split("|"):
        name = part.strip()
        if not name or name.lower() == "nan":
            continue
        idx = TYPE_INDEX.get(name)
        if idx is None:
            logger.warning("Bỏ error_type không có trong taxonomy: %s", name)
            continue
        if idx not in found:
            found.append(idx)
    return found


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_window_table(path: str | Path) -> pd.DataFrame:
    """Mọi cửa sổ performer: suggestions (dài) hoặc dtw_features (rộng)."""
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy bảng cửa sổ: {path}")
    raw = _read_csv(path)
    if "start_time" in raw.columns and "end_time" in raw.columns:
        df = raw.copy()
    else:
        df = _windows_from_wide_dtw(raw)
    need = {"dance_id", "video_id", "start_time", "end_time"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột {sorted(missing)}")
    df["dance_id"] = df["dance_id"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df = df.loc[df["end_time"] > df["start_time"]].copy()
    df = df.loc[~df["video_id"].map(_is_reference)].copy()
    if "window_id" not in df.columns:
        df["window_id"] = df.groupby(["dance_id", "video_id"], sort=False).cumcount()
    df["window_id"] = pd.to_numeric(df["window_id"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(["dance_id", "video_id", "window_id"], keep="first")
    df = df.sort_values(["dance_id", "video_id", "window_id"], kind="mergesort")
    return df.reset_index(drop=True)


def _windows_from_wide_dtw(wide: pd.DataFrame) -> pd.DataFrame:
    seg_cols = [c for c in wide.columns if c.startswith("dtw_seg_")]
    if not seg_cols:
        raise ValueError("CSV cửa sổ không có start_time/end_time cũng không có dtw_seg_*")
    rows = []
    for rec in wide.to_dict(orient="records"):
        for col in seg_cols:
            if pd.isna(rec[col]):
                continue
            window_id = int(col.split("_")[-1])
            start_frame = window_id * HOP_FRAMES_SRC
            end_frame = start_frame + WINDOW_FRAMES_SRC
            rows.append(
                {
                    "dance_id": rec["dance_id"],
                    "video_id": rec["video_id"],
                    "window_id": window_id,
                    "start_time": start_frame / FPS,
                    "end_time": end_frame / FPS,
                }
            )
    return pd.DataFrame(rows)


def load_error_label_table(path: str | Path) -> pd.DataFrame:
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy error_labels: {path}")
    df = _read_csv(path)
    need = {"dance_id", "video_id", "start_time", "end_time", "error_type"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột {sorted(missing)}")
    df = df.copy()
    df["dance_id"] = df["dance_id"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
    df = df.dropna(subset=["start_time", "end_time", "error_type"])
    df = df.loc[df["end_time"] > df["start_time"]].copy()
    return df.reset_index(drop=True)


def resolve_diff_path(diffs_dir: str | Path, dance_id: str, video_id: str) -> Optional[Path]:
    root = resolve_project_path(diffs_dir)
    candidates = [
        root / dance_id / f"{video_id}_diff.npy",
        root / f"{dance_id}_{video_id}_diff.npy",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def as_diff_array(arr: np.ndarray) -> np.ndarray:
    """Chuẩn về (T, 33, 3) float32. Nhận thêm (T, 99) hoặc (T, 33, 4)."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2 and x.shape[1] == NUM_JOINTS * NUM_DIMS:
        x = x.reshape(x.shape[0], NUM_JOINTS, NUM_DIMS)
    if x.ndim != 3:
        raise ValueError(f"diff_sequence phải là 3 chiều, nhận {x.shape}")
    if x.shape[-1] == 4 and x.shape[1] == NUM_JOINTS:
        x = x[..., :3]
    if x.shape[1] != NUM_JOINTS or x.shape[2] != NUM_DIMS:
        raise ValueError(f"diff_sequence phải là (T, 33, 3), nhận {x.shape}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def resample_window(
    diff: np.ndarray,
    start_time: float,
    end_time: float,
    duration: float,
    out_frames: int,
) -> np.ndarray:
    """Cắt [start_time, end_time] trên diff phủ [0, duration], resample về out_frames."""
    if out_frames < 1:
        raise ValueError("out_frames phải >= 1")
    x = as_diff_array(diff)
    length = x.shape[0]
    if length < 1:
        raise ValueError("diff_sequence rỗng")
    span = float(duration) if duration and duration > 0 else 1.0
    t0 = float(np.clip(start_time / span, 0.0, 1.0)) * (length - 1)
    t1 = float(np.clip(end_time / span, 0.0, 1.0)) * (length - 1)
    if t1 < t0:
        t0, t1 = t1, t0
    if out_frames == 1 or length == 1:
        idx = int(np.clip(round((t0 + t1) / 2.0), 0, length - 1))
        return np.repeat(x[idx : idx + 1], out_frames, axis=0)
    src = np.linspace(t0, t1, out_frames, dtype=np.float64)
    left = np.clip(np.floor(src).astype(np.int64), 0, length - 1)
    right = np.clip(left + 1, 0, length - 1)
    alpha = (src - left).astype(np.float32).reshape(-1, 1, 1)
    return ((1.0 - alpha) * x[left] + alpha * x[right]).astype(np.float32)


def _label_applies(label_start: float, label_end: float, win_start: float, win_end: float) -> bool:
    """Nhãn gắn vào cửa sổ khi phần giao > 50% đoạn ngắn hơn.

    Cửa sổ 2 giây bước 1 giây chạm nhau đúng 50% — không lấy phần chạm đó,
    để nhãn của đoạn [t, t+2] không bị gán sang đoạn kế bên.
    """
    overlap = min(label_end, win_end) - max(label_start, win_start)
    if overlap <= 0:
        return False
    base = min(label_end - label_start, win_end - win_start)
    if base <= 0:
        return False
    return (overlap / base) > 0.5


def _pose_duration_sec(poses_dir: Path, dance_id: str, video_id: str) -> Optional[float]:
    if not poses_dir:
        return None
    for path in (
        poses_dir / dance_id / f"{video_id}.npy",
        poses_dir / f"{dance_id}_{video_id}.npy",
    ):
        if not path.is_file():
            continue
        length = int(np.load(path, mmap_mode="r").shape[0])
        if length > 1:
            return length / FPS
    return None


def _label_buckets(labels: pd.DataFrame) -> Dict[Tuple[str, str], List[Tuple[float, float, List[int]]]]:
    buckets: Dict[Tuple[str, str], List[Tuple[float, float, List[int]]]] = {}
    for rec in labels.itertuples(index=False):
        types = parse_error_types(rec.error_type)
        if not types:
            continue
        key = (str(rec.dance_id), str(rec.video_id))
        buckets.setdefault(key, []).append((float(rec.start_time), float(rec.end_time), types))
    return buckets


def _multihot(
    buckets: Dict[Tuple[str, str], List[Tuple[float, float, List[int]]]],
    dance_id: str,
    video_id: str,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    hot = np.zeros(len(ERROR_TYPES), dtype=np.float32)
    for start, end, types in buckets.get((dance_id, video_id), ()):
        if _label_applies(start, end, start_time, end_time):
            for idx in types:
                hot[idx] = 1.0
    return hot


def _count_matched_labels(
    labels: pd.DataFrame,
    windows: pd.DataFrame,
) -> int:
    by_video: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for rec in windows.itertuples(index=False):
        key = (str(rec.dance_id), str(rec.video_id))
        by_video.setdefault(key, []).append((float(rec.start_time), float(rec.end_time)))
    matched = 0
    for rec in labels.itertuples(index=False):
        key = (str(rec.dance_id), str(rec.video_id))
        spans = by_video.get(key, ())
        if any(_label_applies(float(rec.start_time), float(rec.end_time), a, b) for a, b in spans):
            matched += 1
    return matched


class ErrorWindowDataset(Dataset):
    """Một mẫu = một cửa sổ diff (window_frames, 33, 3) + vector multi-hot 5 lớp.

    Cửa sổ không chồng nhãn nào thì target = 0 (negative). BCE cần những mẫu này.
    """

    def __init__(
        self,
        diffs_dir: str | Path = "poses/diffs",
        labels_csv: str | Path = "annotations/error_labels.csv",
        windows_csv: str | Path = "annotations/suggestions.csv",
        scores_csv: str | Path = "annotations/scores.csv",
        poses_dir: str | Path = "poses",
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        synthetic: bool = False,
        seed: int = 42,
    ):
        self.window_frames = int(window_frames)
        self.synthetic = synthetic
        self.diffs_dir = resolve_project_path(diffs_dir)
        self.poses_dir = resolve_project_path(poses_dir)
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

        windows = load_window_table(windows_csv)
        labels = load_error_label_table(labels_csv)
        buckets = _label_buckets(labels)
        person = _person_map(scores_csv)

        durations: Dict[Tuple[str, str], float] = {}
        pose_hits = 0
        for (dance_id, video_id), group in windows.groupby(["dance_id", "video_id"], sort=False):
            max_end = float(group["end_time"].max())
            pose_dur = _pose_duration_sec(self.poses_dir, str(dance_id), str(video_id))
            if pose_dur is not None:
                pose_hits += 1
                durations[(str(dance_id), str(video_id))] = max(pose_dur, max_end)
            else:
                durations[(str(dance_id), str(video_id))] = max_end

        rows = []
        targets = []
        missing_videos = set()
        for rec in windows.itertuples(index=False):
            dance_id = str(rec.dance_id)
            video_id = str(rec.video_id)
            key = (dance_id, video_id)
            path = None if synthetic else resolve_diff_path(self.diffs_dir, dance_id, video_id)
            if path is None and not synthetic:
                missing_videos.add(key)
                continue
            rows.append(
                {
                    "dance_id": dance_id,
                    "video_id": video_id,
                    "person_id": person.get(key, _person_from_video(video_id)),
                    "window_id": int(rec.window_id),
                    "start_time": float(rec.start_time),
                    "end_time": float(rec.end_time),
                    "duration": float(durations[key]),
                    "diff_path": "" if path is None else str(path),
                }
            )
            targets.append(_multihot(buckets, dance_id, video_id, float(rec.start_time), float(rec.end_time)))

        if not rows:
            raise FileNotFoundError(
                f"Không tìm thấy diff_sequence (*.npy) trong {self.diffs_dir}. "
                "File diff không nằm trong Git. Tạo bằng: python -m src.preprocessing.build_diffs. "
                "Kiểm tra vòng train khi chưa có pose: python -m src.error_dl.train --smoke"
            )

        self.meta = pd.DataFrame(rows)
        self.targets = np.stack(targets).astype(np.float32)
        self.person_ids = self.meta["person_id"].astype(str).tolist()
        if float(self.targets.sum()) <= 0:
            raise RuntimeError(
                "Không gán được nhãn lỗi nào lên cửa sổ. Kiểm tra error_labels.csv và suggestions.csv."
            )

        kept = self.meta.groupby(["dance_id", "video_id"]).ngroups
        matched = _count_matched_labels(labels, self.meta)
        per_class = self.targets.sum(axis=0)
        prevalence = ", ".join(
            f"{name}={int(per_class[i])}" for i, name in enumerate(ERROR_TYPES)
        )
        logger.info(
            "Error windows: %d cửa sổ, %d video, nhãn khớp %d/%d dòng | %s | có ≥1 lỗi: %d",
            len(self.meta),
            kept,
            matched,
            len(labels),
            prevalence,
            int((self.targets.sum(axis=1) > 0).sum()),
        )
        if missing_videos and not synthetic:
            logger.warning(
                "Bỏ %d video vì thiếu file diff trong %s",
                len(missing_videos),
                self.diffs_dir,
            )
        if pose_hits:
            logger.info("Duration lấy từ pose.npy: %d video", pose_hits)
        if synthetic:
            self._build_synthetic(seed)

    def _build_synthetic(self, seed: int) -> None:
        """Diff giả: mỗi loại lỗi cộng một cụm khớp riêng, để vòng train có tín hiệu."""
        rng = np.random.default_rng(seed)
        videos = self.meta.groupby(["dance_id", "video_id"], sort=False)
        for (dance_id, video_id), group in videos:
            diff = rng.normal(0.0, 0.02, (SYNTH_DIFF_FRAMES, NUM_JOINTS, NUM_DIMS)).astype(np.float32)
            for idx in group.index:
                hot = self.targets[int(idx)]
                if hot.sum() <= 0:
                    continue
                rec = self.meta.loc[idx]
                i0, i1 = _index_span(
                    float(rec.start_time),
                    float(rec.end_time),
                    float(rec.duration),
                    SYNTH_DIFF_FRAMES,
                )
                for cls, on in enumerate(hot):
                    if on < 0.5:
                        continue
                    joint0 = cls * 4
                    diff[i0:i1, joint0 : joint0 + 4, :] += 2.0
            self._cache[(str(dance_id), str(video_id))] = diff
        logger.info(
            "Smoke: sinh %d diff giả (%d, 33, 3) trong bộ nhớ, không ghi đè poses/diffs",
            len(self._cache),
            SYNTH_DIFF_FRAMES,
        )

    def __len__(self) -> int:
        return len(self.meta)

    def _load_diff(self, dance_id: str, video_id: str) -> np.ndarray:
        key = (dance_id, video_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = resolve_diff_path(self.diffs_dir, dance_id, video_id)
        if path is None:
            raise FileNotFoundError(f"Thiếu diff cho {dance_id}/{video_id} trong {self.diffs_dir}")
        arr = as_diff_array(np.load(path))
        self._cache[key] = arr
        return arr

    def __getitem__(self, index: int):
        row = self.meta.iloc[int(index)]
        diff = self._load_diff(str(row["dance_id"]), str(row["video_id"]))
        window = resample_window(
            diff,
            float(row["start_time"]),
            float(row["end_time"]),
            float(row["duration"]),
            self.window_frames,
        )
        target = self.targets[int(index)]
        return torch.from_numpy(window), torch.from_numpy(target.copy())


def _index_span(start_time: float, end_time: float, duration: float, length: int) -> Tuple[int, int]:
    span = float(duration) if duration and duration > 0 else 1.0
    t0 = float(np.clip(start_time / span, 0.0, 1.0)) * (length - 1)
    t1 = float(np.clip(end_time / span, 0.0, 1.0)) * (length - 1)
    i0 = int(np.clip(np.floor(min(t0, t1)), 0, length - 1))
    i1 = int(np.clip(max(np.ceil(max(t0, t1)), i0 + 1), 1, length))
    return i0, i1


def _person_map(scores_csv: str | Path) -> Dict[Tuple[str, str], str]:
    path = resolve_project_path(scores_csv)
    if not path.is_file():
        return {}
    df = _read_csv(path)
    if "person_id" not in df.columns:
        return {}
    out = {}
    for rec in df.itertuples(index=False):
        out[(str(rec.dance_id), str(rec.video_id))] = str(rec.person_id)
    return out


def split_indices_by_person(
    person_ids: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Cùng một người không vừa train vừa val."""
    rng = np.random.default_rng(seed)
    people = np.array(sorted(set(person_ids)))
    rng.shuffle(people)
    n_val = max(1, int(round(len(people) * val_ratio))) if len(people) > 1 else 0
    val_people = set(people[:n_val].tolist())
    train_idx = [i for i, person in enumerate(person_ids) if person not in val_people]
    val_idx = [i for i, person in enumerate(person_ids) if person in val_people]
    if train_idx and val_idx:
        return train_idx, val_idx
    order = rng.permutation(len(person_ids))
    cut = max(1, int(len(order) * (1.0 - val_ratio)))
    train_idx = order[:cut].tolist()
    val_idx = order[cut:].tolist() or order[-1:].tolist()
    if train_idx == val_idx and len(order) > 1:
        train_idx = order[:-1].tolist()
        val_idx = order[-1:].tolist()
    return train_idx, val_idx


def positive_weights(targets: np.ndarray) -> np.ndarray:
    """pos_weight BCE = số mẫu âm / số mẫu dương, từng lớp. Lớp không có dương → 1."""
    y = np.asarray(targets, dtype=np.float64)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    weights = np.ones(y.shape[1], dtype=np.float32)
    mask = pos > 0
    weights[mask] = (neg[mask] / pos[mask]).astype(np.float32)
    return weights


def get_error_dataloaders(
    diffs_dir: str = "poses/diffs",
    labels_csv: str = "annotations/error_labels.csv",
    windows_csv: str = "annotations/suggestions.csv",
    scores_csv: str = "annotations/scores.csv",
    poses_dir: str = "poses",
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    synthetic: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, ErrorWindowDataset, List[int], List[int]]:
    dataset = ErrorWindowDataset(
        diffs_dir=diffs_dir,
        labels_csv=labels_csv,
        windows_csv=windows_csv,
        scores_csv=scores_csv,
        poses_dir=poses_dir,
        window_frames=window_frames,
        synthetic=synthetic,
        seed=seed,
    )
    train_idx, val_idx = split_indices_by_person(dataset.person_ids, val_ratio=val_ratio, seed=seed)
    train_people = {dataset.person_ids[i] for i in train_idx}
    val_people = {dataset.person_ids[i] for i in val_idx}
    if train_people & val_people:
        raise RuntimeError("Split person_id bị chồng train/val")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    logger.info(
        "Split person_id: train=%d cửa sổ (%d người) | val=%d cửa sổ (%d người)",
        len(train_idx),
        len(train_people),
        len(val_idx),
        len(val_people),
    )
    return train_loader, val_loader, dataset, train_idx, val_idx
