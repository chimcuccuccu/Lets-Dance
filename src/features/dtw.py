"""
Person 2 — DTW alignment / distance.

Person 1 dùng `dtw_align` để time-align trước khi tính diff_sequence.
P2 mở rộng thêm windowed distance + rule engine tuần 3–5.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from dtaidistance import dtw_ndim

    _BACKEND = "dtaidistance"
except ImportError:
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        _BACKEND = "fastdtw"
    except ImportError:
        _BACKEND = "numpy"

logger.debug("DTW backend: %s", _BACKEND)

AlignmentPath = List[Tuple[int, int]]


def _to_2d(seq: np.ndarray) -> np.ndarray:
    """(T, J, 3|4) hoặc (T, D) → (T, D) float64."""
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, :3].reshape(arr.shape[0], -1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected (T,D) or (T,J,C), got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0)


def _naive_dtw(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, AlignmentPath]:
    """DTW O(T1*T2) — ổn với T ≲ 400 sau khi downsample."""
    T1, T2 = len(seq1), len(seq2)
    cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            d = float(np.linalg.norm(seq1[i - 1] - seq2[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # backtrack
    i, j = T1, T2
    path: AlignmentPath = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = (
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        )
        _, i, j = min(choices, key=lambda t: t[0])
    path.reverse()
    return float(cost[T1, T2]), path


def _compute_dtw(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, AlignmentPath]:
    s1, s2 = _to_2d(seq1), _to_2d(seq2)

    if _BACKEND == "dtaidistance":
        dist = float(dtw_ndim.distance(s1, s2))
        path = [(int(i), int(j)) for i, j in dtw_ndim.warping_path(s1, s2)]
        return dist, path

    if _BACKEND == "fastdtw":
        dist, path = fastdtw(s1, s2, dist=euclidean)
        return float(dist), [(int(i), int(j)) for i, j in path]

    return _naive_dtw(s1, s2)


def _downsample_indices(T: int, max_frames: int) -> np.ndarray:
    if T <= max_frames:
        return np.arange(T, dtype=np.int64)
    return np.linspace(0, T - 1, max_frames).astype(np.int64)


def dtw_align(
    performer_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    max_frames: int = 200,
) -> Tuple[AlignmentPath, float]:
    """
    Align 2 chuỗi (pose/geometry) bằng DTW.

    Returns:
        alignment_path: [(i_performer, i_reference), ...]
        dtw_distance: khoảng cách thô (chưa chuẩn hoá)
    """
    p = _to_2d(performer_features)
    r = _to_2d(reference_features)
    idx_p = _downsample_indices(len(p), max_frames)
    idx_r = _downsample_indices(len(r), max_frames)

    dist, path_ds = _compute_dtw(p[idx_p], r[idx_r])
    # Map về index gốc
    path = [(int(idx_p[i]), int(idx_r[j])) for i, j in path_ds]
    # Chuẩn hoá nhẹ theo độ dài path để so sánh giữa clip
    norm_dist = dist / max(len(path), 1)
    return path, float(norm_dist)


def dtw_distance_total(
    performer_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    max_frames: int = 200,
) -> float:
    _, dist = dtw_align(performer_features, reference_features, max_frames=max_frames)
    return dist


def dtw_distance_windowed(
    performer_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    window: int = 60,
    hop: Optional[int] = None,
    max_frames_per_window: int = 80,
) -> np.ndarray:
    """
    DTW theo cửa sổ trượt (mặc định ~2s nếu 30fps → window=60).
    Trả về 1 float / đoạn — Person 3 dùng gợi ý gán nhãn.
    """
    hop = hop or max(window // 2, 1)
    p = _to_2d(performer_features)
    r = _to_2d(reference_features)
    # Resample ref về cùng T thô để cửa sổ khớp index (P1 vẫn dùng full-path riêng)
    T = min(len(p), len(r))
    if len(p) != T:
        idx = np.linspace(0, len(p) - 1, T).astype(np.int64)
        p = p[idx]
    if len(r) != T:
        idx = np.linspace(0, len(r) - 1, T).astype(np.int64)
        r = r[idx]

    distances = []
    start = 0
    while start + window <= T:
        end = start + window
        _, d = dtw_align(p[start:end], r[start:end], max_frames=max_frames_per_window)
        distances.append(d)
        start += hop
    if not distances and T > 0:
        _, d = dtw_align(p, r, max_frames=max_frames_per_window)
        distances.append(d)
    return np.asarray(distances, dtype=np.float64)


def flag_suspicious_segments(
    dtw_windowed_distances: np.ndarray,
    threshold_multiplier: float = 1.5,
    absolute_threshold: float = 0.0
) -> np.ndarray:
    """
    Rule engine cơ bản: Cắm cờ 'nghi ngờ' cho đoạn có dtw_distance cao bất thường.
    Person 3 sẽ dựa vào cờ này để tập trung gán nhãn loại lỗi chi tiết.
    
    Args:
        dtw_windowed_distances: mảng float khoảng cách DTW từng đoạn
        threshold_multiplier: hệ số nhân của độ lệch chuẩn để tính ngưỡng
        absolute_threshold: ngưỡng tối thiểu tĩnh (phòng khi std = 0)
    Returns:
        Mảng int (0 hoặc 1) cùng độ dài với mảng đầu vào.
    """
    if len(dtw_windowed_distances) == 0:
        return np.array([], dtype=int)
        
    distances = np.asarray(dtw_windowed_distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    threshold = max(mean_dist + threshold_multiplier * std_dist, absolute_threshold)
    
    # 1 nếu vượt ngưỡng, ngược lại 0
    flags = (distances > threshold).astype(int)
    return flags
