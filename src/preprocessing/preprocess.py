"""Person 1 — Preprocessing & Alignment (tiền xử lý + diff_sequence)."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

AlignmentPath = Sequence[Union[Tuple[int, int], Sequence[int]]]


def confidence_filter(
    pose: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Landmark visibility < threshold → XYZ = NaN (giữ nguyên visibility)."""
    out = pose.copy().astype(np.float32)
    if out.shape[2] < 4:
        return out
    mask = out[:, :, 3] < threshold
    out[mask, 0] = np.nan
    out[mask, 1] = np.nan
    out[mask, 2] = np.nan
    return out


def interpolate_missing(pose: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính XYZ theo thời gian; visibility thiếu → 0."""
    out = pose.copy().astype(np.float32)
    T, J, C = out.shape
    t_idx = np.arange(T)

    for j in range(J):
        for c in range(min(3, C)):
            values = out[:, j, c]
            nans = np.isnan(values)
            if not nans.any():
                continue
            if np.all(nans):
                out[:, j, c] = 0.0
                continue
            out[:, j, c] = np.interp(t_idx, t_idx[~nans], values[~nans])

        if C >= 4:
            vis = out[:, j, 3]
            vis_nans = np.isnan(vis)
            if vis_nans.any():
                out[vis_nans, j, 3] = 0.0
    return out


def get_hip_center(pose: np.ndarray) -> np.ndarray:
    """Tâm hông (T, 3). Fallback 1 bên hông nếu bên kia NaN/0 bất thường."""
    left = pose[:, LEFT_HIP, :3].astype(np.float32)
    right = pose[:, RIGHT_HIP, :3].astype(np.float32)

    left_ok = np.isfinite(left).all(axis=1) & (np.linalg.norm(left, axis=1) > 1e-8)
    right_ok = np.isfinite(right).all(axis=1) & (np.linalg.norm(right, axis=1) > 1e-8)

    center = np.zeros((pose.shape[0], 3), dtype=np.float32)
    both = left_ok & right_ok
    center[both] = (left[both] + right[both]) * 0.5
    only_l = left_ok & ~right_ok
    only_r = right_ok & ~left_ok
    center[only_l] = left[only_l]
    center[only_r] = right[only_r]

    # Frame không có hông hợp lệ → giữ center frame trước (ffill), rồi bfill.
    missing = ~(both | only_l | only_r)
    if missing.any():
        last = np.zeros(3, dtype=np.float32)
        for t in range(pose.shape[0]):
            if missing[t]:
                center[t] = last
            else:
                last = center[t]
        # bfill đầu chuỗi nếu bắt đầu bằng missing
        first_valid = np.argmax(~missing) if (~missing).any() else 0
        if missing[0] and (~missing).any():
            center[:first_valid] = center[first_valid]
    return center


def center_to_hip(pose: np.ndarray) -> np.ndarray:
    """Đưa gốc toạ độ về tâm hông — bất biến tịnh tiến trong khung hình."""
    coords = pose[:, :, :3].astype(np.float32).copy()
    hip_center = get_hip_center(pose)
    coords = coords - hip_center[:, np.newaxis, :]
    if pose.shape[2] == 4:
        return np.concatenate([coords, pose[:, :, 3:4].astype(np.float32)], axis=2)
    return coords


def _mean_spine_length(pose: np.ndarray, eps: float = 1e-8) -> float:
    """Chiều dài xương sống trung bình: hip_center → shoulder_center (sau recenter)."""
    left_s = pose[:, LEFT_SHOULDER, :3]
    right_s = pose[:, RIGHT_SHOULDER, :3]
    shoulder_center = (left_s + right_s) * 0.5
    # Sau center_to_hip, hip ≈ 0 → spine ≈ ||shoulder_center||
    spine = np.linalg.norm(shoulder_center, axis=1)
    spine = spine[np.isfinite(spine) & (spine > eps)]
    if spine.size == 0:
        return 1.0
    return float(np.mean(spine))


def _mean_shoulder_width(pose: np.ndarray, eps: float = 1e-8) -> float:
    left_s = pose[:, LEFT_SHOULDER, :3]
    right_s = pose[:, RIGHT_SHOULDER, :3]
    width = np.linalg.norm(left_s - right_s, axis=1)
    width = width[np.isfinite(width) & (width > eps)]
    if width.size == 0:
        return 1.0
    return float(np.mean(width))


def scale_by_height(
    pose: np.ndarray,
    eps: float = 1e-8,
    method: str = "spine",
) -> np.ndarray:
    """
    Chuẩn hoá tỷ lệ cơ thể bằng **một** hệ số ổn định trên cả clip
    (tránh per-frame scale làm nhiễu diff_sequence).

    method:
      - "spine": mean ||shoulder_center - hip_center|| (khuyến nghị)
      - "shoulder_width": mean khoảng cách 2 vai
      - "mean_radius": mean L2 của toàn bộ khớp (fallback cũ)
    """
    coords = pose[:, :, :3].astype(np.float32).copy()

    if method == "spine":
        scale = _mean_spine_length(pose, eps=eps)
    elif method == "shoulder_width":
        scale = _mean_shoulder_width(pose, eps=eps)
    elif method == "mean_radius":
        radii = np.sqrt((coords ** 2).sum(axis=-1))
        scale = float(np.nanmean(radii)) + eps
    else:
        raise ValueError(f"Unknown scale method: {method}")

    scale = max(scale, eps)
    coords = coords / scale

    if pose.shape[2] == 4:
        return np.concatenate([coords, pose[:, :, 3:4].astype(np.float32)], axis=2)
    return coords


def smooth_sequence(
    pose: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky–Golay chỉ trên XYZ; không làm mượt visibility."""
    out = pose.copy().astype(np.float32)
    T = out.shape[0]
    if T < 3:
        return out

    wl = window_length
    if T < wl:
        wl = T if T % 2 == 1 else T - 1
    if wl % 2 == 0:
        wl -= 1
    if wl <= polyorder:
        return out

    # Chỉ smooth 3 kênh toạ độ
    for c in range(3):
        out[:, :, c] = savgol_filter(out[:, :, c], window_length=wl, polyorder=polyorder, axis=0)
    return out


def normalize_length(pose: np.ndarray, target_frames: int = 150) -> np.ndarray:
    """Resample tuyến tính độ dài sequence về `target_frames`."""
    T, J, D = pose.shape
    if T == target_frames:
        return pose.astype(np.float32)

    if T < 2:
        # Không đủ điểm để interp — pad/repeat
        tiled = np.repeat(pose[:1], target_frames, axis=0).astype(np.float32)
        return tiled

    t_orig = np.linspace(0.0, 1.0, T)
    t_target = np.linspace(0.0, 1.0, target_frames)
    resampled = np.zeros((target_frames, J, D), dtype=np.float32)

    for j in range(J):
        for d in range(D):
            # bounds_error=False + endpoint clamp thay vì extrapolate mạnh
            f = interp1d(
                t_orig,
                pose[:, j, d],
                kind="linear",
                bounds_error=False,
                fill_value=(pose[0, j, d], pose[-1, j, d]),
            )
            resampled[:, j, d] = f(t_target)
    return resampled


def preprocess_pipeline(
    pose: np.ndarray,
    target_frames: int = -1,
    *,
    visibility_threshold: float = 0.3,
    scale_method: str = "spine",
    smooth_window: int = 7,
    apply_confidence_filter: bool = True,
) -> np.ndarray:
    """
    Pipeline chuẩn (tham chiếu MAS + Demo):
      confidence filter → interpolate → center hip → mean-scale → Savitzky–Golay
      → (optional) resample length

    Không thay thế DTW: resample chỉ để batching / khi chưa có alignment_path.
    """
    if pose.ndim != 3 or pose.shape[1] != 33 or pose.shape[2] not in (3, 4):
        raise ValueError(f"Expected pose (T, 33, 3|4), got {pose.shape}")

    seq = pose.astype(np.float32)
    if apply_confidence_filter and seq.shape[2] == 4:
        seq = confidence_filter(seq, threshold=visibility_threshold)
        seq = interpolate_missing(seq)

    seq = center_to_hip(seq)
    seq = scale_by_height(seq, method=scale_method)
    seq = smooth_sequence(seq, window_length=smooth_window)

    if target_frames is not None and target_frames > 0:
        seq = normalize_length(seq, target_frames)
    return seq


def _parse_alignment_path(alignment_path: AlignmentPath) -> Tuple[np.ndarray, np.ndarray]:
    """Nhận path dạng [(i_p, i_r), ...] hoặc array (L, 2) → hai mảng index."""
    path = np.asarray(alignment_path, dtype=np.int64)
    if path.ndim != 2 or path.shape[1] < 2:
        raise ValueError(
            f"alignment_path phải có shape (L, 2), nhận được {path.shape}"
        )
    idx_p = path[:, 0]
    idx_r = path[:, 1]
    if (idx_p < 0).any() or (idx_r < 0).any():
        raise ValueError("alignment_path chứa index âm.")
    return idx_p, idx_r


def align_and_compute_diff(
    performer_pose: np.ndarray,
    reference_pose: np.ndarray,
    alignment_path: Optional[AlignmentPath] = None,
) -> np.ndarray:
    """
    Khớp thời gian rồi tính diff_sequence = performer − reference (XYZ).

    - Có `alignment_path` từ Person 2 (DTW): warp theo cặp index.
    - Không có path: hai chuỗi phải cùng số frame (đã resample hoặc đã align trước).

    Returns:
        diff_sequence shape (T_aligned, 33, 3)
    """
    p_coords = np.asarray(performer_pose[:, :, :3], dtype=np.float32)
    r_coords = np.asarray(reference_pose[:, :, :3], dtype=np.float32)

    if alignment_path is None:
        if p_coords.shape[0] != r_coords.shape[0]:
            raise ValueError(
                "Không có alignment_path (DTW): 2 chuỗi phải cùng số frame "
                f"(got T_p={p_coords.shape[0]}, T_r={r_coords.shape[0]}). "
                "Gọi preprocess_pipeline(..., target_frames=N) cho cả hai, "
                "hoặc truyền DTW path từ Person 2."
            )
        aligned_p, aligned_r = p_coords, r_coords
    else:
        idx_p, idx_r = _parse_alignment_path(alignment_path)
        if idx_p.max() >= p_coords.shape[0] or idx_r.max() >= r_coords.shape[0]:
            raise ValueError(
                f"alignment_path vượt quá độ dài chuỗi "
                f"(P={p_coords.shape[0]}, R={r_coords.shape[0]}, "
                f"max_p={idx_p.max()}, max_r={idx_r.max()})."
            )
        aligned_p = p_coords[idx_p]
        aligned_r = r_coords[idx_r]

    return aligned_p - aligned_r


def compute_diff(
    performer_pose: np.ndarray,
    reference_pose: np.ndarray,
    alignment_path: Optional[AlignmentPath] = None,
) -> np.ndarray:
    """Alias công khai theo docs — cùng `align_and_compute_diff`."""
    return align_and_compute_diff(performer_pose, reference_pose, alignment_path)


def build_diff_sequence(
    performer_pose: np.ndarray,
    reference_pose: np.ndarray,
    alignment_path: Optional[AlignmentPath] = None,
    *,
    target_frames: int = 150,
    scale_method: str = "spine",
    smooth_window: int = 7,
    visibility_threshold: float = 0.3,
    auto_dtw: bool = True,
    return_meta: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    End-to-end Person 1 week 3:
      preprocess(P), preprocess(R) → DTW-align → diff → resample cố định.

    - Có `alignment_path` từ Person 2: dùng path đó.
    - `auto_dtw=True` (mặc định): tự gọi `features.dtw.dtw_align` nếu chưa có path.
    - `auto_dtw=False` và không path: fallback resample cùng T rồi trừ.
    """
    p_norm = preprocess_pipeline(
        performer_pose,
        target_frames=-1,
        visibility_threshold=visibility_threshold,
        scale_method=scale_method,
        smooth_window=smooth_window,
    )
    r_norm = preprocess_pipeline(
        reference_pose,
        target_frames=-1,
        visibility_threshold=visibility_threshold,
        scale_method=scale_method,
        smooth_window=smooth_window,
    )

    used_dtw = False
    dtw_distance = None
    path = alignment_path

    if path is None and auto_dtw:
        from src.features.dtw import dtw_align

        path, dtw_distance = dtw_align(p_norm[:, :, :3], r_norm[:, :, :3])
        used_dtw = True
    elif path is not None:
        used_dtw = True

    if path is not None:
        diff = compute_diff(p_norm, r_norm, alignment_path=path)
    else:
        # Fallback: resample về cùng độ dài rồi trừ
        if target_frames > 0:
            p_rs = normalize_length(p_norm, target_frames)
            r_rs = normalize_length(r_norm, target_frames)
        else:
            T = min(p_norm.shape[0], r_norm.shape[0])
            p_rs = normalize_length(p_norm, T)
            r_rs = normalize_length(r_norm, T)
        diff = compute_diff(p_rs, r_rs, alignment_path=None)

    if target_frames > 0 and diff.shape[0] != target_frames:
        diff = normalize_length(diff, target_frames)

    diff = diff.astype(np.float32)
    if not return_meta:
        return diff
    meta = {
        "used_dtw": used_dtw,
        "dtw_distance": dtw_distance,
        "alignment_path": path,
        "mean_abs_diff": mean_abs_diff(diff),
    }
    return diff, meta


def mean_abs_diff(diff_sequence: np.ndarray) -> float:
    """Sanity metric: performer khớp tốt → nhỏ; lệch nhiều → lớn."""
    return float(np.mean(np.abs(diff_sequence)))
