"""
Convert skeleton formats → MediaPipe Pose (33 joints).

AIST++ ships 3D keypoints as COCO-17 (from their SMPL-fitting pipeline).
Optional path: SMPL-24 joint *positions* (not axis-angle pose params).

Unmapped MediaPipe joints are filled by anatomical heuristics and marked
visibility=0 so ``preprocess_pipeline`` can interpolate / down-weight them.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Joint name tables
# ---------------------------------------------------------------------------

COCO17_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Standard SMPL body joints (24) — positions after forward kinematics.
SMPL24_NAMES: Tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

MEDIAPIPE33_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Direct index maps: source_idx → mediapipe_idx
COCO17_TO_MP: Dict[int, int] = {
    0: 0,   # nose
    1: 2,   # left_eye
    2: 5,   # right_eye
    3: 7,   # left_ear
    4: 8,   # right_ear
    5: 11,  # left_shoulder
    6: 12,  # right_shoulder
    7: 13,  # left_elbow
    8: 14,  # right_elbow
    9: 15,  # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}

SMPL24_TO_MP: Dict[int, int] = {
    1: 23,   # left_hip
    2: 24,   # right_hip
    4: 25,   # left_knee
    5: 26,   # right_knee
    7: 27,   # left_ankle
    8: 28,   # right_ankle
    10: 31,  # left_foot → foot_index
    11: 32,  # right_foot → foot_index
    15: 0,   # head ≈ nose (best available)
    16: 11,  # left_shoulder
    17: 12,  # right_shoulder
    18: 13,  # left_elbow
    19: 14,  # right_elbow
    20: 15,  # left_wrist
    21: 16,  # right_wrist
    22: 19,  # left_hand → index
    23: 20,  # right_hand → index
}


def _as_float32_xyz(joints: np.ndarray, expected_j: int) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != expected_j or arr.shape[2] < 3:
        raise ValueError(
            f"Expected (T, {expected_j}, 3[+]), got {arr.shape}"
        )
    return arr[:, :, :3].copy()


def _apply_direct_map(
    src_xyz: np.ndarray,
    mapping: Mapping[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (T,33,3) coords and (T,33) visibility in {0,1}."""
    T = src_xyz.shape[0]
    xyz = np.zeros((T, 33, 3), dtype=np.float32)
    vis = np.zeros((T, 33), dtype=np.float32)
    for src_i, mp_i in mapping.items():
        xyz[:, mp_i] = src_xyz[:, src_i]
        vis[:, mp_i] = 1.0
    return xyz, vis


def _fill_mediapipe_heuristics(xyz: np.ndarray, vis: np.ndarray) -> None:
    """In-place fill for joints without a direct source correspondence."""
    # Eyes: inner/outer ≈ eye when missing
    for eye, inner, outer in ((2, 1, 3), (5, 4, 6)):
        if vis[:, eye].any():
            for j in (inner, outer):
                miss = vis[:, j] < 0.5
                xyz[miss, j] = xyz[miss, eye]
                # Keep vis=0 — interpolated / estimated

    # Mouth corners ≈ midpoint nose–shoulder side
    if vis[:, 0].any() and vis[:, 11].any():
        miss = vis[:, 9] < 0.5
        xyz[miss, 9] = 0.6 * xyz[miss, 0] + 0.4 * xyz[miss, 11]
    if vis[:, 0].any() and vis[:, 12].any():
        miss = vis[:, 10] < 0.5
        xyz[miss, 10] = 0.6 * xyz[miss, 0] + 0.4 * xyz[miss, 12]

    # Hand tips ≈ wrist (no finger detail in COCO/SMPL-24)
    for wrist, tips in ((15, (17, 19, 21)), (16, (18, 20, 22))):
        if not vis[:, wrist].any():
            continue
        for tip in tips:
            miss = vis[:, tip] < 0.5
            xyz[miss, tip] = xyz[miss, wrist]

    # Heel / foot_index from ankle (+ optional foot if already mapped)
    for ankle, heel, foot in ((27, 29, 31), (28, 30, 32)):
        if not vis[:, ankle].any():
            continue
        # Heel slightly behind ankle along hip→ankle direction
        hip = ankle - 4  # 23/24
        if vis[:, hip].any():
            leg = xyz[:, ankle] - xyz[:, hip]
            norm = np.linalg.norm(leg, axis=1, keepdims=True) + 1e-8
            unit = leg / norm
            miss_h = vis[:, heel] < 0.5
            xyz[miss_h, heel] = xyz[miss_h, ankle] - 0.15 * unit[miss_h]
        else:
            miss_h = vis[:, heel] < 0.5
            xyz[miss_h, heel] = xyz[miss_h, ankle]

        miss_f = vis[:, foot] < 0.5
        if miss_f.any():
            # Foot index slightly forward of ankle (opposite heel offset)
            xyz[miss_f, foot] = xyz[miss_f, ankle] + (xyz[miss_f, ankle] - xyz[miss_f, heel]) * 0.5


def _pack_pose(xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    return np.concatenate([xyz, vis[:, :, None]], axis=2).astype(np.float32)


def coco17_to_mediapipe(joints: np.ndarray) -> np.ndarray:
    """
    COCO-17 (T, 17, 3) → MediaPipe (T, 33, 4).

    Direct map for 17 shared joints; remaining 16 filled by heuristics with vis=0.
    """
    src = _as_float32_xyz(joints, expected_j=17)
    xyz, vis = _apply_direct_map(src, COCO17_TO_MP)
    _fill_mediapipe_heuristics(xyz, vis)
    return _pack_pose(xyz, vis)


def smpl24_to_mediapipe(joints: np.ndarray) -> np.ndarray:
    """
    SMPL-24 joint *positions* (T, 24, 3) → MediaPipe (T, 33, 4).

    Does **not** accept SMPL axis-angle pose parameters — run SMPL FK first.
    Nose/eyes/ears are approximated from head/shoulders (vis=0 for estimates).
    """
    src = _as_float32_xyz(joints, expected_j=24)
    xyz, vis = _apply_direct_map(src, SMPL24_TO_MP)

    # Nose from head if mapped (index 15 → 0 already). Approximate eyes/ears.
    if vis[:, 0].any() and vis[:, 11].any() and vis[:, 12].any():
        shoulder_mid = 0.5 * (xyz[:, 11] + xyz[:, 12])
        # Ears near shoulder–head line
        miss_le = vis[:, 7] < 0.5
        miss_re = vis[:, 8] < 0.5
        xyz[miss_le, 7] = 0.5 * (xyz[miss_le, 0] + xyz[miss_le, 11])
        xyz[miss_re, 8] = 0.5 * (xyz[miss_re, 0] + xyz[miss_re, 12])
        # Eyes between nose and ears
        xyz[:, 2] = 0.7 * xyz[:, 0] + 0.3 * xyz[:, 7]
        xyz[:, 5] = 0.7 * xyz[:, 0] + 0.3 * xyz[:, 8]
        _ = shoulder_mid  # reserved for future face offsets

    _fill_mediapipe_heuristics(xyz, vis)
    return _pack_pose(xyz, vis)


def convert_to_mediapipe(joints: np.ndarray, fmt: str = "auto") -> np.ndarray:
    """
    Dispatch by ``fmt`` or by joint count when ``fmt='auto'``.

    fmt: 'coco17' | 'smpl24' | 'mediapipe' | 'auto'
    """
    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected (T, J, C), got {arr.shape}")

    kind = fmt.lower()
    if kind == "auto":
        j = arr.shape[1]
        if j == 33:
            kind = "mediapipe"
        elif j == 17:
            kind = "coco17"
        elif j == 24:
            kind = "smpl24"
        else:
            raise ValueError(
                f"Cannot auto-detect format for J={j}; pass fmt explicitly."
            )

    if kind == "mediapipe":
        if arr.shape[1] != 33 or arr.shape[2] not in (3, 4):
            raise ValueError(f"MediaPipe expects (T,33,3|4), got {arr.shape}")
        if arr.shape[2] == 4:
            return arr.astype(np.float32)
        vis = np.ones((arr.shape[0], 33, 1), dtype=np.float32)
        return np.concatenate([arr[:, :, :3], vis], axis=2)

    if kind == "coco17":
        return coco17_to_mediapipe(arr)
    if kind == "smpl24":
        return smpl24_to_mediapipe(arr)
    raise ValueError(f"Unknown fmt={fmt!r}")


def mapped_joint_mask(fmt: str = "coco17") -> np.ndarray:
    """Boolean mask (33,) — True for joints with a direct source mapping."""
    mapping = COCO17_TO_MP if fmt == "coco17" else SMPL24_TO_MP
    mask = np.zeros(33, dtype=bool)
    for mp_i in mapping.values():
        mask[mp_i] = True
    return mask


__all__ = [
    "COCO17_NAMES",
    "SMPL24_NAMES",
    "MEDIAPIPE33_NAMES",
    "COCO17_TO_MP",
    "SMPL24_TO_MP",
    "coco17_to_mediapipe",
    "smpl24_to_mediapipe",
    "convert_to_mediapipe",
    "mapped_joint_mask",
]
