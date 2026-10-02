"""Smoke test Person 1: pose → preprocess → diff → (optional) SpatialModelV3.

Mặc định KHÔNG import torch (tránh treo lâu trên Windows).
Dùng --model nếu muốn test forward Spatial DL.

Ưu tiên load .npy có sẵn trong poses/ nếu truyền đường dẫn pose,
hoặc suy ra từ video path để khỏi chạy MediaPipe lại.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocess import (
    build_diff_sequence,
    get_hip_center,
    mean_abs_diff,
    preprocess_pipeline,
)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _guess_pose_npy(video_or_pose: str) -> Optional[str]:
    """Nếu là .npy → dùng luôn; nếu là video → tìm poses/{dance}/{stem}.npy."""
    if video_or_pose.lower().endswith(".npy") and os.path.exists(video_or_pose):
        return video_or_pose

    if not os.path.exists(video_or_pose):
        return None

    # data/dances/dance_001/performers/D01_P003_T01.mp4 → poses/dance_001/D01_P003_T01.npy
    abs_path = os.path.abspath(video_or_pose)
    parts = abs_path.replace("\\", "/").split("/")
    try:
        i = parts.index("dances")
        dance_id = parts[i + 1]
        stem = os.path.splitext(parts[-1])[0]
        candidate = os.path.join(_repo_root(), "poses", dance_id, f"{stem}.npy")
        if os.path.exists(candidate):
            return candidate
    except (ValueError, IndexError):
        pass
    return None


def _load_pose(path: str) -> Tuple[np.ndarray, str]:
    npy = _guess_pose_npy(path)
    if npy:
        print(f"   Load pose có sẵn: {npy}")
        return np.load(npy), npy

    from src.pose.pose_extractor import extract_pose

    print(f"   Extract MediaPipe từ video: {path}")
    return extract_pose(path), path


def test_pipeline(
    video_or_pose: str,
    reference_path: Optional[str] = None,
    *,
    run_model: bool = False,
) -> None:
    print(f"--- BẮT ĐẦU TEST: {video_or_pose} ---")

    print("1. Load / extract pose...")
    raw_poses, src = _load_pose(video_or_pose)
    print(f"   Raw pose: {raw_poses.shape} | vis_mean={raw_poses[:, :, 3].mean():.3f}")
    print(f"   Hip frame0 (trước preprocess): {get_hip_center(raw_poses)[0]}")

    print("\n2. Preprocess (filter → interp → center → spine-scale → smooth)...")
    processed = preprocess_pipeline(raw_poses, target_frames=150)
    print(f"   Processed: {processed.shape}")
    hip_after = get_hip_center(processed)
    print(f"   Hip frame0 (sau preprocess): {hip_after[0]}  (kỳ vọng ≈ 0)")
    assert np.allclose(hip_after, 0.0, atol=1e-4), "Hip center chưa về gần 0!"

    if reference_path and os.path.exists(reference_path):
        print(f"\n3. Build diff vs reference: {reference_path}")
        ref_raw, _ = _load_pose(reference_path)
        diff = build_diff_sequence(raw_poses, ref_raw, alignment_path=None, target_frames=150)
        self_diff = build_diff_sequence(raw_poses, raw_poses, target_frames=150)
        print(f"   mean|self-diff|={mean_abs_diff(self_diff):.6f}")
        print(f"   mean|vs-ref|={mean_abs_diff(diff):.6f}")
    else:
        print("\n3. Build diff vs bản lệch hình dạng (sanity)...")
        warped = raw_poses.copy()
        warped[:, 15, :3] += 0.08
        warped[:, 13, :3] += 0.05
        diff = build_diff_sequence(raw_poses, warped, target_frames=150)
        self_diff = build_diff_sequence(raw_poses, raw_poses, target_frames=150)
        print(f"   mean|self-diff|={mean_abs_diff(self_diff):.6f} (kỳ vọng ≈ 0)")
        print(f"   mean|shape-diff|={mean_abs_diff(diff):.6f} (kỳ vọng > self-diff)")

    print(f"   diff_sequence: {diff.shape}")

    if run_model:
        print("\n4. SpatialModelV3 forward (import torch)...")
        import torch
        from src.spatial_dl.model_v3 import SpatialModelV3, spatial_model

        model = SpatialModelV3()
        x = torch.from_numpy(diff).unsqueeze(0)
        score, embed = model(x)
        emb_api = spatial_model(x.squeeze(0), model)
        print(
            f"   score={score.item():.2f} | embed={tuple(embed.shape)} | "
            f"L2={embed.norm().item():.4f} | API_L2={emb_api.norm().item():.4f}"
        )
    else:
        print("\n4. Bỏ qua SpatialModel (thêm --model nếu cần).")

    np.save("test_output.npy", processed)
    np.save("test_diff.npy", diff)
    print(f"\n5. Đã lưu test_output.npy + test_diff.npy (từ {src})")
    print("--- HOÀN TẤT ---")


def main():
    p = argparse.ArgumentParser(description="Smoke test Person 1 pipeline")
    p.add_argument("input", nargs="?", help="video.mp4 hoặc pose.npy")
    p.add_argument("reference", nargs="?", help="reference video/pose (optional)")
    p.add_argument("--model", action="store_true", help="Test thêm SpatialModelV3 (cần torch)")
    args = p.parse_args()

    if not args.input:
        default = os.path.join(_repo_root(), "poses", "dance_001", "D01_P003_T01.npy")
        ref = os.path.join(_repo_root(), "poses", "dance_001", "dance_001_ref.npy")
        if os.path.exists(default):
            print(f"Không có arg → dùng mặc định: {default}")
            test_pipeline(default, ref if os.path.exists(ref) else None, run_model=args.model)
        else:
            p.print_help()
        return

    test_pipeline(args.input, args.reference, run_model=args.model)


if __name__ == "__main__":
    main()
