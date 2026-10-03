"""
Person 2 — Tuần 3: Chạy DTW cho toàn bộ dataset và lưu annotations/dtw_features.csv.

Quy ước file thực tế (từ Person 1):
  poses/{dance_id}/{dance_id}_ref.npy          — reference
  poses/{dance_id}/D{nn}_P{nnn}_T01.npy        — performer
  annotations/scores.csv có cột ref_id = "dance_xxx_ref" để tra cứu.

Output: annotations/dtw_features.csv
  cột: dance_id, video_id, dtw_distance_total, dtw_seg_0, dtw_seg_1, ..., flag_seg_0, flag_seg_1, ...
"""
from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Thêm project root vào sys.path khi chạy trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.geometry import compute_geometry_features
from src.features.dtw import dtw_distance_total, dtw_distance_windowed, flag_suspicious_segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _find_ref_pose(dance_dir: Path, dance_id: str) -> Path | None:
    """Tìm file pose reference của một bài nhảy.

    Ưu tiên: {dance_id}_ref.npy (quy ước Person 1 thực tế).
    Fallback: bất kỳ file nào có 'ref' trong tên (D01_REF01.npy...).
    """
    candidate = dance_dir / f"{dance_id}_ref.npy"
    if candidate.exists():
        return candidate
    # Fallback — tìm tên chứa 'ref' không phân biệt hoa thường
    refs = [p for p in dance_dir.glob("*.npy") if "ref" in p.stem.lower()]
    if refs:
        return refs[0]
    return None


def _find_performer_poses(dance_dir: Path, ref_path: Path) -> list[Path]:
    """Liệt kê tất cả .npy trong dance_dir trừ file reference."""
    return [p for p in dance_dir.glob("*.npy") if p != ref_path]


def process_all(
    poses_dir: str | None = None,
    output_csv: str | None = None,
    window_frames: int = 60,   # ~2s ở 30fps
) -> None:
    poses_root = Path(poses_dir) if poses_dir else PROJECT_ROOT / "poses"
    out_path = Path(output_csv) if output_csv else PROJECT_ROOT / "annotations" / "dtw_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dance_dirs = sorted(poses_root.glob("dance_*"))
    if not dance_dirs:
        logger.warning("Không tìm thấy thư mục dance_* trong %s", poses_root)
        return

    all_results: list[dict] = []

    for d_dir in dance_dirs:
        dance_id = d_dir.name
        ref_path = _find_ref_pose(d_dir, dance_id)

        if ref_path is None:
            logger.warning("Bỏ qua %s — không tìm thấy file reference pose.", dance_id)
            continue

        logger.info("Đang xử lý bài: %s | reference: %s", dance_id, ref_path.name)

        try:
            ref_pose = np.load(ref_path)       # (T, 33, 4)
            ref_geom = compute_geometry_features(ref_pose)  # (T, k)
        except Exception as e:
            logger.error("Lỗi load/compute geometry reference %s: %s", ref_path, e)
            continue

        performer_files = _find_performer_poses(d_dir, ref_path)
        if not performer_files:
            logger.warning("%s — không tìm thấy performer nào.", dance_id)
            continue

        for perf_file in sorted(performer_files):
            video_id = perf_file.stem   # VD: D01_P001_T01
            try:
                perf_pose = np.load(perf_file)   # (T, 33, 4)
                perf_geom = compute_geometry_features(perf_pose)  # (T, k)

                total_dist = dtw_distance_total(perf_geom, ref_geom)
                win_dists = dtw_distance_windowed(perf_geom, ref_geom, window=window_frames)
                flags = flag_suspicious_segments(win_dists)

                row: dict = {
                    "dance_id": dance_id,
                    "video_id": video_id,
                    "dtw_distance_total": round(float(total_dist), 6),
                }
                for i, d_val in enumerate(win_dists):
                    row[f"dtw_seg_{i}"] = round(float(d_val), 6)
                for i, f_val in enumerate(flags):
                    row[f"flag_seg_{i}"] = int(f_val)

                all_results.append(row)
                n_flag = int(flags.sum())
                logger.info("  %s/%s  total=%.4f  segs=%d  flagged=%d",
                            dance_id, video_id, total_dist, len(win_dists), n_flag)

            except Exception as e:
                logger.error("Lỗi khi xử lý %s/%s: %s", dance_id, video_id, e)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_path, index=False)
        logger.info("✅ Đã lưu %d dòng → %s", len(df), out_path)
    else:
        logger.warning("Không có kết quả nào để lưu.")


if __name__ == "__main__":
    process_all()
