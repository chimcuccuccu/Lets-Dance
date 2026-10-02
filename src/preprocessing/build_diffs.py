"""Person 1 — batch build diff_sequence từ poses/ + DTW align (auto nếu chưa có path)."""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing.preprocess import build_diff_sequence, mean_abs_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _find_pose(poses_dir: str, dance_id: str, video_id: str) -> Optional[str]:
    candidates = [
        os.path.join(poses_dir, dance_id, f"{video_id}.npy"),
        os.path.join(poses_dir, f"{dance_id}_{video_id}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _find_ref(poses_dir: str, dance_id: str, ref_id: str = "") -> Optional[str]:
    if ref_id:
        found = _find_pose(poses_dir, dance_id, ref_id)
        if found:
            return found
    dance_dir = os.path.join(poses_dir, dance_id)
    if not os.path.isdir(dance_dir):
        return None
    for name in sorted(os.listdir(dance_dir)):
        if name.endswith(".npy") and "ref" in name.lower() and not name.endswith("_diff.npy"):
            return os.path.join(dance_dir, name)
    return None


def _find_alignment(alignment_dir: str, dance_id: str, video_id: str) -> Optional[np.ndarray]:
    if not alignment_dir:
        return None
    candidates = [
        os.path.join(alignment_dir, f"{dance_id}_{video_id}_path.npy"),
        os.path.join(alignment_dir, dance_id, f"{video_id}_path.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return np.load(p)
    return None


def build_all_diffs(
    scores_csv: str,
    poses_dir: str,
    output_dir: str,
    alignment_dir: str = "annotations/dtw_paths",
    target_frames: int = 150,
    auto_dtw: bool = True,
    skip_existing: bool = True,
    save_paths: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    os.makedirs(output_dir, exist_ok=True)
    if save_paths and alignment_dir:
        os.makedirs(alignment_dir, exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        dance_id = str(row["dance_id"])
        video_id = str(row["video_id"])
        ref_id = str(row["ref_id"]) if "ref_id" in row.index and pd.notna(row["ref_id"]) else ""

        dance_out = os.path.join(output_dir, dance_id)
        os.makedirs(dance_out, exist_ok=True)
        nested = os.path.join(dance_out, f"{video_id}_diff.npy")
        flat = os.path.join(output_dir, f"{dance_id}_{video_id}_diff.npy")

        if skip_existing and os.path.exists(nested) and os.path.exists(flat):
            mad = mean_abs_diff(np.load(nested))
            rows.append(
                {
                    "dance_id": dance_id,
                    "video_id": video_id,
                    "diff_path": nested,
                    "mean_abs_diff": mad,
                    "used_dtw": True,
                    "skipped": True,
                }
            )
            continue

        p_path = _find_pose(poses_dir, dance_id, video_id)
        r_path = _find_ref(poses_dir, dance_id, ref_id)
        if not p_path or not r_path:
            logger.warning("Bỏ qua %s/%s — thiếu pose P=%s R=%s", dance_id, video_id, p_path, r_path)
            continue

        performer = np.load(p_path)
        reference = np.load(r_path)
        alignment = _find_alignment(alignment_dir, dance_id, video_id)

        diff, meta = build_diff_sequence(
            performer,
            reference,
            alignment_path=alignment,
            target_frames=target_frames,
            auto_dtw=auto_dtw and alignment is None,
            return_meta=True,
        )

        np.save(nested, diff)
        np.save(flat, diff)

        if save_paths and alignment_dir and meta.get("alignment_path") is not None:
            path_dir = os.path.join(alignment_dir, dance_id)
            os.makedirs(path_dir, exist_ok=True)
            path_arr = np.asarray(meta["alignment_path"], dtype=np.int64)
            np.save(os.path.join(path_dir, f"{video_id}_path.npy"), path_arr)
            np.save(os.path.join(alignment_dir, f"{dance_id}_{video_id}_path.npy"), path_arr)

        mad = float(meta["mean_abs_diff"])
        logger.info(
            "[%d/%d] %s/%s diff=%s MAD=%.5f dtw=%s",
            len(rows) + 1,
            len(df),
            dance_id,
            video_id,
            diff.shape,
            mad,
            meta["used_dtw"],
        )
        rows.append(
            {
                "dance_id": dance_id,
                "video_id": video_id,
                "diff_path": nested,
                "mean_abs_diff": mad,
                "used_dtw": bool(meta["used_dtw"]),
                "dtw_distance": meta.get("dtw_distance"),
                "skipped": False,
            }
        )

    summary = pd.DataFrame(rows)
    if len(summary):
        out_csv = os.path.join(output_dir, "diff_summary.csv")
        summary.to_csv(out_csv, index=False)
        # Correlation với scores nếu có
        if "tong_diem" in df.columns:
            merged = summary.merge(df[["dance_id", "video_id", "tong_diem"]], on=["dance_id", "video_id"], how="inner")
            if len(merged) >= 3:
                r = float(np.corrcoef(merged["mean_abs_diff"], merged["tong_diem"])[0, 1])
                logger.info("Pearson(MAD, tong_diem)=%.3f on %d samples", r, len(merged))
        logger.info("Wrote %s (%d diffs)", out_csv, len(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build diff_sequence for Spatial DL")
    parser.add_argument("--scores", default="annotations/scores.csv")
    parser.add_argument("--poses", default="poses")
    parser.add_argument("--out", default="poses/diffs")
    parser.add_argument("--alignment-dir", default="annotations/dtw_paths")
    parser.add_argument("--target-frames", type=int, default=150)
    parser.add_argument("--no-dtw", action="store_true", help="Tắt auto DTW (chỉ resample)")
    parser.add_argument("--force", action="store_true", help="Build lại dù đã có file")
    args = parser.parse_args()
    build_all_diffs(
        args.scores,
        args.poses,
        args.out,
        alignment_dir=args.alignment_dir,
        target_frames=args.target_frames,
        auto_dtw=not args.no_dtw,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
