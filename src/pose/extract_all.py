"""Person 1 — batch extract pose cho mọi video reference + performer."""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

from src.pose.pose_extractor import extract_pose

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_output_path(video_path: str, data_dir: str, output_dir: str) -> str:
    """
    Map video → poses/{dance_id}/{video_id}.npy theo docs/data_format.md.

    Ví dụ:
      data/dances/dance_001/reference/ref.mp4  → poses/dance_001/ref.npy
      data/dances/dance_001/performers/p01.mp4 → poses/dance_001/p01.npy
    """
    rel = Path(os.path.relpath(video_path, data_dir))
    parts = rel.parts
    # Kỳ vọng: dance_xxx / (reference|performers) / file.ext  hoặc dance_xxx / file.ext
    dance_id = parts[0] if parts else "unknown"
    video_id = Path(parts[-1]).stem
    dance_dir = os.path.join(output_dir, dance_id)
    os.makedirs(dance_dir, exist_ok=True)
    return os.path.join(dance_dir, f"{video_id}.npy")


def process_all_videos(
    data_dir: str,
    output_dir: str,
    *,
    skip_existing: bool = True,
    model_complexity: int = 2,
) -> None:
    """Quét video trong data_dir và lưu pose .npy theo cấu trúc dance_id/video_id."""
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    if not video_paths:
        logger.warning("Không tìm thấy video nào trong %s", data_dir)
        return

    logger.info("Tìm thấy %d videos. Bắt đầu trích xuất...", len(video_paths))
    os.makedirs(output_dir, exist_ok=True)

    ok, skipped, failed = 0, 0, 0
    for video_path in sorted(video_paths):
        try:
            output_path = _resolve_output_path(video_path, data_dir, output_dir)
            if skip_existing and os.path.exists(output_path):
                logger.info("Bỏ qua (đã có): %s", output_path)
                skipped += 1
                continue

            poses = extract_pose(video_path, model_complexity=model_complexity)
            np.save(output_path, poses)
            logger.info("Lưu %s → %s %s", video_path, output_path, poses.shape)
            ok += 1
        except Exception as e:
            failed += 1
            logger.error("Lỗi khi xử lý %s: %s", video_path, e)

    logger.info("Xong. ok=%d skipped=%d failed=%d", ok, skipped, failed)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data", "dances")
    OUTPUT_DIR = os.path.join(BASE_DIR, "poses")
    process_all_videos(DATA_DIR, OUTPUT_DIR)
    logger.info("Hoàn tất batch processing!")
