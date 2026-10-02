"""Person 1 — extract_pose(video_path) -> np.ndarray (T, 33, 4)."""
from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ngưỡng visibility (MAS dùng 0.3). 0.5 quá cao với chân/mắt cá trên nhiều video dance
# (crop nửa người, góc nghiêng) → mask cả clip rồi nội suy thành 0 = hỏng pose.
DEFAULT_VISIBILITY_THRESHOLD = 0.3
# Chỉ cảnh báo gap *gián đoạn* (không phải khớp low-vis suốt cả video).
MAX_GAP_WARN = 30
# Nếu > tỷ lệ này bị mask → giữ toạ độ gốc MediaPipe, không zero-fill.
CHRONIC_LOW_VIS_RATIO = 0.85
CRITICAL_JOINTS = (11, 12, 23, 24, 25, 26, 27, 28)


def _interpolate_xyz(pose: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính XYZ theo thời gian. Visibility thiếu → 0."""
    out = pose.copy().astype(np.float32)
    T, J, _ = out.shape
    t_idx = np.arange(T)

    for j in range(J):
        for c in range(3):
            values = out[:, j, c]
            nans = np.isnan(values)
            if not nans.any():
                continue
            if np.all(nans):
                # Không bịa 0: để NaN, caller sẽ restore từ raw nếu cần.
                continue
            out[:, j, c] = np.interp(t_idx, t_idx[~nans], values[~nans])

        vis = out[:, j, 3]
        vis_nans = np.isnan(vis)
        if vis_nans.any():
            out[vis_nans, j, 3] = 0.0

    return out


def _apply_visibility_mask(
    pose: np.ndarray,
    visibility_threshold: float,
    raw_fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Landmark visibility thấp → NaN để nội suy.

    Nếu một khớp bị low-vis gần như cả clip (crop/camera), **giữ toạ độ gốc**
    MediaPipe thay vì mask hết rồi fill 0 (an toàn hơn cho dance nửa người).
    """
    out = pose.copy().astype(np.float32)
    if out.shape[2] < 4:
        return out

    T = out.shape[0]
    low = out[:, :, 3] < visibility_threshold
    valid_vis = ~np.isnan(out[:, :, 3])
    candidate = low & valid_vis

    fallback = raw_fallback if raw_fallback is not None else pose
    chronic = []
    for j in range(out.shape[1]):
        n_low = int(candidate[:, j].sum())
        # Frame người mất hẳn (NaN sẵn) không tính vào chronic ratio của joint
        n_detected = int(valid_vis[:, j].sum())
        if n_detected == 0:
            continue
        ratio = n_low / n_detected
        if ratio >= CHRONIC_LOW_VIS_RATIO:
            # Giữ raw XYZ cho khớp này trên các frame đã detect
            chronic.append(j)
            keep = candidate[:, j]
            out[keep, j, 0] = fallback[keep, j, 0]
            out[keep, j, 1] = fallback[keep, j, 1]
            out[keep, j, 2] = fallback[keep, j, 2]
        else:
            mask = candidate[:, j]
            out[mask, j, 0] = np.nan
            out[mask, j, 1] = np.nan
            out[mask, j, 2] = np.nan

    if chronic:
        logger.info(
            "Khớp low-vis gần cả clip %s — giữ toạ độ MediaPipe gốc (không zero-fill).",
            chronic,
        )
    return out


def _warn_long_gaps(pose_before_interp: np.ndarray, max_gap: int = MAX_GAP_WARN) -> None:
    """Cảnh báo gap gián đoạn trên khớp quan trọng (bỏ qua missing 100% cả clip)."""
    T = pose_before_interp.shape[0]
    for j in CRITICAL_JOINTS:
        missing = np.isnan(pose_before_interp[:, j, 0])
        miss_ratio = missing.mean()
        if not missing.any() or miss_ratio >= CHRONIC_LOW_VIS_RATIO:
            continue

        longest = 0
        run = 0
        for m in missing:
            if m:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        if longest >= max_gap:
            logger.warning(
                "Khớp %d mất liên tục %d/%d frames (%.0f%%) — nội suy có thể kém tin cậy.",
                j,
                longest,
                T,
                100.0 * miss_ratio,
            )


def extract_pose(
    video_path: str,
    *,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    smooth_landmarks: bool = True,
) -> np.ndarray:
    """
    Trích xuất pose từ video bằng MediaPipe Pose.

    Returns:
        np.ndarray shape (T, 33, 4) — (x, y, z, visibility).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose.close()
        raise IOError(f"Không thể mở video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    logger.info("Bắt đầu xử lý %s (%s frames, %.1f fps)", video_path, frame_count, fps)

    frames_landmarks = []
    frame_idx = 0
    missing_person_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            # MediaPipe protobuf deprecation noise
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*SymbolDatabase.GetPrototype.*",
                    category=UserWarning,
                )
                results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = np.zeros((33, 4), dtype=np.float32)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
                frames_landmarks.append(landmarks)
            else:
                frames_landmarks.append(np.full((33, 4), np.nan, dtype=np.float32))
                missing_person_frames += 1
                logger.debug("Frame %d: không tìm thấy pose.", frame_idx)

            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    if len(frames_landmarks) == 0:
        raise ValueError(f"Video {video_path} không có frame nào hợp lệ.")

    pose_array = np.asarray(frames_landmarks, dtype=np.float32)

    if missing_person_frames == len(frames_landmarks):
        raise ValueError(
            f"Video {video_path}: MediaPipe không phát hiện người ở bất kỳ frame nào."
        )

    if missing_person_frames > 0:
        logger.info(
            "Thiếu người ở %d/%d frames — sẽ nội suy sau khi mask visibility.",
            missing_person_frames,
            len(frames_landmarks),
        )

    masked = _apply_visibility_mask(
        pose_array, visibility_threshold, raw_fallback=pose_array
    )
    _warn_long_gaps(masked)
    processed = _interpolate_xyz(masked)

    # Còn NaN (toàn-joint missing): fill từ raw nếu có, không thì 0
    still_nan = np.isnan(processed[:, :, :3])
    if still_nan.any():
        raw_ok = ~np.isnan(pose_array[:, :, :3])
        take_raw = still_nan & raw_ok
        processed[:, :, :3][take_raw] = pose_array[:, :, :3][take_raw]
        left = np.isnan(processed[:, :, :3]).sum()
        if left:
            logger.warning("Còn %d XYZ NaN sau restore — gán 0.", left)
            processed[:, :, :3] = np.nan_to_num(processed[:, :, :3], nan=0.0)

    logger.info("Hoàn thành trích xuất pose. Shape: %s", processed.shape)
    return processed


if __name__ == "__main__":
    import glob
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    # .env nằm ở root repo (Lets-Dance/), không phụ thuộc cwd
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    video_path: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None
    if video_path is None:
        fallback_dir = os.getenv("FALLBACK_VIDEO_DIR", "").strip()
        if fallback_dir and os.path.exists(fallback_dir):
            videos = glob.glob(os.path.join(fallback_dir, "*.mp4"))
            if videos:
                video_path = videos[0]
                print(f"Không có tham số đầu vào. Tự động dùng video mẫu: {video_path}")


    if video_path and os.path.exists(video_path):
        print(f"Testing video: {video_path}")
        try:
            poses = extract_pose(video_path)
            print(f"Thành công! Kích thước kết quả: {poses.shape}")
            print(f"Visibility mean: {poses[:, :, 3].mean():.3f}")
        except Exception as e:
            print(f"Lỗi: {e}")
    else:
        print("Sử dụng: python src/pose/pose_extractor.py <duong_dan_video>")
