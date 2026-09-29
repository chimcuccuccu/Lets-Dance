"""Person 1 — extract_pose(video_path) -> np.ndarray (T, 33, 4)."""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import logging
import os

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_pose(video_path: str) -> np.ndarray:
    """
    Trích xuất pose 3D từ video sử dụng MediaPipe.
    
    Args:
        video_path: Đường dẫn tới file video
        
    Returns:
        numpy.ndarray: Mảng 3D shape (T, 33, 4) chứa x, y, z, visibility của 33 landmarks theo T frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")
        
    mp_pose = mp.solutions.pose
    # model_complexity=2 cho độ chính xác cao nhất (nặng hơn), nếu máy yếu có thể đổi thành 1
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, 
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không thể mở video: {video_path}")
        
    frames_landmarks = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Bắt đầu xử lý {video_path} ({frame_count} frames, {fps} fps)")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # OpenCV dùng BGR, MediaPipe yêu cầu RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Để tăng hiệu năng, có thể set image.flags.writeable = False
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = np.zeros((33, 4))
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
            frames_landmarks.append(landmarks)
        else:
            # Không phát hiện được pose -> gán NaN để xử lý interpolate sau
            frames_landmarks.append(np.full((33, 4), np.nan))
            logger.debug(f"Frame {frame_idx}: Không tìm thấy pose.")
            
        frame_idx += 1
        
    cap.release()
    pose.close()
    
    if len(frames_landmarks) == 0:
        raise ValueError(f"Video {video_path} không có frame nào hợp lệ.")
        
    pose_array = np.array(frames_landmarks) # Shape: (T, 33, 4)
    
    # --- XỬ LÝ EDGE CASE: NỘI SUY (INTERPOLATE) CÁC FRAME BỊ MISSING ---
    # Reshape thành 2D (T, 33*4) để dùng pandas nội suy dễ dàng
    T = pose_array.shape[0]
    flattened = pose_array.reshape(T, -1)
    
    df = pd.DataFrame(flattened)
    
    # Kiểm tra số lượng NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"Phát hiện dữ liệu khuyết ({nan_count // (33*4)} frames). Đang tiến hành nội suy tuyến tính...")
        
        # Interpolate tuyến tính cho các frame ở giữa
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        
        # Backfill (bfill) và Forward-fill (ffill) cho các frame bị khuyết ở đầu hoặc cuối video
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        
    # Trả về shape ban đầu (T, 33, 4)
    processed_pose_array = df.to_numpy().reshape(T, 33, 4)
    
    logger.info(f"Hoàn thành trích xuất pose. Kích thước dữ liệu: {processed_pose_array.shape}")
    return processed_pose_array

if __name__ == "__main__":
    import sys
    import os
    import glob
    
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        fallback_dir = r"d:\Study\Đồ án\Demo\lets_dance\data\raw"
        if os.path.exists(fallback_dir):
            videos = glob.glob(os.path.join(fallback_dir, "\*.mp4"))
            if videos:
                video_path = videos[0]
                print(f"Không có tham số đầu vào. Tự động dùng video mẫu: {video_path}")
                
    if video_path and os.path.exists(video_path):
        print(f"Testing video: {video_path}")
        try:
            poses = extract_pose(video_path)
            print(f"Thành công! Kích thước kết quả: {poses.shape}")
        except Exception as e:
            print(f"Lỗi: {e}")
    else:
        print("Sử dụng: python src/pose/pose_extractor.py <duong_dan_video>")

