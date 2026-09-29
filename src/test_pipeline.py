import os
import sys
import numpy as np

# Thêm đường dẫn gốc vào sys.path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose.pose_extractor import extract_pose
from src.preprocessing.preprocess import preprocess_pipeline, get_hip_center

def test_pipeline(video_path):
    print(f"--- BẮT ĐẦU TEST: {video_path} ---")
    
    # Bước 1: Trích xuất Pose
    print("1. Đang trích xuất pose bằng MediaPipe...")
    raw_poses = extract_pose(video_path)
    print(f"   -> Thành công! Kích thước Raw Pose: {raw_poses.shape}")
    
    # In thử vị trí hông trước khi tiền xử lý
    hip_before = get_hip_center(raw_poses)
    print(f"   -> Tọa độ Hông (Hip Center) ở frame 0 trước xử lý: {hip_before[0]}")
    
    # Bước 2: Tiền xử lý (Smooth, Center, Scale, Resample)
    print("\n2. Đang chạy pipeline tiền xử lý (Center, Scale, Smooth, Normalize về 150 frames)...")
    processed_poses = preprocess_pipeline(raw_poses, target_frames=150)
    print(f"   -> Thành công! Kích thước Processed Pose: {processed_poses.shape}")
    
    # Kiểm tra xem hông đã được đưa về (0,0,0) chưa
    hip_after = get_hip_center(processed_poses)
    print(f"   -> Tọa độ Hông (Hip Center) ở frame 0 sau xử lý: {hip_after[0]}")
    
    # Bước 3: Lưu thử ra file
    output_path = "test_output.npy"
    np.save(output_path, processed_poses)
    print(f"\n3. Đã lưu file test thành công tại: {output_path}")
    print("--- HOÀN TẤT ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        default_video = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos", "test_video.mp4")
        if os.path.exists(default_video):
            print(f"Sử dụng video mặc định: {default_video}")
            test_pipeline(default_video)
        else:
            print("Cách sử dụng: python src/test_pipeline.py <đường_dẫn_tới_video_bất_kỳ.mp4>")
    else:
        test_pipeline(sys.argv[1])
