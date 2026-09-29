"""Person 1 — Chạy thử trên toàn bộ video reference + performer đã quay → lưu .npy cho từng file."""
import os
import glob
import numpy as np
import logging
from src.pose.pose_extractor import extract_pose

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_all_videos(data_dir: str, output_dir: str):
    """
    Quét toàn bộ video trong data_dir và trích xuất pose lưu vào output_dir.
    """
    video_extensions = ('*.mp4', '*.avi', '*.mov')
    video_paths = []
    for ext in video_extensions:
        # Lấy tất cả video trong cấu trúc data/dances/dance_xxx/
        search_pattern = os.path.join(data_dir, '**', ext)
        video_paths.extend(glob.glob(search_pattern, recursive=True))
        
    if not video_paths:
        logger.warning(f"Không tìm thấy video nào trong {data_dir}")
        return
        
    logger.info(f"Tìm thấy {len(video_paths)} videos. Bắt đầu trích xuất...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for video_path in video_paths:
        try:
            # Tạo đường dẫn lưu file tương ứng
            # Ví dụ: data/dances/dance_001/reference/vid.mp4 
            # -> poses/dance_001_reference_vid.npy
            rel_path = os.path.relpath(video_path, data_dir)
            safe_name = rel_path.replace(os.sep, '_').rsplit('.', 1)[0] + '.npy'
            output_path = os.path.join(output_dir, safe_name)
            
            if os.path.exists(output_path):
                logger.info(f"Bỏ qua {video_path} vì đã tồn tại file pose.")
                continue
                
            poses = extract_pose(video_path)
            np.save(output_path, poses)
            logger.info(f"Lưu thành công pose cho {video_path} tại {output_path}")
        except Exception as e:
            logger.error(f"Lỗi khi xử lý {video_path}: {e}")

if __name__ == "__main__":
    # Đảm bảo import PYTHONPATH đúng nếu chạy từ ngoài gốc
    # python -m src.pose.extract_all
    
    # Path tương đối dựa vào gốc repo
    # data_dir = data/dances, output_dir = poses
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data", "dances")
    OUTPUT_DIR = os.path.join(BASE_DIR, "poses")
    
    process_all_videos(DATA_DIR, OUTPUT_DIR)
    logger.info("Hoàn tất batch processing!")
