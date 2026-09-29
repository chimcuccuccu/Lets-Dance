"""Person 1 — Preprocessing & Alignment (Tiền xử lý và Khớp chuỗi)"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def get_hip_center(pose: np.ndarray) -> np.ndarray:
    """
    Tính trung bình tọa độ của Left Hip (23) và Right Hip (24) theo chuẩn MediaPipe Pose.
    """
    return (pose[:, 23, :3] + pose[:, 24, :3]) / 2.0

def center_to_hip(pose: np.ndarray) -> np.ndarray:
    """
    Đưa gốc tọa độ về hông (Hip Center) để các vị trí đứng khác nhau trong khung hình không ảnh hưởng.
    """
    coords = pose[:, :, :3].copy()
    hip_center = get_hip_center(pose) # Shape: (T, 3)
    coords = coords - hip_center[:, np.newaxis, :]
    
    # Nếu có visibility (D=4), ghép lại
    if pose.shape[2] == 4:
        return np.concatenate([coords, pose[:, :, 3:]], axis=2)
    return coords

def scale_by_height(pose: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Chuẩn hoá tỷ lệ theo kích thước người nhảy.
    Sử dụng trung bình độ dài các xương (L2 norm) để scale.
    """
    coords = pose[:, :, :3].copy()
    scale = np.sqrt((coords ** 2).sum(axis=-1)).mean() + eps
    coords = coords / scale
    
    if pose.shape[2] == 4:
        return np.concatenate([coords, pose[:, :, 3:]], axis=2)
    return coords

def smooth_sequence(pose: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Làm mượt nhiễu (jitter) trong chuỗi pose bằng bộ lọc Savitzky-Golay.
    """
    # Đảm bảo window_length lẻ và phù hợp với số frame
    if pose.shape[0] < window_length:
        window_length = pose.shape[0] if pose.shape[0] % 2 != 0 else pose.shape[0] - 1
        
    if window_length <= polyorder:
        return pose # Không đủ frame để mượt, trả về nguyên gốc
        
    smoothed = savgol_filter(pose, window_length=window_length, polyorder=polyorder, axis=0)
    return smoothed

def normalize_length(pose: np.ndarray, target_frames: int = 150) -> np.ndarray:
    """
    Chuẩn hóa độ dài sequence về `target_frames` bằng nội suy tuyến tính (Linear Interpolation).
    """
    T, J, D = pose.shape
    if T == target_frames:
        return pose
        
    t_orig = np.linspace(0, 1, T)
    t_target = np.linspace(0, 1, target_frames)
    resampled = np.zeros((target_frames, J, D), dtype=np.float32)
    
    for j in range(J):
        for d in range(D):
            f = interp1d(t_orig, pose[:, j, d], kind='linear', fill_value='extrapolate')
            resampled[:, j, d] = f(t_target)
            
    return resampled

def preprocess_pipeline(pose: np.ndarray, target_frames: int = -1) -> np.ndarray:
    """
    Chạy toàn bộ pipeline tiền xử lý (Smoothing -> Centering -> Scaling -> Resampling).
    Nếu target_frames > 0, tiến hành chuẩn hóa số khung hình.
    """
    pose = smooth_sequence(pose)
    pose = center_to_hip(pose)
    pose = scale_by_height(pose)
    if target_frames > 0:
        pose = normalize_length(pose, target_frames)
    return pose

def align_and_compute_diff(performer_pose: np.ndarray, reference_pose: np.ndarray, alignment_path: list = None) -> np.ndarray:
    """
    Khớp thời gian 2 chuỗi (Alignment) và tính độ lệch (Diff Sequence).
    - alignment_path: danh sách cặp index [(i, j), ...] từ hàm DTW của Person 2.
    - Trả về diff_sequence: (T_aligned, 33, 3)
    """
    # Chỉ lấy tọa độ x, y, z (bỏ qua visibility)
    p_coords = performer_pose[:, :, :3]
    r_coords = reference_pose[:, :, :3]

    if alignment_path is None:
        # Nếu chưa có DTW từ Person 2, yêu cầu 2 chuỗi đã được chuẩn hóa về cùng chiều dài (vd 150 frame)
        if p_coords.shape[0] != r_coords.shape[0]:
            raise ValueError("Nếu không có alignment_path (DTW), 2 chuỗi phải được normalize về cùng số frame.")
        aligned_performer = p_coords
        aligned_reference = r_coords
    else:
        # Dùng Alignment Path của Person 2 để cắt/khớp (Warping)
        idx_p = [p[0] for p in alignment_path]
        idx_r = [p[1] for p in alignment_path]
        aligned_performer = p_coords[idx_p]
        aligned_reference = r_coords[idx_r]
        
    # diff_sequence = performer - reference (đã align)
    diff_sequence = aligned_performer - aligned_reference
    return diff_sequence
