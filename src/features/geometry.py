"""
Person 2 — compute_geometry_features(pose) -> features (T, k).
Tính toán các góc khớp (geometry features) từ pose 3D (T, 33, 3|4).
"""
import numpy as np

# Định nghĩa các khớp theo MediaPipe Pose
# Format dictionary: "tên_góc": (điểm_1, điểm_chốt, điểm_3)
ANGLE_DEFINITIONS = {
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (23, 11, 13),
    "right_shoulder": (24, 12, 14),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
}

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Tính góc (đơn vị: độ) tại điểm p2, tạo bởi 2 vector p2->p1 và p2->p3.
    p1, p2, p3 có shape (T, 3).
    Output: np.ndarray shape (T,) với giá trị từ 0 đến 180 độ.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Tính dot product và magnitude
    dot = np.einsum('ij,ij->i', v1, v2)
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    
    # Tránh chia cho 0
    mask = (mag1 > 1e-6) & (mag2 > 1e-6)
    
    angles = np.zeros_like(dot)
    
    # Cosine = dot / (mag1 * mag2), clip để tránh lỗi floating point > 1.0 hoặc < -1.0
    cos_theta = np.clip(dot[mask] / (mag1[mask] * mag2[mask]), -1.0, 1.0)
    
    angles[mask] = np.degrees(np.arccos(cos_theta))
    
    # Thay thế những chỗ bị mất (do NaN hoặc vector=0) bằng 0 (hoặc có thể dùng nội suy)
    angles[~np.isfinite(angles)] = 0.0
    
    return angles

def compute_geometry_features(pose: np.ndarray) -> np.ndarray:
    """
    Từ mảng pose shape (T, 33, 3|4), trích xuất mảng đặc trưng góc khớp.
    Output: (T, 8) mảng numpy chứa 8 góc (bên trái/phải cho gối, khuỷu, vai, hông).
    """
    T = pose.shape[0]
    num_features = len(ANGLE_DEFINITIONS)
    features = np.zeros((T, num_features), dtype=np.float32)
    
    # Đảm bảo chỉ lấy XYZ
    coords = pose[:, :, :3]
    
    for i, (name, (j1, j2, j3)) in enumerate(ANGLE_DEFINITIONS.items()):
        p1 = coords[:, j1, :]
        p2 = coords[:, j2, :]
        p3 = coords[:, j3, :]
        features[:, i] = calculate_angle(p1, p2, p3)
        
    return features

if __name__ == "__main__":
    # Test stub
    T = 100
    dummy_pose = np.random.rand(T, 33, 4).astype(np.float32)
    feat = compute_geometry_features(dummy_pose)
    print(f"Geometry features shape: {feat.shape} (Expected: {T}, {len(ANGLE_DEFINITIONS)})")
    print(f"Sample features [0]: {feat[0]}")
