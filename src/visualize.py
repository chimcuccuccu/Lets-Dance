import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Khai báo các điểm nối (bones) của MediaPipe Pose (33 điểm)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def visualize_pose(npy_path):
    print(f"Đang tải dữ liệu từ {npy_path}...")
    poses = np.load(npy_path)
    
    # Poses shape có thể là (T, 33, 3) hoặc (T, 33, 4). Ta chỉ lấy x, y, z
    coords = poses[:, :, :3]
    T, J, D = coords.shape
    
    # Do matplotlib 3D mặc định là Z hướng lên, Y hướng vào trong, X hướng ngang
    # MediaPipe thì Y hướng xuống, Z hướng từ camera vào trong.
    # Nên ta cần đổi lại một chút để xem cho thuận mắt:
    X = coords[:, :, 0]
    Y = -coords[:, :, 1] # Đảo ngược trục Y để đầu hướng lên trên
    Z = coords[:, :, 2]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Thiết lập giới hạn trục dựa trên min/max của dữ liệu để khung hình không bị giật
    ax.set_xlim3d([np.min(X), np.max(X)])
    ax.set_ylim3d([np.min(Z), np.max(Z)]) # Đảo Z vào trục Y của plot
    ax.set_zlim3d([np.min(Y), np.max(Y)]) # Đảo Y vào trục Z của plot
    
    ax.set_xlabel('X')
    ax.set_ylabel('Depth (Z)')
    ax.set_zlabel('Height (Y)')
    ax.set_title("3D Pose Visualization\n(Click giữ chuột trái và kéo để xoay góc nhìn)")
    
    # Đặt góc nhìn mặc định trực diện (Elevation: 10 độ, Azimuth: -90 độ)
    # Góc này giúp lúc mở lên sẽ nhìn thẳng vào mặt người nhảy giống góc camera video.
    ax.view_init(elev=10, azim=-90)
    
    # Khởi tạo các đoạn thẳng (bones)
    lines = [ax.plot([], [], [], color='blue', linewidth=2)[0] for _ in POSE_CONNECTIONS]
    
    def update(frame_idx):
        for line, connection in zip(lines, POSE_CONNECTIONS):
            idx1, idx2 = connection
            # Lấy toạ độ 2 điểm
            x_line = [X[frame_idx, idx1], X[frame_idx, idx2]]
            y_line = [Z[frame_idx, idx1], Z[frame_idx, idx2]] # map Z -> trục y
            z_line = [Y[frame_idx, idx1], Y[frame_idx, idx2]] # map Y -> trục z
            
            line.set_data(x_line, y_line)
            line.set_3d_properties(z_line)
        return lines

    print("Đang tạo Animation... (Vui lòng đợi vài giây)")
    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/30, blit=False)
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python src/visualize.py <đường_dẫn_file.npy>")
        print("Ví dụ: python src/visualize.py test_output.npy")
    else:
        visualize_pose(sys.argv[1])
