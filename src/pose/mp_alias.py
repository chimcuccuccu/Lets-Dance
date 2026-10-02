"""Alias theo docs cũ: mp_alias.py → pose_extractor.extract_pose.

Không đặt tên mediapipe.py — khi chạy `python src/pose/pose_extractor.py`
thư mục script vào sys.path và sẽ che package mediapipe thật.
"""
from src.pose.pose_extractor import extract_pose

__all__ = ["extract_pose"]
