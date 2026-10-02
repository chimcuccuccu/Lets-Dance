"""
Person 1 — SpatialModelV3: 1D-CNN trên diff_sequence → embedding + score.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialModelV3(nn.Module):
    """
    Input:  (B, T, 33, 3) diff_sequence
    Output: score (B, 1), embedding (B, embed_size)

    Embedding được L2-normalize để sẵn sàng cho fusion / FAISS (tuần 6–7).
    """

    def __init__(
        self,
        num_joints: int = 33,
        num_dims: int = 3,
        embed_size: int = 128,
        dropout: float = 0.3,
        normalize_embedding: bool = True,
    ):
        super().__init__()
        self.input_size = num_joints * num_dims
        self.normalize_embedding = normalize_embedding

        self.conv1 = nn.Conv1d(self.input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, embed_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(embed_size)

        # Avg + Max pool: giữ cả tín hiệu trung bình và đỉnh lệch tư thế
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.embed_proj = nn.Linear(embed_size * 2, embed_size)

        self.fc = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """diff_sequence → spatial embedding (B, embed_size)."""
        B, T, J, D = x.shape
        x = x.view(B, T, J * D).permute(0, 2, 1)  # (B, 99, T)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        avg = self.avg_pool(x).squeeze(-1)
        mx = self.max_pool(x).squeeze(-1)
        embedding = self.embed_proj(torch.cat([avg, mx], dim=1))

        if self.normalize_embedding:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, x: torch.Tensor):
        embedding = self.encode(x)
        score = self.fc(embedding)
        return score, embedding


def spatial_model(diff_sequence: torch.Tensor, model: SpatialModelV3) -> torch.Tensor:
    """API fusion: spatial_model(diff) -> embedding."""
    model.eval()
    with torch.no_grad():
        if diff_sequence.dim() == 3:
            diff_sequence = diff_sequence.unsqueeze(0)
        return model.encode(diff_sequence)


if __name__ == "__main__":
    model = SpatialModelV3()
    dummy = torch.randn(8, 150, 33, 3)
    score, embed = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Embedding: {embed.shape} | L2≈1: {(embed.norm(dim=1).mean().item()):.4f}")
    print(f"Score: {score.shape}")
