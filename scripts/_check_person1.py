"""Synthetic checks for optimized Person 1 pipeline."""
import numpy as np
import torch

from src.preprocessing.preprocess import (
    build_diff_sequence,
    compute_diff,
    get_hip_center,
    mean_abs_diff,
    preprocess_pipeline,
)
from src.spatial_dl.dataset import split_indices_by_person
from src.spatial_dl.model_v3 import SpatialModelV3, spatial_model
from src.spatial_dl.train import train_model


def main():
    rng = np.random.default_rng(0)
    T = 60
    pose = rng.normal(0.5, 0.05, size=(T, 33, 4)).astype(np.float32)
    pose[:, :, 3] = rng.uniform(0.6, 1.0, size=(T, 33)).astype(np.float32)
    pose[:, 23, :3] = 0.45
    pose[:, 24, :3] = 0.55
    pose[:, 11, :3] = [0.4, 0.3, 0.0]
    pose[:, 12, :3] = [0.6, 0.3, 0.0]
    pose[10:15, 15, 3] = 0.1

    out = preprocess_pipeline(pose, target_frames=150)
    assert out.shape == (150, 33, 4), out.shape
    hip = get_hip_center(out)
    assert np.allclose(hip, 0, atol=1e-4), hip[:3]
    print("OK preprocess", out.shape, "hip~0")

    # Jitter hình dạng (không phải tịnh tiến toàn cục — center hip sẽ nuốt offset đều)
    pose_bad = pose.copy()
    pose_bad[:, 15, :3] += 0.15  # lệch cổ tay trái
    pose_bad[:, 16, :3] -= 0.10
    d0 = build_diff_sequence(pose, pose, target_frames=150)
    d1 = build_diff_sequence(pose, pose_bad, target_frames=150)
    m0, m1 = mean_abs_diff(d0), mean_abs_diff(d1)
    assert m0 < 1e-5, m0
    assert m1 > m0 * 10, (m0, m1)
    print(f"OK diff sanity self={m0:.2e} shape-error={m1:.4f}")

    path = [(i, max(0, i - 2)) for i in range(T)]
    p = preprocess_pipeline(pose, target_frames=-1)
    r = preprocess_pipeline(pose, target_frames=-1)
    d_aligned = compute_diff(p, r, alignment_path=path)
    assert d_aligned.shape[0] == T
    print("OK DTW path align", d_aligned.shape)

    model = SpatialModelV3()
    x = torch.from_numpy(d1).unsqueeze(0)
    score, emb = model(x)
    assert score.shape == (1, 1) and emb.shape == (1, 128)
    assert abs(emb.norm().item() - 1.0) < 1e-4
    emb2 = spatial_model(torch.from_numpy(d1), model)
    assert emb2.shape == (1, 128)
    print("OK model", float(score.item()), emb.norm().item())

    ids = ["a", "a", "b", "b", "c", "c"]
    tr, va = split_indices_by_person(ids, val_ratio=0.34, seed=0)
    assert set(ids[i] for i in tr).isdisjoint(set(ids[i] for i in va))
    print("OK person split", tr, va)

    train_model(
        num_epochs=2,
        batch_size=8,
        use_dummy=True,
        checkpoint_dir="experiments/spatial_dl_test",
    )
    print("OK dummy train")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
