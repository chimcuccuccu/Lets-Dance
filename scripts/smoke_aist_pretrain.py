"""
Smoke test AIST pretrain pipeline without downloading the full dataset.

Creates a tiny fake keypoints3d tree, converts, trains 1 epoch, loads into SpatialModelV3.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.spatial_dl.joints_convert import coco17_to_mediapipe, smpl24_to_mediapipe
from src.spatial_dl.model_v3 import SpatialAutoEncoder, SpatialModelV3, load_pretrained_encoder
from src.spatial_dl.pretrain import pretrain


def _fake_coco(T: int = 120) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)[:, None]
    base = np.zeros((T, 17, 3), dtype=np.float32)
    # Rough standing figure
    base[:, 0] = [0, 1.7, 0]  # nose
    base[:, 5] = [-0.2, 1.4, 0]  # L shoulder
    base[:, 6] = [0.2, 1.4, 0]
    base[:, 7] = [-0.25, 1.1, 0]
    base[:, 8] = [0.25, 1.1, 0]
    base[:, 9] = [-0.3, 0.8, 0]
    base[:, 10] = [0.3, 0.8, 0]
    base[:, 11] = [-0.15, 1.0, 0]
    base[:, 12] = [0.15, 1.0, 0]
    base[:, 13] = [-0.15, 0.5, 0]
    base[:, 14] = [0.15, 0.5, 0]
    base[:, 15] = [-0.15, 0.05, 0]
    base[:, 16] = [0.15, 0.05, 0]
    # Animate arms
    base[:, 9, 0] += 0.1 * np.sin(t[:, 0])
    base[:, 10, 0] += 0.1 * np.cos(t[:, 0])
    return base


def test_convert():
    mp = coco17_to_mediapipe(_fake_coco())
    assert mp.shape == (120, 33, 4), mp.shape
    assert mp[:, 11, 3].mean() > 0.9  # mapped shoulder vis
    smpl = np.random.randn(40, 24, 3).astype(np.float32)
    mp2 = smpl24_to_mediapipe(smpl)
    assert mp2.shape == (40, 33, 4)
    print("OK convert", mp.shape, mp2.shape)


def test_pretrain_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "aistpp")
        kp = os.path.join(root, "keypoints3d")
        os.makedirs(kp)
        genres = ["gBR", "gPO", "gLO", "gWA"]
        for i, g in enumerate(genres):
            for d in (1, 2):
                name = f"{g}_sBM_cAll_d{d:02d}_mBR0_ch{i:02d}"
                with open(os.path.join(kp, f"{name}.pkl"), "wb") as f:
                    pickle.dump({"keypoints3d_optim": _fake_coco(90 + i * 5)}, f)

        ckpt = os.path.join(tmp, "spatial_pretrained.pt")
        pretrain(
            aist_root=root,
            epochs=1,
            batch_size=4,
            mode="synthetic_diff",
            max_sequences=0,
            checkpoint_path=ckpt,
            target_frames=60,
            window_frames=60,
        )
        assert os.path.isfile(ckpt)

        model = SpatialModelV3()
        n_loaded, n_missing = load_pretrained_encoder(model, ckpt)
        assert n_loaded > 0
        x = torch.randn(2, 60, 33, 3)
        score, emb = model(x)
        assert score.shape == (2, 1) and emb.shape == (2, 128)
        print("OK pretrain smoke", "loaded", n_loaded, "missing", n_missing)


def test_ae_shapes():
    ae = SpatialAutoEncoder()
    x = torch.randn(4, 80, 33, 3)
    recon, emb = ae(x)
    assert recon.shape == x.shape and emb.shape == (4, 128)
    print("OK AE shapes")


if __name__ == "__main__":
    test_convert()
    test_ae_shapes()
    test_pretrain_smoke()
    print("All smoke tests passed.")
