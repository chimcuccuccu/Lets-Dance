"""
Tuần 5 Person 1 — Ablation tạm qua XGBoost:

  1) Raw geometry          (góc khớp Person 2 — baseline không DL)
  2) Raw diff stats        (thống kê |diff| — baseline thuần Person 1)
  3) Spatial embedding     (Spatial DL)
  4) Geometry + DTW        (baseline thủ công)
  5) Spatial + Geo + DTW   (full tạm, chưa phải fusion tuần 6)

Target mặc định: tong_diem. Split theo person_id.
Chỉ *đọc* output Person 2 (geometry/DTW), không sửa code của họ.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.spatial_dl.dataset import split_indices_by_person
from src.spatial_dl.infer import encode_diff, load_spatial_checkpoint
from src.preprocessing.preprocess import normalize_length


def _diff_path(data_dir: str, dance_id: str, video_id: str) -> Optional[str]:
    for p in (
        os.path.join(data_dir, dance_id, f"{video_id}_diff.npy"),
        os.path.join(data_dir, f"{dance_id}_{video_id}_diff.npy"),
    ):
        if os.path.isfile(p):
            return p
    return None


def _pose_path(poses_dir: str, dance_id: str, video_id: str) -> Optional[str]:
    for p in (
        os.path.join(poses_dir, dance_id, f"{video_id}.npy"),
        os.path.join(poses_dir, f"{dance_id}_{video_id}.npy"),
    ):
        if os.path.isfile(p):
            return p
    return None


def _raw_diff_stats(diff: np.ndarray) -> np.ndarray:
    """Baseline Person 1: mean/std/max |diff| theo nhóm khớp + toàn cục."""
    abs_d = np.abs(diff.astype(np.float32))  # (T, 33, 3)
    # nhóm khớp MediaPipe gần đúng
    groups = {
        "face": list(range(0, 11)),
        "arms": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "torso": [11, 12, 23, 24],
        "legs": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    }
    feats = []
    flat = abs_d.reshape(abs_d.shape[0], -1)
    feats.extend([flat.mean(), flat.std(), flat.max()])
    for idxs in groups.values():
        g = abs_d[:, idxs, :].reshape(abs_d.shape[0], -1)
        feats.extend([g.mean(), g.std(), g.max()])
    # per-joint mean abs (33)
    feats.extend(abs_d.mean(axis=(0, 2)).tolist())
    return np.asarray(feats, dtype=np.float32)


def _geo_summary(pose: np.ndarray) -> Optional[np.ndarray]:
    try:
        from src.features.geometry import compute_geometry_features
    except Exception:
        return None
    feat = compute_geometry_features(pose)  # (T, 8)
    return np.concatenate(
        [feat.mean(axis=0), feat.std(axis=0), feat.min(axis=0), feat.max(axis=0)]
    ).astype(np.float32)


def _dtw_row_features(row: pd.Series) -> np.ndarray:
    total = float(row.get("dtw_distance_total", 0.0) or 0.0)
    seg_cols = [c for c in row.index if str(c).startswith("dtw_seg_")]
    if not seg_cols:
        return np.array([total, 0.0, 0.0, 0.0], dtype=np.float32)
    segs = pd.to_numeric(row[seg_cols], errors="coerce").to_numpy(dtype=np.float64)
    segs = segs[np.isfinite(segs)]
    if segs.size == 0:
        return np.array([total, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array(
        [total, float(segs.mean()), float(segs.std()), float(segs.max())],
        dtype=np.float32,
    )


def _fit_xgb(X_train, y_train, X_val, y_val, seed: int = 42) -> Dict[str, float]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("Cần xgboost trong requirements.txt") from exc

    # Dataset nhỏ → model XGB vừa phải, tránh overfit feature nhiều chiều
    n_feat = X_train.shape[1]
    model = XGBRegressor(
        n_estimators=250,
        max_depth=3 if n_feat > 64 else 4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.1,
        min_child_weight=3,
        random_state=seed,
        n_jobs=4,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    pearson = float("nan")
    if len(y_val) > 2 and np.std(y_val) > 1e-8 and np.std(pred) > 1e-8:
        pearson = float(np.corrcoef(y_val, pred)[0, 1])
    return {
        "MAE": float(mean_absolute_error(y_val, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
        "R2": float(r2_score(y_val, pred)),
        "Pearson": pearson,
        "n_features": int(n_feat),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="annotations/scores.csv")
    ap.add_argument("--dtw", default="annotations/dtw_features.csv")
    ap.add_argument("--diffs", default="poses/diffs")
    ap.add_argument("--poses", default="poses")
    ap.add_argument("--ckpt", default="experiments/spatial_dl_improved/spatial_model_v3_best.pth")
    ap.add_argument("--target", default="tong_diem")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="experiments/spatial_dl_improved/ablation_xgb.csv",
        help="CSV bảng kết quả ablation",
    )
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=32,
        help="PCA-reduce feature > pca-dim trước XGB (0=tắt). Cần thiết với embedding 256-d.",
    )
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
    if os.path.isfile(args.dtw):
        dtw = pd.read_csv(args.dtw)
        df = scores.merge(dtw, on=["dance_id", "video_id"], how="left")
    else:
        print(f"WARNING: thiếu {args.dtw} — bỏ DTW features")
        df = scores.copy()
    if args.target not in df.columns:
        raise SystemExit(f"Thiếu cột target {args.target}")

    spatial_net = None
    device = torch.device("cpu")
    if os.path.isfile(args.ckpt):
        spatial_net, _ = load_spatial_checkpoint(args.ckpt)
        device = next(spatial_net.parameters()).device
        print(f"Loaded spatial ckpt: {args.ckpt}")
    else:
        print(f"WARNING: no spatial ckpt at {args.ckpt} — skip spatial features")

    rows = []
    skipped = 0
    for _, r in df.iterrows():
        dance_id, video_id = str(r["dance_id"]), str(r["video_id"])
        dpath = _diff_path(args.diffs, dance_id, video_id)
        ppath = _pose_path(args.poses, dance_id, video_id)
        if not dpath:
            skipped += 1
            continue
        diff = np.load(dpath).astype(np.float32)
        if diff.shape[0] != 150:
            diff = normalize_length(diff, 150)
        raw_diff = _raw_diff_stats(diff)
        geo = None
        if ppath:
            pose = np.load(ppath).astype(np.float32)
            geo = _geo_summary(pose)
        dtw_f = _dtw_row_features(r)
        if spatial_net is not None:
            emb = encode_diff(diff, spatial_net, return_numpy=True)
            assert isinstance(emb, np.ndarray)
        else:
            emb = np.zeros(256, dtype=np.float32)
        rows.append(
            {
                "person_id": str(r.get("person_id", video_id)),
                "y": float(r[args.target]),
                "geo": geo,
                "raw_diff": raw_diff,
                "dtw": dtw_f,
                "spatial": emb.astype(np.float32),
            }
        )

    if len(rows) < 10:
        raise SystemExit(f"Quá ít mẫu hợp lệ: {len(rows)} (skipped={skipped})")

    persons = [x["person_id"] for x in rows]
    y = np.array([x["y"] for x in rows], dtype=np.float32)
    X_raw = np.stack([x["raw_diff"] for x in rows])
    X_sp = np.stack([x["spatial"] for x in rows])
    X_dtw = np.stack([x["dtw"] for x in rows])

    has_geo = all(x["geo"] is not None for x in rows)
    if has_geo:
        X_geo = np.stack([x["geo"] for x in rows])
        X_geo_dtw = np.concatenate([X_geo, X_dtw], axis=1)
        X_full = np.concatenate([X_sp, X_geo, X_dtw], axis=1)
    else:
        X_geo = None
        X_geo_dtw = X_dtw
        X_full = np.concatenate([X_sp, X_dtw], axis=1)
        print("WARNING: geometry không dùng được — bỏ setup Raw geometry")

    train_idx, val_idx = split_indices_by_person(persons, val_ratio=0.2, seed=args.seed)

    setups = {}
    if X_geo is not None:
        setups["Raw geometry"] = X_geo
    setups["Raw diff stats"] = X_raw
    setups["Spatial only"] = X_sp
    setups["Geometry+DTW"] = X_geo_dtw
    setups["Spatial+Geo+DTW"] = X_full

    print(
        f"n={len(rows)} train={len(train_idx)} val={len(val_idx)} "
        f"target={args.target} pca_dim={args.pca_dim}"
    )
    print(f"{'setup':<22} {'#f':>4} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'r':>8}")

    results = []
    for name, X in setups.items():
        # Fit PCA chỉ trên train (tránh leakage) khi feature cao chiều
        if args.pca_dim > 0 and X.shape[1] > args.pca_dim:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            n_comp = min(args.pca_dim, len(train_idx) - 1, X.shape[1])
            scaler = StandardScaler().fit(X[train_idx])
            Xs = scaler.transform(X)
            pca = PCA(n_components=n_comp, random_state=args.seed).fit(Xs[train_idx])
            X_use = pca.transform(Xs).astype(np.float32)
        else:
            X_use = X
        metrics = _fit_xgb(
            X_use[train_idx], y[train_idx], X_use[val_idx], y[val_idx], seed=args.seed
        )
        print(
            f"{name:<22} {metrics['n_features']:4d} {metrics['MAE']:8.2f} "
            f"{metrics['RMSE']:8.2f} {metrics['R2']:8.3f} {metrics['Pearson']:8.3f}"
        )
        results.append({"setup": name, "target": args.target, **metrics})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["setup", "target", "n_features", "MAE", "RMSE", "R2", "Pearson"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved ablation table → {args.out}")

    # Verdict ngắn cho báo cáo
    by_name = {r["setup"]: r for r in results}
    if "Spatial only" in by_name and "Raw geometry" in by_name:
        sp, geo = by_name["Spatial only"], by_name["Raw geometry"]
        delta = geo["MAE"] - sp["MAE"]
        print(
            f"\nSpatial vs Raw geometry: ΔMAE={delta:+.2f} "
            f"(dương = Spatial tốt hơn trên MAE)"
        )
    if "Spatial only" in by_name and "Raw diff stats" in by_name:
        sp, raw = by_name["Spatial only"], by_name["Raw diff stats"]
        print(
            f"Spatial vs Raw diff stats: ΔMAE={raw['MAE'] - sp['MAE']:+.2f}"
        )


if __name__ == "__main__":
    main()
