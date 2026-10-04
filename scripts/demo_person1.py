"""
Demo trực quan Person 1 — test end-to-end những gì đã làm (tuần 2–5).

Xuất PNG vào experiments/spatial_dl_improved/demo/ và in checklist PASS/FAIL.

Chạy:
  python scripts/demo_person1.py
  python scripts/demo_person1.py --show   # mở cửa sổ matplotlib (nếu có GUI)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess import build_diff_sequence, mean_abs_diff  # noqa: E402
from src.spatial_dl.dataset import get_dataloaders  # noqa: E402
from src.spatial_dl.infer import encode_diff, load_spatial_checkpoint, predict_score  # noqa: E402


def _savefig(fig, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Chỉ gọi tight_layout khi figure chưa dùng constrained/tight layout engine
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_train_history(hist_csv: Path, out: Path, show: bool) -> None:
    df = pd.read_csv(hist_csv)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(df["epoch"], df["train_mae"], label="train MAE", alpha=0.85)
    ax.plot(df["epoch"], df["val_mae"], label="val MAE", alpha=0.85)
    best_i = int(df["val_loss"].idxmin())
    ax.axvline(df.loc[best_i, "epoch"], color="gray", ls="--", lw=1, label="best epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (khop_dong_tac)")
    ax.set_title("Spatial DL — training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out, show)


def plot_pred_vs_actual(ckpt: Path, out: Path, show: bool) -> dict:
    model, meta = load_spatial_checkpoint(str(ckpt))
    scale = float(meta.get("score_scale", 100.0))
    target = meta.get("target_col", "khop_dong_tac")
    _, val_loader = get_dataloaders(
        batch_size=32,
        use_dummy=False,
        data_path="poses/diffs",
        csv_path="annotations/scores.csv",
        train_csv="annotations/scores_train.csv",
        val_csv="annotations/scores_test.csv",
        normalize_score=True,
        augment=False,
        target_col=target,
        score_max=scale,
    )
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in val_loader:
            p = predict_score(x, model, score_scale=scale, use_tta=False, return_numpy=True)
            preds.append(np.asarray(p).reshape(-1))
            tgts.append(y.numpy().ravel() * scale)
    preds = np.concatenate(preds)
    tgts = np.concatenate(tgts)
    mae = float(np.mean(np.abs(preds - tgts)))
    r = float(np.corrcoef(preds, tgts)[0, 1]) if len(preds) >= 3 else float("nan")

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(tgts, preds, c="#2a6f97", s=48, alpha=0.8, edgecolors="none")
    lo = float(min(tgts.min(), preds.min()))
    hi = float(max(tgts.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"Val pred vs actual\nMAE={mae:.2f}  Pearson={r:.3f}  n={len(preds)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out, show)
    return {"mae": mae, "pearson": r, "n": len(preds)}


def plot_diff_heatmaps(out: Path, show: bool) -> dict:
    """So sánh |diff| theo thời gian: điểm cao vs điểm thấp."""
    scores = pd.read_csv(ROOT / "annotations" / "scores.csv")
    scores = scores.sort_values("khop_dong_tac")
    low = scores.iloc[0]
    high = scores.iloc[-1]

    def load_diff(row):
        d = str(row["dance_id"])
        v = str(row["video_id"])
        p = ROOT / "poses" / "diffs" / d / f"{v}_diff.npy"
        if not p.exists():
            p = ROOT / "poses" / "diffs" / f"{d}_{v}_diff.npy"
        return np.load(p), d, v

    d_low, dance_l, vid_l = load_diff(low)
    d_high, dance_h, vid_h = load_diff(high)
    # (T, 33) mean abs over xyz
    h_low = np.abs(d_low).mean(axis=-1).T
    h_high = np.abs(d_high).mean(axis=-1).T
    vmax = float(np.percentile(np.concatenate([h_low.ravel(), h_high.ravel()]), 95))

    # constrained_layout tránh colorbar bị “nhô” ra ngoài khi dùng tight_layout
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout="constrained")
    for ax, heat, row, title in (
        (axes[0], h_low, low, "LOW score"),
        (axes[1], h_high, high, "HIGH score"),
    ):
        im = ax.imshow(heat, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(
            f"{title}: {row['video_id']}\n"
            f"khop_dong_tac={row['khop_dong_tac']:.1f}  tong={row['tong_diem']:.1f}"
        )
        ax.set_xlabel("frame (resampled)")
        ax.set_ylabel("joint index")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, label="mean |diff|")
    fig.suptitle("diff_sequence heatmap — lệch hình càng đỏ càng lớn", fontsize=12)
    _savefig(fig, out, show)
    return {
        "low": f"{dance_l}/{vid_l}",
        "high": f"{dance_h}/{vid_h}",
        "mad_low": float(mean_abs_diff(d_low)),
        "mad_high": float(mean_abs_diff(d_high)),
    }


def plot_embedding_pca(ckpt: Path, out: Path, show: bool) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    model, _ = load_spatial_checkpoint(str(ckpt))
    meta = pd.read_csv(ROOT / "annotations" / "scores.csv")
    embs, scores, labels = [], [], []
    for _, row in meta.iterrows():
        d, v = str(row["dance_id"]), str(row["video_id"])
        p = ROOT / "poses" / "diffs" / d / f"{v}_diff.npy"
        if not p.exists():
            continue
        emb = encode_diff(np.load(p), model, use_tta=False, normalize=True)
        embs.append(np.asarray(emb).ravel())
        scores.append(float(row["khop_dong_tac"]))
        labels.append(v)
    X = np.stack(embs)
    y = np.asarray(scores)
    xy = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=y, cmap="RdYlGn", s=40, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="khop_dong_tac")
    ax.set_title(f"Spatial embedding PCA (n={len(y)})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    _savefig(fig, out, show)
    return {"n": len(y)}


def plot_ablation(csv_path: Path, out: Path, show: bool) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(df))
    bars = ax.bar(x, df["MAE"], color="#468faf", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(df["setup"], rotation=20, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Ablation tạm XGBoost — khop_dong_tac (thấp hơn = tốt hơn)")
    for b, r in zip(bars, df["Pearson"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.15,
            f"r={r:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, out, show)


def sample_api_demo(ckpt: Path) -> dict:
    model, meta = load_spatial_checkpoint(str(ckpt))
    scale = float(meta.get("score_scale", 100.0))
    # Một mẫu điểm cao (P003 thường = 100 trên dance_001)
    path = ROOT / "poses" / "diffs" / "dance_001" / "D01_P003_T01_diff.npy"
    if not path.exists():
        path = next((ROOT / "poses" / "diffs").rglob("*_diff.npy"))
    diff = np.load(path)
    emb = encode_diff(diff, model)
    score = float(np.asarray(predict_score(diff, model, score_scale=scale)).ravel()[0])
    return {
        "file": str(path.relative_to(ROOT)),
        "emb_shape": tuple(np.asarray(emb).shape),
        "emb_l2": float(np.linalg.norm(emb)),
        "pred_score": score,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo trực quan Person 1")
    ap.add_argument("--ckpt", default="experiments/spatial_dl_improved/spatial_model_v3_best.pth")
    ap.add_argument("--out-dir", default="experiments/spatial_dl_improved/demo")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    out_dir = ROOT / args.out_dir
    ckpt = ROOT / args.ckpt
    show = args.show

    print("=" * 64)
    print("DEMO PERSON 1 — Spatial DL (tuần 2–5)")
    print("=" * 64)

    if not ckpt.exists():
        print(f"FAIL: thiếu checkpoint {ckpt}")
        print("  Chạy: python -m src.spatial_dl.train --improved --official --epochs 40")
        return 1

    # 1) Diff heatmap
    print("\n[1/5] Diff heatmap (low vs high score)...")
    hm = plot_diff_heatmaps(out_dir / "01_diff_heatmap.png", show)
    print(f"  low={hm['low']} mad={hm['mad_low']:.4f}")
    print(f"  high={hm['high']} mad={hm['mad_high']:.4f}")
    print(f"  → kỳ vọng mad_high < mad_low: {'PASS' if hm['mad_high'] < hm['mad_low'] else 'WARN'}")

    # 2) Train curve
    hist = ROOT / "experiments" / "spatial_dl_improved" / "train_history.csv"
    print("\n[2/5] Training curve...")
    if hist.exists():
        plot_train_history(hist, out_dir / "02_train_curve.png", show)
        print(f"  saved {out_dir / '02_train_curve.png'}")
    else:
        print("  SKIP (không có train_history.csv)")

    # 3) Pred vs actual
    print("\n[3/5] Pred vs actual (val)...")
    metrics = plot_pred_vs_actual(ckpt, out_dir / "03_pred_vs_actual.png", show)
    print(f"  n={metrics['n']} MAE={metrics['mae']:.2f} Pearson={metrics['pearson']:.3f}")

    # 4) Embedding PCA
    print("\n[4/5] Embedding PCA...")
    pca_info = plot_embedding_pca(ckpt, out_dir / "04_embedding_pca.png", show)
    print(f"  n={pca_info['n']} → {out_dir / '04_embedding_pca.png'}")

    # 5) Ablation bars
    print("\n[5/5] Ablation bars...")
    abl = ROOT / "experiments" / "spatial_dl_improved" / "ablation_xgb_khop_dong_tac.csv"
    plot_ablation(abl, out_dir / "05_ablation_xgb.png", show)
    print(f"  → {out_dir / '05_ablation_xgb.png'}")

    # API smoke
    print("\n[API] encode_diff / predict_score...")
    api = sample_api_demo(ckpt)
    print(f"  file={api['file']}")
    print(f"  embedding={api['emb_shape']} L2={api['emb_l2']:.4f}")
    print(f"  predicted khop_dong_tac ≈ {api['pred_score']:.1f}")

    print("\n" + "=" * 64)
    print("XEM ẢNH (mở thư mục):")
    print(f"  {out_dir}")
    for name in sorted(out_dir.glob("*.png")):
        print(f"  - {name.name}")
    print("\nLệnh test khác:")
    print("  python scripts/qa_person1.py")
    print("  python src/visualize.py poses/dance_001/D01_P003_T01.npy")
    print("  python src/test_pipeline.py --model")
    print("  python scripts/eval_spatial.py")
    print("  python scripts/visualize_spatial_embeddings.py")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
