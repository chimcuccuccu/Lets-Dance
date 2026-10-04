"""
Tuần 5 Person 1 — Xuất + visualize spatial_embedding (PCA / t-SNE).

Sanity check định tính: video tong_diem cao có xu hướng tách khỏi điểm thấp?

Chạy:
  python scripts/visualize_spatial_embeddings.py
  python scripts/visualize_spatial_embeddings.py --ckpt experiments/spatial_dl_improved/spatial_model_v3_best.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.spatial_dl.infer import export_spatial_embeddings  # noqa: E402


def _score_bins(scores: np.ndarray) -> np.ndarray:
    """3 nhóm: low / mid / high theo tertile (ổn định hơn fixed threshold)."""
    q1, q2 = np.nanpercentile(scores, [33.3, 66.6])
    labels = np.full(scores.shape, "mid", dtype=object)
    labels[scores <= q1] = "low"
    labels[scores >= q2] = "high"
    return labels


def _scatter(
    xy: np.ndarray,
    scores: np.ndarray,
    title: str,
    out_path: str,
    score_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=scores,
        cmap="RdYlGn",
        s=42,
        alpha=0.85,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(score_name)
    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _cluster_separation_report(xy: np.ndarray, scores: np.ndarray) -> dict:
    """Khoảng cách centroid low vs high (sanity metric đơn giản)."""
    bins = _score_bins(scores)
    out = {}
    for name in ("low", "mid", "high"):
        mask = bins == name
        if mask.sum() == 0:
            continue
        out[f"n_{name}"] = int(mask.sum())
        out[f"mean_score_{name}"] = float(np.mean(scores[mask]))
        out[f"centroid_{name}"] = xy[mask].mean(axis=0)
    if "centroid_low" in out and "centroid_high" in out:
        out["centroid_dist_low_high"] = float(
            np.linalg.norm(out["centroid_high"] - out["centroid_low"])
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA/t-SNE spatial embeddings (Person 1)")
    ap.add_argument("--ckpt", default="experiments/spatial_dl_improved/spatial_model_v3_best.pth")
    ap.add_argument("--scores", default="annotations/scores.csv")
    ap.add_argument("--diffs", default="poses/diffs")
    ap.add_argument("--out-dir", default="experiments/spatial_dl_improved")
    ap.add_argument("--score-col", default="tong_diem", choices=["tong_diem", "khop_dong_tac"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=20.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    emb_path = os.path.join(args.out_dir, "spatial_embeddings.npy")
    meta_path = os.path.join(args.out_dir, "spatial_embeddings_meta.csv")

    embeddings, meta = export_spatial_embeddings(
        args.scores,
        args.diffs,
        args.ckpt,
        out_npy=emb_path,
        out_meta=meta_path,
    )
    if args.score_col not in meta.columns:
        raise SystemExit(f"Thiếu cột {args.score_col} trong scores")

    scores = meta[args.score_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(scores)
    embeddings = embeddings[valid]
    scores = scores[valid]
    meta = meta.loc[valid].reset_index(drop=True)

    # PCA
    x_std = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=args.seed)
    xy_pca = pca.fit_transform(x_std)
    pca_png = os.path.join(args.out_dir, f"embeddings_pca_{args.score_col}.png")
    _scatter(
        xy_pca,
        scores,
        f"Spatial embedding PCA (var={pca.explained_variance_ratio_.sum():.1%}) — {args.score_col}",
        pca_png,
        args.score_col,
    )

    # t-SNE
    n = len(embeddings)
    perplexity = min(args.perplexity, max(5.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    xy_tsne = tsne.fit_transform(x_std)
    tsne_png = os.path.join(args.out_dir, f"embeddings_tsne_{args.score_col}.png")
    _scatter(
        xy_tsne,
        scores,
        f"Spatial embedding t-SNE (perplexity={perplexity:.0f}) — {args.score_col}",
        tsne_png,
        args.score_col,
    )

    # Lưu toạ độ + báo cáo tách cụm
    coords = meta.copy()
    coords["pca_1"] = xy_pca[:, 0]
    coords["pca_2"] = xy_pca[:, 1]
    coords["tsne_1"] = xy_tsne[:, 0]
    coords["tsne_2"] = xy_tsne[:, 1]
    coords_path = os.path.join(args.out_dir, f"embeddings_2d_{args.score_col}.csv")
    coords.to_csv(coords_path, index=False)

    report = _cluster_separation_report(xy_pca, scores)
    report_lines = [
        "=== Spatial embedding sanity (PCA) ===",
        f"n={len(scores)}  score={args.score_col}",
        f"PCA explained variance: {pca.explained_variance_ratio_} (sum={pca.explained_variance_ratio_.sum():.3f})",
        f"score mean±std: {scores.mean():.1f} ± {scores.std():.1f}",
    ]
    for k in ("n_low", "n_mid", "n_high", "mean_score_low", "mean_score_mid", "mean_score_high"):
        if k in report:
            report_lines.append(f"{k}: {report[k]}")
    if "centroid_dist_low_high" in report:
        report_lines.append(
            f"centroid_dist(low, high) on PCA: {report['centroid_dist_low_high']:.3f}"
        )
        report_lines.append(
            "Gợi ý đọc: dist càng lớn → embedding càng tách điểm thấp/cao (sanity tốt)."
        )

    report_path = os.path.join(args.out_dir, f"embeddings_sanity_{args.score_col}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))
    print(f"\nSaved:\n  {emb_path}\n  {meta_path}\n  {pca_png}\n  {tsne_png}\n  {coords_path}\n  {report_path}")


if __name__ == "__main__":
    main()
