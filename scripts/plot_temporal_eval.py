"""
Person 2 — sinh biểu đồ từ các CSV đã xuất.

Chỉ đọc CSV, KHÔNG load checkpoint — để người review vẽ lại được hình mà không
cần file `*.pth` (bị .gitignore loại). Thiếu CSV nào thì bỏ hình đó và báo, không
crash.

Output → experiments/temporal_dl/plots/
  01_train_curve.png     val_mae_video_raw theo epoch + 2 đường baseline
  02_ablation_mae.png    MAE theo từng kịch bản + 2 đường baseline
  03_dtw_vs_khopnhip.png scatter dtw_distance_total × khop_nhip + đường hồi quy
  04_cv_fold.png         MAE từng fold + mean ± std (GroupKFold)
  05_embedding_pca.png   PCA 2D của embedding, tô màu theo khop_nhip

Màu: palette tham chiếu của skill dataviz (slot 1 blue, slot 2 orange — cặp đã
validate: CVD ΔE 9.2 > ngưỡng 8). Ramp sequential = blue 100→700.
Hình dùng cho báo cáo in nên chỉ làm light mode.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Console Windows mặc định cp1252 → print() tiếng Việt sẽ UnicodeEncodeError.
# Ép UTF-8 để chạy được bằng `python -m ...` mà không cần set PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ── Palette (dataviz reference instance, light mode) ──────────────────────────
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
# Sequential blue ramp (steps 100 → 700)
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
CMAP_BLUE = LinearSegmentedColormap.from_list("ds_blue", BLUE_RAMP)

# Baseline đã precompute trên split official 183/34 (xem docs/temporal_eval.md)
BASE_GLOBAL_MEAN = 2.850
BASE_DANCE_MEAN = 1.902

EXP = ROOT / "experiments" / "temporal_dl"
PLOTS = EXP / "plots"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def _despine(ax, keep=("left", "bottom")) -> None:
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


def _title(ax, title: str, subtitle: str = "") -> None:
    """
    Tiêu đề đậm + phụ đề mực thứ cấp, nhẹ hơn.

    Dùng text riêng thay vì nhét 2 dòng vào set_title() — nếu không, phụ đề
    thừa hưởng luôn weight của tiêu đề và trông nặng ngang nhau.
    """
    n_lines = subtitle.count("\n") + 1 if subtitle else 0
    # Phụ đề neo đáy ở +8 và nở lên trên, nên title phải lùi đủ cho mọi dòng
    ax.set_title(title, pad=(12 + 13 * n_lines) if subtitle else 10)
    if subtitle:
        ax.annotate(
            subtitle, xy=(0, 1.0), xycoords="axes fraction",
            xytext=(0, 8), textcoords="offset points",
            fontsize=9.5, color=INK_SECONDARY, va="bottom", ha="left",
            linespacing=1.4,
        )


def _save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã ghi: {path}")


def _baseline_lines(ax, horizontal: bool = True) -> None:
    """2 đường baseline + direct label. Không có chúng thì MAE vô nghĩa."""
    draw = ax.axhline if horizontal else ax.axvline
    for val, label in (
        (BASE_DANCE_MEAN, f"dance-mean {BASE_DANCE_MEAN:.2f}"),
        (BASE_GLOBAL_MEAN, f"global-mean {BASE_GLOBAL_MEAN:.2f}"),
    ):
        draw(val, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
        if horizontal:
            ax.annotate(
                label, xy=(1.0, val), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points",
                color=INK_MUTED, fontsize=8, va="center",
            )
        else:
            ax.annotate(
                label, xy=(val, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 4), textcoords="offset points",
                color=INK_MUTED, fontsize=8, ha="center", rotation=0,
            )


# ──────────────────────────────────────────────────────────────────────────────
# 01 — Train curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_train_curve(tag_dtw: str, tag_nodtw: str) -> bool:
    runs = [
        (EXP / f"train_history_{tag_dtw}.csv", "Encoder (c) 17-ch, có kênh DTW", SERIES_1),
        (EXP / f"train_history_{tag_nodtw}.csv", "Encoder (b) 16-ch, bỏ kênh DTW", SERIES_2),
    ]
    found = [(p, lbl, c) for p, lbl, c in runs if p.is_file()]
    if not found:
        print(f"bỏ 01_train_curve — không thấy {runs[0][0].name}")
        return False

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for path, label, color in found:
        h = pd.read_csv(path)
        col = "val_mae_video_raw" if "val_mae_video_raw" in h.columns else "val_mae"
        y = pd.to_numeric(h[col], errors="coerce")
        ax.plot(h["epoch"], y, color=color, label=label, zorder=3)
        best = int(y.idxmin())
        ax.plot(
            h["epoch"][best], y[best], "o", color=color, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
        )
        ax.annotate(
            f"{y[best]:.2f}", xy=(h["epoch"][best], y[best]),
            xytext=(6, -11), textcoords="offset points",
            color=INK_SECONDARY, fontsize=9, fontweight="bold",
        )

    _baseline_lines(ax, horizontal=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE cấp video (điểm khop_nhip)")
    _title(
        ax,
        "MAE trên tập val theo epoch, so với 2 baseline tầm thường",
        "Thấp hơn là tốt hơn. Nằm trên đường dance-mean nghĩa là chưa thắng "
        "được dự đoán hằng số.",
    )
    # Nhãn 2 đường baseline nằm ở mép phải → legend phải tránh góc trên phải
    ax.legend(loc="lower left", fontsize=9, labelcolor=INK_SECONDARY)
    _despine(ax)
    _save(fig, "01_train_curve.png")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 02 — Ablation
# ──────────────────────────────────────────────────────────────────────────────

def plot_ablation(csv: Path) -> bool:
    if not csv.is_file():
        print(f"bỏ 02_ablation_mae — không thấy {csv.name}")
        return False
    df = pd.read_csv(csv)
    df = df[~df["setup"].str.startswith("Baseline global")]
    df = df[~df["setup"].str.startswith("Baseline dance")]
    if df.empty:
        print("bỏ 02_ablation_mae — bảng chỉ có baseline")
        return False
    df = df.sort_values("MAE", ascending=False).reset_index(drop=True)

    _style()
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(df) + 2.0))
    ypos = np.arange(len(df))
    # Highlight 3 kịch bản bắt buộc của spec
    is_spec = df["setup"].str.match(r"^\([abc]\) ")
    colors = [SERIES_1 if s else BASELINE for s in is_spec]
    ax.barh(ypos, df["MAE"], height=0.68, color=colors, zorder=3)
    for i, (v, spec) in enumerate(zip(df["MAE"], is_spec)):
        ax.annotate(
            f"{v:.3f}", xy=(v, i), xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK_PRIMARY if spec else INK_SECONDARY,
            fontweight="bold" if spec else "normal",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["setup"], fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("MAE (điểm khop_nhip) — thấp hơn là tốt hơn")
    ax.set_xlim(0, max(df["MAE"].max(), BASE_GLOBAL_MEAN) * 1.18)
    _title(
        ax,
        "Ablation: DTW vs LSTM trên target khop_nhip",
        "Xanh = 3 kịch bản bắt buộc của spec (a)/(b)/(c). "
        "Bên phải đường dance-mean là chưa thắng baseline.",
    )
    ax.grid(axis="y", visible=False)
    _baseline_lines(ax, horizontal=False)
    _despine(ax)
    _save(fig, "02_ablation_mae.png")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 03 — DTW vs khop_nhip
# ──────────────────────────────────────────────────────────────────────────────

def plot_dtw_scatter(scores_csv: Path, dtw_csv: Path, target: str) -> bool:
    if not (scores_csv.is_file() and dtw_csv.is_file()):
        print("bỏ 03_dtw_vs_khopnhip — thiếu scores.csv hoặc dtw_features.csv")
        return False
    sc = pd.read_csv(scores_csv)
    sc.columns = [str(c).strip().strip('"').lstrip("﻿") for c in sc.columns]
    df = sc.merge(pd.read_csv(dtw_csv), on=["dance_id", "video_id"], how="inner")
    x = df["dtw_distance_total"].to_numpy(float)
    y = df[target].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    from scipy.stats import pearsonr, spearmanr
    r, rp = pearsonr(x, y)
    rho, sp = spearmanr(x, y)

    # train/val để người đọc thấy tập test nằm đâu
    split = np.full(len(df), "train", dtype=object)
    val_csv = scores_csv.parent / "scores_test.csv"
    if val_csv.is_file():
        vk = set(
            zip(pd.read_csv(val_csv)["dance_id"].astype(str),
                pd.read_csv(val_csv)["video_id"].astype(str))
        )
        split = np.array(
            ["val" if (str(a), str(b)) in vk else "train"
             for a, b in zip(df["dance_id"], df["video_id"])], dtype=object,
        )
    split = split[ok]

    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, color, label in (
        ("train", SERIES_1, "train (183)"),
        ("val", SERIES_2, "val / held-out (34)"),
    ):
        m = split == name
        if m.sum():
            ax.plot(
                x[m], y[m], "o", color=color, markersize=7, alpha=0.75,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                linestyle="none", label=f"{label}", zorder=3,
            )
    # Đường hồi quy trên toàn bộ n
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b * xs + a, color=INK_SECONDARY, linewidth=1.6, zorder=4)

    # Self-compare: video chính là người nhảy reference → dtw=0, điểm tuyệt đối.
    # Một điểm này kéo Pearson từ -0.33 xuống -0.44, nên phải chỉ ra, không giấu.
    self_cmp = x <= 1e-9
    r_wo = rho_wo = float("nan")
    if self_cmp.any():
        r_wo, _ = pearsonr(x[~self_cmp], y[~self_cmp])
        rho_wo, _ = spearmanr(x[~self_cmp], y[~self_cmp])
        ax.annotate(
            "self-compare (dtw=0)",
            xy=(x[self_cmp][0], y[self_cmp][0]),
            xytext=(26, -6), textcoords="offset points",
            fontsize=8.5, color=INK_MUTED, va="center",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1),
        )

    ax.set_xlabel("dtw_distance_total")
    ax.set_ylabel(f"{target} (điểm 0–100)")
    sub = (
        f"n={len(x)}   Pearson r={r:.4f} (p={rp:.1e})   Spearman ρ={rho:.4f} (p={sp:.1e})"
    )
    if self_cmp.any():
        sub += f"\nBỏ {int(self_cmp.sum())} video self-compare: r={r_wo:.4f}, ρ={rho_wo:.4f}"
    _title(ax, f"DTW tổng × {target} — dấu đúng: DTW lớn ⇒ nhịp tệ hơn", sub)
    ax.legend(loc="upper right", fontsize=9, labelcolor=INK_SECONDARY)
    _despine(ax)
    _save(fig, "03_dtw_vs_khopnhip.png")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 04 — CV folds
# ──────────────────────────────────────────────────────────────────────────────

def plot_cv(csv: Path) -> bool:
    """
    Dot plot mean ± std, KHÔNG phải boxplot: với 5 fold thì tứ phân vị của
    boxplot chỉ là nhiễu. Mục đích hình này là cho thấy chênh lệch giữa các
    kịch bản nằm trong khoảng dao động giữa các fold.
    """
    if not csv.is_file():
        print(f"bỏ 04_cv_fold — không thấy {csv.name}")
        return False
    df = pd.read_csv(csv).sort_values("MAE_mean", ascending=False).reset_index(drop=True)

    _style()
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(df) + 2.0))
    ypos = np.arange(len(df))
    is_spec = df["setup"].str.match(r"^\([abc]\) ")
    for i, (_, r) in enumerate(df.iterrows()):
        color = SERIES_1 if is_spec[i] else INK_MUTED
        ax.errorbar(
            r["MAE_mean"], i, xerr=r["MAE_std"], fmt="o", color=color,
            markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
            elinewidth=2, capsize=4, zorder=3,
        )
        ax.annotate(
            f"{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}",
            xy=(r["MAE_mean"] + r["MAE_std"], i), xytext=(7, 0),
            textcoords="offset points", va="center", fontsize=9,
            color=INK_PRIMARY if is_spec[i] else INK_SECONDARY,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["setup"], fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel(f"MAE qua {int(df['folds'].max())} fold (mean ± std), điểm khop_nhip")
    _title(
        ax,
        "GroupKFold theo person_id — chênh lệch giữa các kịch bản",
        "Thanh lỗi phủ nhau ⇒ chưa phân định được kịch bản nào tốt hơn.",
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
    _despine(ax)
    _save(fig, "04_cv_fold.png")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 05 — Embedding PCA
# ──────────────────────────────────────────────────────────────────────────────

def plot_embedding_pca(csv: Path, target: str) -> bool:
    if not csv.is_file():
        print(f"bỏ 05_embedding_pca — không thấy {csv.name}")
        return False
    df = pd.read_csv(csv)
    if not {"pc1", "pc2", target} <= set(df.columns):
        print(f"bỏ 05_embedding_pca — {csv.name} thiếu pc1/pc2/{target}")
        return False

    _style()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    # Video self-compare được chấm 100 điểm trong khi cả bộ chỉ trải 63-84.
    # Để thang màu chạy hết [63, 100] thì toàn bộ dữ liệu thật dồn vào ~25%
    # nhạt nhất và gradient biến mất → clip theo percentile 2-98.
    vals = df[target].to_numpy(dtype=float)
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    n_clip = int(((vals < vmin) | (vals > vmax)).sum())
    sc_ = ax.scatter(
        df["pc1"], df["pc2"], c=vals, cmap=CMAP_BLUE, s=58,
        vmin=vmin, vmax=vmax,
        edgecolors=SURFACE, linewidths=1.2, zorder=3,
    )
    cb = fig.colorbar(sc_, ax=ax, pad=0.02, extend="both" if n_clip else "neither")
    cb.set_label(f"{target} (điểm)", color=INK_SECONDARY, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=INK_MUTED, labelcolor=INK_MUTED, labelsize=8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _title(
        ax,
        f"PCA 2D của temporal embedding, tô màu theo {target}",
        "Sanity check định tính: điểm cao/thấp có tách cụm không. "
        "PCA fit chỉ trên hàng train.\n"
        f"Thang màu cắt ở percentile 2-98 ({vmin:.0f}-{vmax:.0f} điểm)"
        + (f", {n_clip} điểm ngoài khoảng." if n_clip else "."),
    )
    _despine(ax)
    _save(fig, "05_embedding_pca.png")
    return True


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Person 2 — sinh biểu đồ từ CSV")
    ap.add_argument("--target", default="khop_nhip")
    ap.add_argument("--tag-dtw", default="khopnhip_dtw_official")
    ap.add_argument("--tag-nodtw", default="khopnhip_nodtw_official")
    ap.add_argument("--scores", default=str(ROOT / "annotations" / "scores.csv"))
    ap.add_argument("--dtw", default=str(ROOT / "annotations" / "dtw_features.csv"))
    ap.add_argument("--plots-dir", default=None)
    args = ap.parse_args()

    global PLOTS
    if args.plots_dir:
        PLOTS = Path(args.plots_dir)

    done = sum(
        [
            plot_train_curve(args.tag_dtw, args.tag_nodtw),
            plot_ablation(EXP / f"ablation_xgb_{args.target}.csv"),
            plot_dtw_scatter(Path(args.scores), Path(args.dtw), args.target),
            plot_cv(EXP / f"ablation_xgb_{args.target}_cv.csv"),
            plot_embedding_pca(EXP / "temporal_embeddings_2d.csv", args.target),
        ]
    )
    print(f"\n{done}/5 hình đã sinh → {PLOTS}")


if __name__ == "__main__":
    main()
