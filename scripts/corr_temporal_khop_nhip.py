"""
Person 2 — Pearson/Spearman giữa output Temporal/DTW và cột `khop_nhip`.

Granularity: **per-video**. Correlation ở cấp cửa sổ sẽ phồng n lên ~6× với các
mẫu không độc lập (cửa sổ cùng một video) → p-value vô nghĩa. Không làm.

Regression test của cả pipeline (phải ra đúng, nếu lệch là join hoặc thứ tự cột
dtw_seg_* đang sai — dừng và sửa trước khi đọc số của model):
    dtw_distance_total × khop_nhip @ split=all, n=217
      Pearson  = -0.4419      Spearman = -0.3764

Cột `in_sample`: True cho mọi đại lượng dẫn từ embedding trên split `train`/`all`
— encoder đã thấy nhãn của những video đó, nên **chỉ dòng in_sample=False được
dùng làm bằng chứng**.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Console Windows mặc định cp1252 → print() tiếng Việt sẽ UnicodeEncodeError.
# Ép UTF-8 để chạy được bằng `python -m ...` mà không cần set PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from src.temporal_dl.dataset import build_temporal_sequence, sorted_seg_columns
from src.temporal_dl.infer import encode_video, load_temporal_checkpoint

VideoKey = Tuple[str, str]
FIELDNAMES = [
    "quantity", "target", "granularity", "split", "n",
    "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
    "fdr_q", "in_sample", "note",
]


def _read_keys(path: str) -> List[VideoKey]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().strip('"').lstrip("﻿") for c in df.columns]
    return [(str(d), str(v)) for d, v in zip(df["dance_id"], df["video_id"])]


def _corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {
            "n": len(x), "pearson_r": float("nan"), "pearson_p": float("nan"),
            "spearman_rho": float("nan"), "spearman_p": float("nan"),
        }
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
        "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
    }


def _seg_stats(row: pd.Series) -> Dict[str, float]:
    seg_cols = sorted_seg_columns(row)
    segs = pd.to_numeric(row[seg_cols], errors="coerce").to_numpy(dtype=np.float64)
    segs = segs[np.isfinite(segs)]
    flag_cols = sorted_seg_columns(row, prefix="flag_seg_")
    flags = pd.to_numeric(row[flag_cols], errors="coerce").to_numpy(dtype=np.float64)
    flags = flags[np.isfinite(flags)]
    n_seg = max(int(segs.size), 1)
    n_flagged = float(flags.sum()) if flags.size else 0.0
    return {
        "dtw_seg_mean": float(segs.mean()) if segs.size else np.nan,
        "dtw_seg_std": float(segs.std()) if segs.size else np.nan,
        "dtw_seg_max": float(segs.max()) if segs.size else np.nan,
        "dtw_seg_min": float(segs.min()) if segs.size else np.nan,
        "n_segments": float(n_seg),
        "n_flagged": n_flagged,
        "flag_ratio": n_flagged / n_seg,
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Person 2 — correlation với khop_nhip")
    ap.add_argument("--scores", default="annotations/scores.csv")
    ap.add_argument("--dtw", default="annotations/dtw_features.csv")
    ap.add_argument("--poses", default="poses")
    ap.add_argument("--ckpt", default=None, help="Encoder 17-ch (kịch bản (c))")
    ap.add_argument("--ckpt-nodtw", default=None, help="Encoder 16-ch (kịch bản (b))")
    ap.add_argument("--target", default="khop_nhip")
    ap.add_argument("--split", default="official", choices=["official"])
    ap.add_argument("--train-csv", default="annotations/scores_train.csv")
    ap.add_argument("--val-csv", default="annotations/scores_test.csv")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--hop", type=int, default=30)
    ap.add_argument("--aggregate", default="mean_std",
                    choices=["mean", "mean_std", "mean_std_minmax"])
    ap.add_argument("--fdr", action="store_true",
                    help="Benjamini-Hochberg cho họ test per-dimension")
    ap.add_argument("--allow-leakage", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-emb", default=None)
    ap.add_argument("--out-2d", default=None)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    out = args.out or f"experiments/temporal_dl/correlation_{args.target}.csv"
    out_emb = args.out_emb or "experiments/temporal_dl/temporal_embeddings_meta.csv"
    out_2d = args.out_2d or "experiments/temporal_dl/temporal_embeddings_2d.csv"

    # ---- Dữ liệu ----
    scores = pd.read_csv(args.scores)
    scores.columns = [str(c).strip().strip('"').lstrip("﻿") for c in scores.columns]
    dtw = pd.read_csv(args.dtw)
    df = scores.merge(dtw, on=["dance_id", "video_id"], how="inner").reset_index(drop=True)
    print(f"Inner-join scores × dtw: {len(df)} video")
    if args.target not in df.columns:
        raise SystemExit(f"Thiếu cột target {args.target}")

    train_keys = set(_read_keys(args.train_csv))
    val_keys = set(_read_keys(args.val_csv))
    keys = [(str(d), str(v)) for d, v in zip(df["dance_id"], df["video_id"])]
    split_of = {
        k: ("train" if k in train_keys else "val" if k in val_keys else "")
        for k in keys
    }

    # ---- Encoder ----
    enc_c = enc_b = None
    cfg_c: Dict[str, Any] = {}
    if args.ckpt:
        enc_c, ck_c = load_temporal_checkpoint(args.ckpt)
        cfg_c = ck_c.get("config", {})
        bad = set(map(tuple, cfg_c.get("train_videos", []))) & val_keys
        if "train_videos" not in cfg_c:
            raise SystemExit(
                "--ckpt: checkpoint thiếu config.train_videos → không xác minh "
                "được leakage. Retrain bằng src/temporal_dl/train.py mới."
            )
        if bad and not args.allow_leakage:
            raise SystemExit(
                f"LEAKAGE: {len(bad)} val video nằm trong tập train của encoder. "
                f"Dùng --allow-leakage nếu cố ý."
            )
    else:
        print("WARNING: không có --ckpt → chỉ tính correlation cho DTW/sub-score")
    if args.ckpt_nodtw:
        enc_b, _ = load_temporal_checkpoint(args.ckpt_nodtw)

    # ---- Trích đại lượng per-video ----
    seg_cols = sorted_seg_columns(dtw)
    target_range = cfg_c.get("target_range")
    target_range = (float(target_range[0]), float(target_range[1])) if target_range else None

    recs: List[Dict[str, Any]] = []
    emb_c_list: List[np.ndarray] = []
    emb_b_list: List[np.ndarray] = []
    for i, r in df.iterrows():
        dance_id, video_id = str(r["dance_id"]), str(r["video_id"])
        rec: Dict[str, Any] = {
            "dance_id": dance_id, "video_id": video_id,
            "person_id": str(r.get("person_id", video_id)),
            "split": split_of[(dance_id, video_id)],
            "khop_dong_tac": float(r.get("khop_dong_tac", np.nan)),
            "khop_nhip": float(r.get("khop_nhip", np.nan)),
            "nang_luong": float(r.get("nang_luong", np.nan)),
            "tong_diem": float(r.get("tong_diem", np.nan)),
            "dtw_distance_total": float(r.get("dtw_distance_total", np.nan)),
            **_seg_stats(r),
        }
        if enc_c is not None or enc_b is not None:
            seq17 = build_temporal_sequence(
                dance_id, video_id, r, poses_dir=args.poses,
                use_dtw=True, seg_cols=seg_cols,
            )
            if seq17 is None:
                print(f"  bỏ {dance_id}/{video_id} — thiếu pose")
                continue
            if enc_c is not None:
                e, sc = encode_video(
                    seq17, enc_c, window_frames=args.window, hop_frames=args.hop,
                    aggregate=args.aggregate, return_scores=True,
                )
                emb_c_list.append(e)
                rec["n_windows"] = int(len(sc))
                rec["temporal_emb_l2norm"] = float(np.linalg.norm(e))
                pred = float(np.mean(sc))
                rec["temporal_pred_score"] = (
                    pred * (target_range[1] - target_range[0]) + target_range[0]
                    if target_range else pred
                )
            if enc_b is not None:
                emb_b_list.append(
                    encode_video(
                        seq17[:, :-1], enc_b, window_frames=args.window,
                        hop_frames=args.hop, aggregate=args.aggregate,
                    )
                )
        recs.append(rec)

    meta = pd.DataFrame(recs)
    y_all = meta[args.target].to_numpy(dtype=np.float64)

    # ---- PC1 (PCA fit CHỈ trên hàng train) ----
    emb_c = np.stack(emb_c_list) if emb_c_list else None
    emb_b = np.stack(emb_b_list) if emb_b_list else None
    tr_mask = (meta["split"] == "train").to_numpy()
    if emb_c is not None:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        sc_ = StandardScaler().fit(emb_c[tr_mask])
        pca = PCA(n_components=2, random_state=42).fit(sc_.transform(emb_c[tr_mask]))
        pcs = pca.transform(sc_.transform(emb_c))
        meta["temporal_emb_pc1"] = pcs[:, 0]
        meta["temporal_emb_pc2"] = pcs[:, 1]
    if emb_b is not None:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        sc2 = StandardScaler().fit(emb_b[tr_mask])
        p2 = PCA(n_components=1, random_state=42).fit(sc2.transform(emb_b[tr_mask]))
        meta["temporal_emb_pc1_nodtw"] = p2.transform(sc2.transform(emb_b))[:, 0]

    # ---- Bảng correlation ----
    scalar_quantities = [
        ("dtw_distance_total", "DTW tổng — 'dtw score' của spec", False),
        ("dtw_seg_mean", "", False),
        ("dtw_seg_std", "", False),
        ("dtw_seg_max", "", False),
        ("dtw_seg_min", "", False),
        ("flag_ratio", "validate flag_suspicious_segments (Tuần 3)", False),
        ("temporal_pred_score", "prediction của Temporal DL (đã de-normalize)", True),
        ("temporal_emb_pc1", "PC1 của embedding mean⊕std (PCA fit trên train)", True),
        ("temporal_emb_pc1_nodtw", "PC1 embedding encoder 16-ch", True),
        ("temporal_emb_l2norm", "phát hiện encoder suy biến", True),
        ("khop_dong_tac", "sub-score khác — bối cảnh cho caveat nguồn gốc nhãn", False),
        ("nang_luong", "sub-score khác", False),
        ("tong_diem", "tổng 0-300", False),
    ]
    splits = [("all", np.ones(len(meta), bool)),
              ("train", tr_mask),
              ("val", (meta["split"] == "val").to_numpy())]

    out_rows: List[Dict[str, Any]] = []
    for qty, note, from_emb in scalar_quantities:
        if qty not in meta.columns:
            continue
        x_all = meta[qty].to_numpy(dtype=np.float64)
        for sname, mask in splits:
            if mask.sum() < 3:
                continue
            c = _corr(x_all[mask], y_all[mask])
            out_rows.append(
                {
                    "quantity": qty, "target": args.target, "granularity": "video",
                    "split": sname, **c, "fdr_q": "",
                    # embedding đã fit vào nhãn của hàng train → in-sample
                    "in_sample": bool(from_emb and sname in ("train", "all")),
                    "note": note,
                }
            )

    # ---- Độ nhạy: bỏ video self-compare (dtw_distance_total == 0) ----
    # Video reference tự so với chính nó → dtw=0 và điểm tuyệt đối. Một điểm như
    # vậy kéo Pearson rất mạnh (Spearman thì gần như không đổi vì dựa trên hạng),
    # nên phải báo cả hai con số, không chọn con số đẹp hơn.
    dtw_tot = meta["dtw_distance_total"].to_numpy(dtype=np.float64)
    self_cmp = dtw_tot <= 1e-9
    if self_cmp.any():
        print(
            f"\n⚠️  {int(self_cmp.sum())} video có dtw_distance_total == 0 "
            f"(self-compare): {meta.loc[self_cmp, 'video_id'].tolist()}"
        )
        for sname, mask in splits:
            m2 = mask & ~self_cmp
            if m2.sum() < 3:
                continue
            c = _corr(dtw_tot[m2], y_all[m2])
            out_rows.append(
                {
                    "quantity": "dtw_distance_total (bỏ self-compare)",
                    "target": args.target, "granularity": "video", "split": sname,
                    **c, "fdr_q": "", "in_sample": False,
                    "note": "bỏ video reference tự so chính nó (dtw=0, điểm tuyệt đối)",
                }
            )

    # ---- Per-dimension + BH-FDR ----
    if emb_c is not None:
        val_mask = (meta["split"] == "val").to_numpy()
        for sname, mask, in_s in (("val", val_mask, False), ("train", tr_mask, True)):
            if mask.sum() < 3:
                continue
            dim_rows, ps = [], []
            for d in range(emb_c.shape[1]):
                c = _corr(emb_c[mask, d], y_all[mask])
                dim_rows.append(
                    {
                        "quantity": f"temporal_emb_dim_{d:03d}", "target": args.target,
                        "granularity": "video", "split": sname, **c,
                        "in_sample": in_s, "note": "",
                    }
                )
                ps.append(c["pearson_p"])
            qs = [""] * len(ps)
            if args.fdr:
                from scipy.stats import false_discovery_control

                arr = np.array(ps, dtype=float)
                ok = np.isfinite(arr)
                if ok.any():
                    q = np.full(arr.shape, np.nan)
                    q[ok] = false_discovery_control(arr[ok], method="bh")
                    qs = [f"{v:.6g}" if np.isfinite(v) else "" for v in q]
            for row, q in zip(dim_rows, qs):
                row["fdr_q"] = q
            out_rows.extend(dim_rows)
            n_sig = sum(
                1 for row, q in zip(dim_rows, qs)
                if q != "" and float(q) < 0.05
            )
            print(
                f"Per-dimension @ split={sname}: best |r| = "
                f"{max((abs(r['pearson_r']) for r in dim_rows if np.isfinite(r['pearson_r'])), default=float('nan')):.3f}"
                f" | {n_sig}/{emb_c.shape[1]} chiều qua được q<0.05 (BH-FDR)"
                + ("" if args.fdr else "  [chưa bật --fdr]")
            )

    # ---- In các dòng chính ----
    print(f"\n{'quantity':<26} {'split':>6} {'n':>5} {'pearson':>9} {'spearman':>9} {'in_sample':>10}")
    print("-" * 72)
    for r in out_rows:
        if r["quantity"].startswith("temporal_emb_dim_"):
            continue
        print(
            f"{r['quantity']:<26} {r['split']:>6} {r['n']:5d} "
            f"{r['pearson_r']:9.4f} {r['spearman_rho']:9.4f} {str(r['in_sample']):>10}"
        )

    ref = next(
        (r for r in out_rows
         if r["quantity"] == "dtw_distance_total" and r["split"] == "all"), None,
    )
    if ref:
        ok = abs(ref["pearson_r"] + 0.4419) < 5e-4 and abs(ref["spearman_rho"] + 0.3764) < 5e-4
        print(
            f"\nRegression test (dtw_distance_total @ all, n={ref['n']}): "
            f"r={ref['pearson_r']:.4f} ρ={ref['spearman_rho']:.4f} → "
            f"{'✅ KHỚP (-0.4419 / -0.3764)' if ok else '❌ LỆCH kỳ vọng -0.4419 / -0.3764 — kiểm tra join & thứ tự cột dtw_seg_*'}"
        )

    # ---- Ghi file ----
    for p in (out, out_emb, out_2d):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    pd.DataFrame(out_rows).reindex(columns=FIELDNAMES).to_csv(out, index=False)
    print(f"\nĐã ghi: {out}")

    if emb_c is not None:
        emb_df = pd.DataFrame(
            emb_c, columns=[f"emb_{i:03d}" for i in range(emb_c.shape[1])]
        )
        full = pd.concat([meta.reset_index(drop=True), emb_df], axis=1)
        full["window_frames"] = args.window
        full["hop_frames"] = args.hop
        full["aggregate"] = args.aggregate
        full["emb_dim"] = emb_c.shape[1]
        full["ckpt_target"] = cfg_c.get("target_col", "")
        full["use_dtw"] = cfg_c.get("use_dtw", "")
        full.to_csv(out_emb, index=False)
        print(f"Đã ghi: {out_emb}")

        meta[["dance_id", "video_id", "person_id", args.target, "split",
              "temporal_emb_pc1", "temporal_emb_pc2"]].rename(
            columns={"temporal_emb_pc1": "pc1", "temporal_emb_pc2": "pc2"}
        ).to_csv(out_2d, index=False)
        print(f"Đã ghi: {out_2d}")
    else:
        meta.to_csv(out_emb, index=False)
        print(f"Đã ghi: {out_emb}  (chưa có embedding — thiếu --ckpt)")


if __name__ == "__main__":
    main()
