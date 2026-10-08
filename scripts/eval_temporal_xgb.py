"""
Tuần 5 Person 2 — Ablation 3 kịch bản bắt buộc qua XGBoost tạm:

  (a) DTW-only    — chỉ dtw_distance features, KHÔNG dùng DL
  (b) LSTM-only   — bỏ kênh dtw, chỉ raw sequence (encoder input_dim=16)
  (c) DTW+LSTM    — kết hợp (encoder input_dim=17)

Kèm các baseline tầm thường — BẮT BUỘC đọc trước mọi số của model, vì `khop_nhip`
có std chỉ ~4.1 điểm nên MAE tuyệt đối gần như vô nghĩa nếu không so với chúng:
  global-mean  ≈ MAE 2.850    dance-mean ≈ MAE 1.902  (split official 183/34)

`_fit_xgb`, `_dtw_row_features`, `_geo_summary` copy nguyên văn từ
`scripts/eval_spatial_xgb.py` (Person 1) để hyperparameter XGB y hệt → hai bảng
so sánh được với nhau. Chỉ *đọc* output của Person 1/3, không sửa code của họ.

QUY TẮC CHỐNG LEAKAGE ENCODER (xem docs/temporal_week5.md):
  Embedding là một hàm đã fit vào nhãn. Nếu một video từng nằm trong tập train
  của encoder thì vector của nó đã hấp thụ nhãn của chính nó. Vậy tập video
  train của encoder phải là TẬP CON của hàng train XGBoost. Script assert điều
  này từ `config.train_videos` trong checkpoint và dừng nếu vi phạm.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    "setup", "target", "split", "n_train", "n_val",
    "n_features", "MAE", "RMSE", "R2", "Pearson",
]


# ──────────────────────────────────────────────────────────────────────────────
# Copy nguyên văn từ scripts/eval_spatial_xgb.py (Person 1) — giữ y hệt để
# bảng của Person 1 và Person 2 so sánh được.
# ──────────────────────────────────────────────────────────────────────────────

def _geo_summary(pose: np.ndarray) -> Optional[np.ndarray]:
    """Copy từ scripts/eval_spatial_xgb.py::_geo_summary."""
    try:
        from src.features.geometry import compute_geometry_features
    except Exception:
        return None
    feat = compute_geometry_features(pose)  # (T, 8)
    return np.concatenate(
        [feat.mean(axis=0), feat.std(axis=0), feat.min(axis=0), feat.max(axis=0)]
    ).astype(np.float32)


def _dtw_row_features(row: pd.Series) -> np.ndarray:
    """Copy từ scripts/eval_spatial_xgb.py::_dtw_row_features."""
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
    """Copy từ scripts/eval_spatial_xgb.py::_fit_xgb."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("Cần xgboost trong requirements.txt") from exc

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
    return {**_metrics(y_val, pred), "n_features": int(n_feat), "_pred": pred}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers riêng của Person 2
# ──────────────────────────────────────────────────────────────────────────────

def _safe_nanmean(values: Sequence[float]) -> float:
    """nanmean nhưng trả nan lặng lẽ khi mọi phần tử là nan (baseline hằng số)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    pearson = float("nan")
    if len(y_true) > 2 and np.std(y_true) > 1e-8 and np.std(pred) > 1e-8:
        # Bộ dự đoán hằng số (baseline) làm corrcoef chia cho 0 → tắt warning,
        # giá trị nan trả về là đúng ngữ nghĩa ("không định nghĩa được").
        with np.errstate(invalid="ignore", divide="ignore"):
            pearson = float(np.corrcoef(y_true, pred)[0, 1])
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": float(r2_score(y_true, pred)),
        "Pearson": pearson,
    }


def _extra_dtw_features(row: pd.Series, n_frames: int) -> np.ndarray:
    """(a+) — seg_min, n_flagged, flag_ratio, T. Phần thêm so với _dtw_row_features."""
    seg_cols = sorted_seg_columns(row)
    segs = pd.to_numeric(row[seg_cols], errors="coerce").to_numpy(dtype=np.float64)
    segs = segs[np.isfinite(segs)]
    flag_cols = sorted_seg_columns(row, prefix="flag_seg_")
    flags = pd.to_numeric(row[flag_cols], errors="coerce").to_numpy(dtype=np.float64)
    flags = flags[np.isfinite(flags)]
    n_seg = max(int(segs.size), 1)
    n_flagged = float(flags.sum()) if flags.size else 0.0
    return np.array(
        [
            float(segs.min()) if segs.size else 0.0,
            n_flagged,
            n_flagged / n_seg,
            float(n_frames),
        ],
        dtype=np.float32,
    )


def _pose_path(poses_dir: str, dance_id: str, video_id: str) -> Optional[str]:
    for p in (
        os.path.join(poses_dir, dance_id, f"{video_id}.npy"),
        os.path.join(poses_dir, f"{dance_id}_{video_id}.npy"),
    ):
        if os.path.isfile(p):
            return p
    return None


def _assert_no_encoder_leakage(
    name: str, cfg: Dict[str, Any], val_keys: Sequence[VideoKey],
    target: str, expect_use_dtw: bool, allow: bool,
) -> None:
    """
    Chặn leakage encoder. Checkpoint Tuần 4 không có `config.train_videos` nên
    sẽ bị từ chối — đúng ý: nó được train bằng random_split cấp window.
    """
    if "train_videos" not in cfg:
        raise SystemExit(
            f"{name}: checkpoint thiếu config.train_videos → không xác minh được "
            f"leakage. Retrain bằng src/temporal_dl/train.py mới."
        )
    bad = set(map(tuple, cfg["train_videos"])) & set(val_keys)
    if bad and not allow:
        raise SystemExit(
            f"LEAKAGE: {len(bad)} val video nằm trong tập train của encoder "
            f"{name}: {sorted(bad)[:5]}{'...' if len(bad) > 5 else ''}\n"
            f"Dùng --allow-leakage nếu cố ý chạy bản chẩn đoán."
        )
    if cfg.get("target_col") != target:
        raise SystemExit(
            f"{name}: encoder train trên target {cfg.get('target_col')!r} nhưng "
            f"đang eval target {target!r}"
        )
    if bool(cfg.get("use_dtw")) is not expect_use_dtw:
        raise SystemExit(
            f"{name}: use_dtw={cfg.get('use_dtw')} nhưng cần {expect_use_dtw}"
        )


def _reduce(X: np.ndarray, train_idx: Sequence[int], pca_dim: int, seed: int) -> np.ndarray:
    """StandardScaler + PCA fit CHỈ trên train (guard chống leakage của Person 1)."""
    if pca_dim <= 0 or X.shape[1] <= pca_dim:
        return X
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_comp = min(pca_dim, len(train_idx) - 1, X.shape[1])
    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X)
    pca = PCA(n_components=n_comp, random_state=seed).fit(Xs[train_idx])
    return pca.transform(Xs).astype(np.float32)


def _read_keys(csv_path: str) -> List[VideoKey]:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().strip('"').lstrip("﻿") for c in df.columns]
    return [(str(d), str(v)) for d, v in zip(df["dance_id"], df["video_id"])]


# ──────────────────────────────────────────────────────────────────────────────
# Trích feature cho toàn bộ dataset
# ──────────────────────────────────────────────────────────────────────────────

def collect_features(args, enc_dtw, enc_nodtw) -> Dict[str, Any]:
    scores = pd.read_csv(args.scores)
    scores.columns = [str(c).strip().strip('"').lstrip("﻿") for c in scores.columns]
    dtw = pd.read_csv(args.dtw)
    df = scores.merge(dtw, on=["dance_id", "video_id"], how="inner")
    if args.target not in df.columns:
        raise SystemExit(f"Thiếu cột target {args.target}")
    print(f"Inner-join scores × dtw: {len(df)} video")
    if len(df) < 200:
        raise SystemExit(
            f"Chỉ {len(df)} video sau join — kỳ vọng 217. Kiểm tra dance_id/video_id."
        )

    seg_cols = sorted_seg_columns(dtw)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        dance_id, video_id = str(r["dance_id"]), str(r["video_id"])
        seq17 = build_temporal_sequence(
            dance_id, video_id, r, poses_dir=args.poses, use_dtw=True, seg_cols=seg_cols
        )
        if seq17 is None:
            print(f"  bỏ {dance_id}/{video_id} — thiếu pose")
            continue
        seq16 = seq17[:, :-1]      # bỏ kênh dtw — y hệt use_dtw=False

        ppath = _pose_path(args.poses, dance_id, video_id)
        geo = _geo_summary(np.load(ppath).astype(np.float32)) if ppath else None

        emb_c = (
            encode_video(seq17, enc_dtw, window_frames=args.window,
                         hop_frames=args.hop, aggregate=args.aggregate)
            if enc_dtw is not None else None
        )
        emb_b = (
            encode_video(seq16, enc_nodtw, window_frames=args.window,
                         hop_frames=args.hop, aggregate=args.aggregate)
            if enc_nodtw is not None else None
        )

        rows.append(
            {
                "key": (dance_id, video_id),
                "dance_id": dance_id,
                "video_id": video_id,
                "person_id": str(r.get("person_id", video_id)),
                "y": float(r[args.target]),
                "khop_dong_tac": float(r.get("khop_dong_tac", np.nan)),
                "dtw": _dtw_row_features(r),
                "dtw_extra": _extra_dtw_features(r, seq17.shape[0]),
                "geo": geo,
                "emb_c": emb_c,
                "emb_b": emb_b,
            }
        )

    if len(rows) < 10:
        raise SystemExit(f"Quá ít mẫu hợp lệ: {len(rows)}")
    return {"rows": rows}


def build_setups(rows: List[Dict[str, Any]], with_dance: bool) -> Dict[str, np.ndarray]:
    """Ma trận feature cho từng kịch bản. Baseline không feature xử lý riêng."""
    X_dtw = np.stack([r["dtw"] for r in rows])
    X_dtw_ext = np.concatenate(
        [X_dtw, np.stack([r["dtw_extra"] for r in rows])], axis=1
    )
    setups: Dict[str, np.ndarray] = {}
    setups["Baseline khop_dong_tac (oracle)"] = np.stack(
        [[r["khop_dong_tac"]] for r in rows]
    ).astype(np.float32)
    setups["(a) DTW-only"] = X_dtw
    setups["(a+) DTW-only extended"] = X_dtw_ext

    has_b = all(r["emb_b"] is not None for r in rows)
    has_c = all(r["emb_c"] is not None for r in rows)
    if has_b:
        X_b = np.stack([r["emb_b"] for r in rows])
        setups["(b) LSTM-only"] = X_b
        setups["(c-) DTW+LSTM, no-dtw encoder"] = np.concatenate([X_dtw, X_b], axis=1)
    if has_c:
        X_c = np.stack([r["emb_c"] for r in rows])
        setups["(c) DTW+LSTM"] = np.concatenate([X_dtw, X_c], axis=1)

    if all(r["geo"] is not None for r in rows):
        setups["Geometry-only (no DL)"] = np.stack([r["geo"] for r in rows])

    if with_dance:
        dances = sorted({r["dance_id"] for r in rows})
        d_idx = {d: i for i, d in enumerate(dances)}
        onehot = np.zeros((len(rows), len(dances)), dtype=np.float32)
        for i, r in enumerate(rows):
            onehot[i, d_idx[r["dance_id"]]] = 1.0
        for name in ["(a) DTW-only", "(a+) DTW-only extended",
                     "(b) LSTM-only", "(c) DTW+LSTM"]:
            if name in setups:
                setups[f"{name} +dance"] = np.concatenate([setups[name], onehot], axis=1)
    return setups


def run_split(
    rows: List[Dict[str, Any]], train_idx: List[int], val_idx: List[int],
    args, split_label: str,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Fit mọi setup trên một split, trả (results, bảng prediction của hàng val)."""
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    y_fit = y.copy()
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(train_idx))
        y_fit[train_idx] = y[np.array(train_idx)[perm]]
        print("⚠️  --shuffle-labels: đã xáo nhãn TRONG hàng train (control S5)")

    n_tr, n_va = len(train_idx), len(val_idx)
    results: List[Dict[str, Any]] = []
    preds: Dict[str, np.ndarray] = {}

    def add(name: str, m: Dict[str, Any]) -> None:
        results.append(
            {
                "setup": name, "target": args.target, "split": split_label,
                "n_train": n_tr, "n_val": n_va,
                "n_features": m["n_features"], "MAE": m["MAE"],
                "RMSE": m["RMSE"], "R2": m["R2"], "Pearson": m["Pearson"],
            }
        )
        if "_pred" in m:
            preds[name] = m["_pred"]

    y_val = y[val_idx]

    # --- Baseline 1: global mean của train ---
    gm = float(y_fit[train_idx].mean())
    p = np.full(n_va, gm, dtype=np.float32)
    add("Baseline global-mean", {**_metrics(y_val, p), "n_features": 0, "_pred": p})

    # --- Baseline 2: mean theo dance_id của train ---
    buckets: Dict[str, List[float]] = {}
    for i in train_idx:
        buckets.setdefault(rows[i]["dance_id"], []).append(float(y_fit[i]))
    dmean = {k: float(np.mean(v)) for k, v in buckets.items()}
    p = np.array([dmean.get(rows[i]["dance_id"], gm) for i in val_idx], dtype=np.float32)
    add("Baseline dance-mean", {**_metrics(y_val, p), "n_features": 0, "_pred": p})

    # --- Các setup có feature ---
    for name, X in build_setups(rows, args.with_dance_onehot).items():
        X_use = _reduce(X, train_idx, args.pca_dim, args.seed)
        m = _fit_xgb(
            X_use[train_idx], y_fit[train_idx], X_use[val_idx], y_val, seed=args.seed
        )
        add(name, m)

    pred_df = pd.DataFrame(
        {
            "dance_id": [rows[i]["dance_id"] for i in val_idx],
            "video_id": [rows[i]["video_id"] for i in val_idx],
            "person_id": [rows[i]["person_id"] for i in val_idx],
            "y_true": y_val,
            **{f"pred::{k}": v for k, v in preds.items()},
        }
    )
    return results, pred_df


def _print_table(results: List[Dict[str, Any]]) -> None:
    print(f"\n{'setup':<34} {'#f':>4} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'r':>8}")
    print("-" * 74)
    for r in results:
        print(
            f"{r['setup']:<34} {r['n_features']:4d} {r['MAE']:8.3f} "
            f"{r['RMSE']:8.3f} {r['R2']:8.3f} {r['Pearson']:8.3f}"
        )


def _print_verdict(results: List[Dict[str, Any]]) -> None:
    by = {r["setup"]: r for r in results}

    def d(a: str, b: str) -> None:
        """ΔMAE: dương = b tốt hơn a."""
        if a in by and b in by:
            print(f"  ΔMAE[{a} → {b}] = {by[a]['MAE'] - by[b]['MAE']:+.3f}")

    print("\nVerdict (ΔMAE dương = kịch bản sau tốt hơn kịch bản trước):")
    d("(a) DTW-only", "(b) LSTM-only")
    d("(a) DTW-only", "(c) DTW+LSTM")
    d("(c-) DTW+LSTM, no-dtw encoder", "(c) DTW+LSTM")
    d("(a) DTW-only +dance", "(c) DTW+LSTM +dance")
    best = min(
        (r for r in results if not r["setup"].startswith("Baseline")),
        key=lambda r: r["MAE"], default=None,
    )
    if best and "Baseline dance-mean" in by:
        delta = by["Baseline dance-mean"]["MAE"] - best["MAE"]
        print(
            f"  ΔMAE[Baseline dance-mean → {best['setup']}] = {delta:+.3f}"
            f"   {'(model thắng baseline)' if delta > 0 else '(KHÔNG thắng baseline)'}"
        )
    print(
        "\nLưu ý: n_val nhỏ → SE(MAE) ≈ 0.45 điểm. Chênh lệch dưới ~1 điểm "
        "KHÔNG phân định được trên một split. Xem bảng CV."
    )


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames=FIELDNAMES) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi: {path}")


#: Các cặp đáng kiểm định — (trước, sau); dương = "sau" tốt hơn.
PAIRED_COMPARISONS: List[Tuple[str, str]] = [
    ("(a) DTW-only", "(c) DTW+LSTM"),
    ("(a) DTW-only", "(b) LSTM-only"),
    ("(c-) DTW+LSTM, no-dtw encoder", "(c) DTW+LSTM"),
    ("Baseline dance-mean", "(c) DTW+LSTM"),
    ("Baseline global-mean", "(c) DTW+LSTM"),
    ("Geometry-only (no DL)", "(c) DTW+LSTM"),
    ("Baseline khop_dong_tac (oracle)", "(c) DTW+LSTM"),
    ("(c) DTW+LSTM", "(c) DTW+LSTM +dance"),
    ("(a) DTW-only", "(a+) DTW-only extended"),
]


def _paired_tests(
    agg: Dict[str, List[Dict[str, Any]]], target: str, path: str
) -> None:
    """
    Kiểm định t ghép cặp trên các fold.

    Các fold dùng chung tập val cho mọi setup, nên so sánh ghép cặp mạnh hơn
    nhiều so với đối chiếu hai khoảng `mean ± std` — vốn chỉ cho thấy chúng phủ
    nhau mà không tính tới việc fold khó thì mọi setup đều tệ.
    """
    from scipy.stats import ttest_rel

    rows: List[Dict[str, Any]] = []
    for before, after in PAIRED_COMPARISONS:
        if before not in agg or after not in agg:
            continue
        # Ghép theo đúng fold, không dựa vào thứ tự chèn
        by_fold_b = {r["split"]: r["MAE"] for r in agg[before]}
        by_fold_a = {r["split"]: r["MAE"] for r in agg[after]}
        folds = sorted(set(by_fold_b) & set(by_fold_a))
        if len(folds) < 3:
            continue
        x = np.array([by_fold_b[f] for f in folds], dtype=float)
        y = np.array([by_fold_a[f] for f in folds], dtype=float)
        d = x - y                      # dương = `after` tốt hơn
        t, p = ttest_rel(x, y)
        rows.append(
            {
                "before": before, "after": after, "target": target,
                "folds": len(folds),
                "dMAE": float(d.mean()),
                "SE": float(d.std(ddof=1) / np.sqrt(len(folds))),
                "t": float(t), "p": float(p),
                "significant_0.05": bool(p < 0.05),
            }
        )

    if not rows:
        return
    print(f"\n{'so sánh (dMAE dương = vế sau tốt hơn)':<52}{'dMAE':>8}{'t':>7}{'p':>9}")
    print("-" * 76)
    for r in rows:
        mark = "✅" if r["p"] < 0.05 else ("~" if r["p"] < 0.10 else "")
        print(
            f"{r['before'] + ' → ' + r['after']:<52}"
            f"{r['dMAE']:+8.3f}{r['t']:7.2f}{r['p']:9.4f} {mark}"
        )
    print(
        "\nGhép cặp theo fold mạnh hơn việc nhìn hai khoảng mean±std phủ nhau: "
        "fold khó thì mọi setup đều tệ, ghép cặp khử được yếu tố đó."
    )
    _write_csv(
        path, rows,
        ["before", "after", "target", "folds", "dMAE", "SE", "t", "p",
         "significant_0.05"],
    )


def _sanity_report(rows: List[Dict[str, Any]], path: str) -> None:
    """S6 — phát hiện encoder suy biến (ReLU chết → embedding hằng số)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = ["Sanity check embedding — Person 2 Tuần 5", "=" * 44, ""]
    for key, label in (("emb_c", "(c) encoder 17-ch"), ("emb_b", "(b) encoder 16-ch")):
        if not all(r[key] is not None for r in rows):
            lines.append(f"{label}: không có embedding (thiếu checkpoint)\n")
            continue
        E = np.stack([r[key] for r in rows])
        stds = E.std(axis=0)
        n_dead = int((stds < 1e-6).sum())
        lines += [
            f"{label}: {E.shape[0]} video × {E.shape[1]} chiều",
            f"  std theo chiều: min={stds.min():.3e} median={np.median(stds):.3e} "
            f"max={stds.max():.3e}",
            f"  chiều chết (std < 1e-6): {n_dead}/{E.shape[1]}",
            f"  ‖emb‖₂: mean={np.linalg.norm(E, axis=1).mean():.3f}",
        ]
        if n_dead > E.shape[1] // 2:
            lines.append(
                "  ❌ FAIL: hơn nửa số chiều là hằng số → ReLU cuối đã chết, "
                "mọi correlation từ embedding này vô nghĩa."
            )
        lines.append("")
    nw = [r.get("n_windows") for r in rows if r.get("n_windows")]
    if nw:
        lines.append(f"n_window/video: min={min(nw)} max={max(nw)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Đã ghi: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scores", default="annotations/scores.csv")
    ap.add_argument("--dtw", default="annotations/dtw_features.csv")
    ap.add_argument("--poses", default="poses")
    ap.add_argument("--ckpt-dtw", default=None, help="Encoder 17-ch cho kịch bản (c)")
    ap.add_argument("--ckpt-nodtw", default=None, help="Encoder 16-ch cho kịch bản (b)")
    ap.add_argument("--ckpt-dir", default="experiments/temporal_dl",
                    help="Folder chứa temporal_khopnhip_{dtw,nodtw}_fold{k}.pth (chế độ CV)")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help=">0 = chế độ GroupKFold, đọc split từ config của từng checkpoint")
    ap.add_argument("--target", default="khop_nhip")
    ap.add_argument("--split", default="official",
                    choices=["official", "groupkfold"])
    ap.add_argument("--train-csv", default="annotations/scores_train.csv")
    ap.add_argument("--val-csv", default="annotations/scores_test.csv")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--hop", type=int, default=30)
    ap.add_argument("--aggregate", default="mean_std",
                    choices=["mean", "mean_std", "mean_std_minmax"])
    ap.add_argument("--with-dance-onehot", action="store_true",
                    help="Thêm khối +dance (one-hot dance_id) — khối kết luận dựa vào")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="Control S5: xáo nhãn trong hàng train, mọi MAE phải tụt về baseline")
    ap.add_argument("--allow-leakage", action="store_true",
                    help="Bỏ qua assert leakage encoder (chỉ để chạy bản chẩn đoán)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-preds", default=None)
    ap.add_argument("--out-sanity", default=None)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    suffix = ""
    if args.shuffle_labels:
        suffix = "_shuffled"
    elif args.allow_leakage:
        suffix = "_LEAKY"
    base = f"experiments/temporal_dl/ablation_xgb_{args.target}"
    out = args.out or (
        f"{base}_cv.csv" if args.cv_folds > 0 else f"{base}{suffix}.csv"
    )
    out_preds = args.out_preds or f"experiments/temporal_dl/ablation_preds_{args.target}{suffix}.csv"
    out_sanity = args.out_sanity or f"experiments/temporal_dl/embeddings_sanity_{args.target}{suffix}.txt"
    out_folds = f"{base}_folds.csv"
    out_paired = f"experiments/temporal_dl/ablation_paired_tests_{args.target}.csv"

    # ════════ Chế độ CV ════════
    if args.cv_folds > 0:
        all_fold_results: List[Dict[str, Any]] = []
        for k in range(args.cv_folds):
            c_path = os.path.join(args.ckpt_dir, f"temporal_{args.target.replace('_','')}_dtw_fold{k}.pth")
            b_path = os.path.join(args.ckpt_dir, f"temporal_{args.target.replace('_','')}_nodtw_fold{k}.pth")
            if not os.path.isfile(c_path):
                raise SystemExit(f"Thiếu checkpoint fold {k}: {c_path}")
            enc_c, ck_c = load_temporal_checkpoint(c_path)
            enc_b, ck_b = (
                load_temporal_checkpoint(b_path) if os.path.isfile(b_path) else (None, {})
            )
            cfg_c = ck_c.get("config", {})
            val_keys = [tuple(x) for x in cfg_c["val_videos"]]
            _assert_no_encoder_leakage(
                f"fold{k}/ckpt-dtw", cfg_c, val_keys, args.target, True, args.allow_leakage
            )
            if enc_b is not None:
                _assert_no_encoder_leakage(
                    f"fold{k}/ckpt-nodtw", ck_b.get("config", {}), val_keys,
                    args.target, False, args.allow_leakage,
                )
            data = collect_features(args, enc_c, enc_b)
            rows = data["rows"]
            key_to_i = {r["key"]: i for i, r in enumerate(rows)}
            val_set = set(val_keys)
            val_idx = [key_to_i[k2] for k2 in val_keys if k2 in key_to_i]
            train_idx = [i for key, i in key_to_i.items() if key not in val_set]
            print(f"\n──── fold {k}: train={len(train_idx)} val={len(val_idx)} ────")
            res, _ = run_split(rows, sorted(train_idx), sorted(val_idx), args, f"fold{k}")
            _print_table(res)
            all_fold_results.extend(res)

        agg: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_fold_results:
            agg.setdefault(r["setup"], []).append(r)
        cv_rows = [
            {
                "setup": name, "target": args.target, "folds": len(rs),
                "MAE_mean": float(np.mean([r["MAE"] for r in rs])),
                "MAE_std": float(np.std([r["MAE"] for r in rs])),
                "RMSE_mean": float(np.mean([r["RMSE"] for r in rs])),
                "RMSE_std": float(np.std([r["RMSE"] for r in rs])),
                "R2_mean": float(np.mean([r["R2"] for r in rs])),
                "Pearson_mean": _safe_nanmean([r["Pearson"] for r in rs]),
            }
            for name, rs in agg.items()
        ]
        cv_rows.sort(key=lambda r: r["MAE_mean"])
        print(f"\n{'setup':<34} {'MAE_mean':>9} {'MAE_std':>8} {'RMSE_mean':>10}")
        print("-" * 64)
        for r in cv_rows:
            print(f"{r['setup']:<34} {r['MAE_mean']:9.3f} {r['MAE_std']:8.3f} "
                  f"{r['RMSE_mean']:10.3f}")
        _write_csv(
            out, cv_rows,
            ["setup", "target", "folds", "MAE_mean", "MAE_std",
             "RMSE_mean", "RMSE_std", "R2_mean", "Pearson_mean"],
        )
        # Lưu từng fold — không có thì không tái lập được kiểm định ghép cặp
        _write_csv(out_folds, all_fold_results)
        _paired_tests(agg, args.target, out_paired)
        return

    # ════════ Chế độ single split ════════
    enc_c = enc_b = None
    cfg_c: Dict[str, Any] = {}
    cfg_b: Dict[str, Any] = {}
    if args.ckpt_dtw:
        enc_c, ck = load_temporal_checkpoint(args.ckpt_dtw)
        cfg_c = ck.get("config", {})
    else:
        print("WARNING: không có --ckpt-dtw → bỏ kịch bản (c)")
    if args.ckpt_nodtw:
        enc_b, ck = load_temporal_checkpoint(args.ckpt_nodtw)
        cfg_b = ck.get("config", {})
    else:
        print("WARNING: không có --ckpt-nodtw → bỏ kịch bản (b)")

    data = collect_features(args, enc_c, enc_b)
    rows = data["rows"]
    key_to_i = {r["key"]: i for i, r in enumerate(rows)}

    train_keys = set(_read_keys(args.train_csv))
    val_keys_l = [k for k in _read_keys(args.val_csv) if k in key_to_i]
    val_idx = sorted(key_to_i[k] for k in val_keys_l)
    train_idx = sorted(key_to_i[k] for k in train_keys if k in key_to_i)

    tr_p = {rows[i]["person_id"] for i in train_idx}
    va_p = {rows[i]["person_id"] for i in val_idx}
    print(
        f"len(train)={len(train_idx)}, len(val)={len(val_idx)}, "
        f"train_persons={len(tr_p)}, val_persons={len(va_p)}, "
        f"key_overlap={len(set(train_idx) & set(val_idx))}, "
        f"person_overlap={len(tr_p & va_p)}"
    )
    if set(train_idx) & set(val_idx) or (tr_p & va_p):
        raise SystemExit("Split không rời nhau — dừng.")
    print(f"Val persons: {sorted(va_p)}")

    if enc_c is not None:
        _assert_no_encoder_leakage(
            "ckpt-dtw", cfg_c, val_keys_l, args.target, True, args.allow_leakage
        )
    if enc_b is not None:
        _assert_no_encoder_leakage(
            "ckpt-nodtw", cfg_b, val_keys_l, args.target, False, args.allow_leakage
        )

    results, pred_df = run_split(rows, train_idx, val_idx, args, args.split)
    _print_table(results)
    _print_verdict(results)

    _write_csv(out, results)
    os.makedirs(os.path.dirname(out_preds) or ".", exist_ok=True)
    pred_df.to_csv(out_preds, index=False)
    print(f"Đã ghi: {out_preds}")
    _sanity_report(rows, out_sanity)


if __name__ == "__main__":
    main()
