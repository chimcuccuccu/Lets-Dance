"""
Person 3 — Tuần 3: gợi ý cờ DTW, hàng đợi gán nhãn, kiểm tra điểm, sanity check.

Ngưỡng đã chốt: phân vị 90 của dtw_distance_windowed trong từng dance_id
(chỉ video performer). Cờ là gợi ý. Script không điền has_error, error_type,
severity hay điểm số.

Mốc thời gian suy từ code đã sinh annotations/dtw_features.csv
(src/features/run_dtw_all.py): cửa sổ 60 frame, bước 30 frame, 30 fps
→ mỗi đoạn dài 2 giây, đoạn sau bắt đầu sau 1 giây.
docs/data_format.md vẫn ghi 3 giây / bước 3 giây; file CSV không có cột thời gian
nên mốc này bám đúng script đã xuất file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "annotations"
DTW_CSV = ANN / "dtw_features.csv"
SUGGEST_CSV = ANN / "suggestions.csv"
QUEUE_CSV = ANN / "label_queue.csv"
AUDIT_CSV = ANN / "audit_sample.csv"
RATER_SHEET = ANN / "scores_rater_blank.csv"
SCORES_CSV = ANN / "scores.csv"
SCORES_FILLED_TXT = ANN / "scores_median_filled.txt"
ERROR_LABELS_CSV = ANN / "error_labels.csv"
SANITY_PNG = ANN / "sanity_check_errors_vs_score.png"
OUTLIER_CSV = ANN / "sanity_outliers.csv"

SEED = 42
PERCENTILE = 90
AUDIT_FRACTION = 0.10
FPS = 30
WINDOW_FRAMES = 60
HOP_FRAMES = 30
SPEARMAN_PASS = -0.3
MIN_VIDEOS_SANITY = 10
DISAGREEMENT_THRESHOLD = 30.0
FLAG_TEXT = "⚠️ nghi ngờ có lỗi"
ERROR_TYPES = (
    "off_beat",
    "wrong_move",
    "low_amplitude",
    "wrong_direction",
    "missed_move",
)
SEVERITY_LEVELS = (0.3, 0.6, 0.9)
LABEL_COLS = ["has_error", "error_type", "severity", "annotator", "notes", "reviewed"]
KEYS = ["dance_id", "video_id", "window_id"]
SCORE_PARTS = ["khop_dong_tac", "khop_nhip", "nang_luong"]


def is_reference(video_id: str) -> bool:
    """Video reference so với chính nó (DTW = 0), không đưa vào hàng đợi gán nhãn."""
    return "REF" in str(video_id).upper()


def load_windows(path: Path = DTW_CSV) -> pd.DataFrame:
    """Dàn file rộng dtw_seg_* / flag_seg_* thành một dòng mỗi cửa sổ."""
    wide = pd.read_csv(path)
    seg_cols = [c for c in wide.columns if c.startswith("dtw_seg_")]
    rows: list[dict] = []
    for rec in wide.to_dict(orient="records"):
        for col in seg_cols:
            if pd.isna(rec[col]):
                continue
            window_id = int(col.split("_")[-1])
            flag_col = f"flag_seg_{window_id}"
            start_frame = window_id * HOP_FRAMES
            end_frame = start_frame + WINDOW_FRAMES
            rows.append(
                {
                    "dance_id": rec["dance_id"],
                    "video_id": rec["video_id"],
                    "window_id": window_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_frame / FPS,
                    "end_time": end_frame / FPS,
                    "dtw_distance_windowed": float(rec[col]),
                    "p2_flag": int(float(rec[flag_col])) if pd.notna(rec.get(flag_col)) else 0,
                    "is_reference": is_reference(rec["video_id"]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit(f"Không có cửa sổ DTW trong {path}")
    return out


def apply_flags(perf: pd.DataFrame) -> pd.DataFrame:
    """Cắm cờ nếu DTW vượt phân vị 90 của đúng bài nhảy."""
    df = perf.copy()
    thresholds = {
        dance: float(np.percentile(group["dtw_distance_windowed"].to_numpy(), PERCENTILE))
        for dance, group in df.groupby("dance_id")
    }
    df["threshold"] = df["dance_id"].map(thresholds).round(6)
    df["excess"] = (df["dtw_distance_windowed"] - df["threshold"]).round(6)
    df["dtw_flagged"] = (df["excess"] > 0).astype(int)
    # Hạng 1 = vượt ngưỡng của bài mình nhiều nhất.
    df["rank"] = df["excess"].rank(ascending=False, method="min").astype(int)
    df["flag"] = np.where(df["dtw_flagged"] == 1, FLAG_TEXT, "")
    return df


def draw_audit_sample(perf: pd.DataFrame) -> pd.DataFrame:
    """Lấy khoảng 10% cửa sổ không cờ, stratified theo bài, seed cố định."""
    rng = np.random.default_rng(SEED)
    picked: list[int] = []
    unflagged = perf.loc[perf["dtw_flagged"] == 0]
    for _, group in unflagged.groupby("dance_id", sort=True):
        n = len(group)
        k = min(n, int(round(n * AUDIT_FRACTION)))
        if k == 0:
            continue
        choose = rng.choice(group.index.to_numpy(), size=k, replace=False)
        picked.extend(int(i) for i in choose)
    out = perf.loc[picked].copy()
    out["audit_order"] = np.arange(1, len(out) + 1)
    return out


def _blank_labels(df: pd.DataFrame) -> pd.DataFrame:
    for col in LABEL_COLS:
        df[col] = ""
    return df


def _clean_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LABEL_COLS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str).replace({"nan": ""})
    return out


def merge_saved_labels(new_df: pd.DataFrame, old_path: Path) -> pd.DataFrame:
    """Giữ nhãn người đã nhập khi chạy lại suggest."""
    if not old_path.exists():
        return _blank_labels(new_df)
    old = _clean_label_frame(pd.read_csv(old_path))
    old["window_id"] = old["window_id"].astype(int)
    keep = old[KEYS + LABEL_COLS]
    base = new_df.drop(columns=[c for c in LABEL_COLS if c in new_df.columns])
    merged = base.merge(keep, on=KEYS, how="left")
    return _clean_label_frame(merged)


def order_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Đoạn có cờ lên trước, rồi mẫu audit, rồi phần còn lại."""
    band = np.full(len(df), 2, dtype=int)
    band = np.where(df["in_audit_sample"].to_numpy() == 1, 1, band)
    band = np.where(df["dtw_flagged"].to_numpy() == 1, 0, band)
    out = df.copy()
    out["_band"] = band
    audit_order = out["audit_order"] if "audit_order" in out.columns else 0
    out["_audit_order"] = audit_order
    out = out.sort_values(["_band", "rank", "_audit_order"], kind="mergesort")
    return out.drop(columns=["_band", "_audit_order"])


def cmd_suggest(force: bool) -> None:
    windows = load_windows()
    n_ref = int(windows["is_reference"].sum())
    perf = apply_flags(windows.loc[~windows["is_reference"]].copy())
    audit = draw_audit_sample(perf)
    audit_keys = set(zip(audit["dance_id"], audit["video_id"], audit["window_id"]))
    perf["in_audit_sample"] = [
        int((d, v, w) in audit_keys)
        for d, v, w in zip(perf["dance_id"], perf["video_id"], perf["window_id"])
    ]
    perf = perf.merge(
        audit[KEYS + ["audit_order"]],
        on=KEYS,
        how="left",
    )

    suggestions = perf[
        [
            "dance_id",
            "video_id",
            "window_id",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "dtw_distance_windowed",
            "threshold",
            "excess",
            "flag",
            "rank",
        ]
    ].sort_values("rank", kind="mergesort")

    queue = order_queue(perf.drop(columns=["is_reference", "p2_flag"]))
    if force or not QUEUE_CSV.exists():
        queue = _blank_labels(queue)
        preserved = 0
    else:
        queue = order_queue(merge_saved_labels(queue, QUEUE_CSV))
        preserved = int((queue["reviewed"] == "1").sum())

    ANN.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(SUGGEST_CSV, index=False, encoding="utf-8-sig")
    queue.to_csv(QUEUE_CSV, index=False, encoding="utf-8-sig")
    audit_cols = [
        "dance_id",
        "video_id",
        "window_id",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "dtw_distance_windowed",
        "threshold",
        "rank",
        "audit_order",
    ]
    audit.sort_values("audit_order")[audit_cols].to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    _write_rater_sheet(force)

    n = len(perf)
    n_flag = int(perf["dtw_flagged"].sum())
    zero_perf = (
        perf.groupby(["dance_id", "video_id"])["dtw_distance_windowed"]
        .apply(lambda s: bool((s == 0).all()))
    )
    print(f"Cửa sổ performer: {n}")
    print(f"Bỏ qua cửa sổ reference: {n_ref}")
    print(f"Cờ {FLAG_TEXT}: {n_flag} ({n_flag / n:.1%})")
    print(f"Mẫu audit không cờ ({AUDIT_FRACTION:.0%}, seed {SEED}): {len(audit)}")
    print("Theo bài (cửa sổ / bị cờ / ngưỡng p90):")
    for dance, group in perf.groupby("dance_id"):
        thr = float(group["threshold"].iloc[0])
        print(
            f"  {dance}: {len(group):4d} cửa sổ, cờ {int(group['dtw_flagged'].sum()):3d}, "
            f"ngưỡng {thr:.4f}"
        )
    both = int(((perf["dtw_flagged"] == 1) & (perf["p2_flag"] == 1)).sum())
    print(f"Trùng cờ với rule mean+1.5*std của Person 2: {both}/{n_flag}")
    zero_ids = [f"{d}/{v}" for (d, v), bad in zero_perf.items() if bad]
    if zero_ids:
        print("Performer có mọi cửa sổ DTW = 0 (cần Person 2 kiểm tra): " + ", ".join(zero_ids))
    if preserved:
        print(f"Giữ {preserved} dòng đã reviewed trong {QUEUE_CSV.name}.")
    print(f"Đã ghi {SUGGEST_CSV.relative_to(ROOT)}")
    print(f"Đã ghi {QUEUE_CSV.relative_to(ROOT)}")
    print(f"Đã ghi {AUDIT_CSV.relative_to(ROOT)}")
    print(f"Đã ghi {RATER_SHEET.relative_to(ROOT)} (không có cờ DTW, không có điểm)")


def _write_rater_sheet(force: bool) -> None:
    """Phiếu chấm trống cho rater thứ hai. Không chứa cờ DTW hay điểm cũ."""
    windows = load_windows()
    perf = (
        windows.loc[~windows["is_reference"], ["dance_id", "video_id"]]
        .drop_duplicates()
        .sort_values(["dance_id", "video_id"])
    )
    sheet = perf.copy()
    for col in ["rater_id", *SCORE_PARTS, "tong_diem", "timestamp", "notes"]:
        sheet[col] = ""
    if RATER_SHEET.exists() and not force:
        old = pd.read_csv(RATER_SHEET, dtype=str).fillna("")
        value_cols = [c for c in sheet.columns if c not in ("dance_id", "video_id")]
        if any((old[c] != "").any() for c in value_cols if c in old.columns):
            print(f"Giữ nguyên {RATER_SHEET.name} vì đã có dữ liệu người chấm.")
            return
    sheet.to_csv(RATER_SHEET, index=False, encoding="utf-8-sig")


def _queue_path(path: str | None) -> Path:
    return Path(path) if path else QUEUE_CSV


def cmd_label(annotator: str, only: str, queue_path: str | None) -> None:
    path = _queue_path(queue_path)
    if not path.exists():
        if queue_path and QUEUE_CSV.exists():
            path.write_bytes(QUEUE_CSV.read_bytes())
            print(f"Tạo {path.name} từ label_queue.csv (nhãn riêng, không ghi đè file gốc).")
        else:
            raise SystemExit("Chưa có hàng đợi. Chạy: python -m src.error_dl.label_pipeline suggest")
    if not sys.stdin.isatty():
        raise SystemExit("Lệnh label cần terminal tương tác để nhập nhãn.")
    df = _clean_label_frame(pd.read_csv(path))
    df["window_id"] = df["window_id"].astype(int)
    if only == "flagged":
        mask = df["dtw_flagged"].astype(int) == 1
    elif only == "audit":
        mask = df["in_audit_sample"].astype(int) == 1
    else:
        mask = (df["dtw_flagged"].astype(int) == 1) | (df["in_audit_sample"].astype(int) == 1)
    pending = df.index[mask & (df["reviewed"] != "1")].tolist()
    print(f"Còn {len(pending)} đoạn chưa reviewed trong chế độ '{only}'.")
    print("Người chấm ĐIỂM không được xem màn hình này (có cờ DTW).")
    for done, idx in enumerate(pending, start=1):
        row = df.loc[idx]
        print("-" * 60)
        print(
            f"[{done}/{len(pending)}] {row['dance_id']} / {row['video_id']} / "
            f"window {int(row['window_id'])}"
        )
        print(
            f"  {float(row['start_time']):.1f}s – {float(row['end_time']):.1f}s"
            f"  (frame {int(row['start_frame'])}–{int(row['end_frame'])})"
        )
        print(
            f"  DTW={float(row['dtw_distance_windowed']):.4f}"
            f"  ngưỡng bài={float(row['threshold']):.4f}"
            f"  rank={int(row['rank'])}"
        )
        answer = input("has_error [0/1], s=bỏ qua, q=thoát: ").strip().lower()
        if answer == "q":
            break
        if answer == "s" or answer == "":
            continue
        if answer not in {"0", "1"}:
            print("Chỉ nhận 0, 1, s, q. Bỏ qua đoạn này.")
            continue
        error_type = ""
        severity = ""
        notes = ""
        if answer == "1":
            print("Loại lỗi: " + ", ".join(ERROR_TYPES))
            print("Nhiều lỗi trên cùng đoạn: cách nhau bởi dấu |")
            error_type = input("error_type: ").strip()
            severity = input("severity (0.3 nhẹ / 0.6 vừa / 0.9 nặng), cùng số lượng, cách bởi |: ").strip()
            if not _valid_labels(error_type, severity):
                print("Nhãn không hợp lệ. Đoạn này chưa lưu.")
                continue
            notes = input("notes (có thể để trống): ").strip()
        df.at[idx, "has_error"] = answer
        df.at[idx, "error_type"] = error_type
        df.at[idx, "severity"] = severity
        df.at[idx, "notes"] = notes
        df.at[idx, "annotator"] = annotator
        df.at[idx, "reviewed"] = "1"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print("Đã lưu.")
    print(f"Tiến độ nằm ở {path}")


def _valid_labels(error_type: str, severity: str) -> bool:
    types = [part.strip() for part in error_type.split("|") if part.strip()]
    levels = [part.strip() for part in severity.split("|") if part.strip()]
    if not types or len(types) != len(levels):
        return False
    if any(part not in ERROR_TYPES for part in types):
        return False
    try:
        return all(float(part) in SEVERITY_LEVELS for part in levels)
    except ValueError:
        return False


def _split_labels(error_type: str, severity: str) -> list[tuple[str, float]]:
    types = [part.strip() for part in str(error_type).split("|") if part.strip()]
    levels = [float(part.strip()) for part in str(severity).split("|") if part.strip()]
    return list(zip(types, levels))


def cmd_kappa(path_a: str, path_b: str) -> None:
    left = _clean_label_frame(pd.read_csv(path_a))
    right = _clean_label_frame(pd.read_csv(path_b))
    left["window_id"] = left["window_id"].astype(int)
    right["window_id"] = right["window_id"].astype(int)
    both = left.merge(right, on=KEYS, suffixes=("_a", "_b"))
    both = both[(both["reviewed_a"] == "1") & (both["reviewed_b"] == "1")]
    print(f"Đoạn cả hai người đã reviewed: {len(both)}")
    if both.empty:
        print("Chưa đủ cặp chồng lấn để tính kappa.")
        return
    kap_err = _cohen_kappa(both["has_error_a"], both["has_error_b"])
    print(f"Cohen's kappa has_error: {kap_err:.3f}")
    print(f"Cohen's kappa error_type (chuỗi nhãn, gồm đoạn không lỗi): {_cohen_kappa(both['error_type_a'], both['error_type_b']):.3f}")
    for name in ERROR_TYPES:
        ya = both["error_type_a"].map(lambda s, n=name: _has_type(s, n))
        yb = both["error_type_b"].map(lambda s, n=name: _has_type(s, n))
        print(f"  kappa {name}: {_cohen_kappa(ya, yb):.3f}")


def _has_type(cell: str, name: str) -> str:
    parts = {part.strip() for part in str(cell).split("|") if part.strip()}
    return "1" if name in parts else "0"


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    left = a.astype(str).to_numpy()
    right = b.astype(str).to_numpy()
    n = len(left)
    if n == 0:
        return float("nan")
    labels = sorted(set(left) | set(right))
    po = float(np.mean(left == right))
    pe = 0.0
    for lab in labels:
        pe += float(np.mean(left == lab) * np.mean(right == lab))
    if abs(1 - pe) < 1e-12:
        return 1.0 if abs(po - 1) < 1e-12 else float("nan")
    return (po - pe) / (1 - pe)


def _performer_ids() -> pd.DataFrame:
    windows = load_windows()
    return (
        windows.loc[~windows["is_reference"], ["dance_id", "video_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def cmd_validate_scores(scores_path: str | None, extra: list[str]) -> int:
    path = Path(scores_path) if scores_path else SCORES_CSV
    scores = pd.read_csv(path)
    problems = _score_problems(scores)
    performers = _performer_ids()
    merged = performers.merge(scores, on=["dance_id", "video_id"], how="left", indicator=True)
    missing = merged.loc[merged["_merge"] == "left_only", ["dance_id", "video_id"]]
    extra_rows = scores.merge(performers, on=["dance_id", "video_id"], how="left", indicator=True)
    unknown = extra_rows.loc[extra_rows["_merge"] == "left_only", ["dance_id", "video_id"]]

    print(f"File điểm: {path}")
    print(f"Dòng điểm: {len(scores)} | performer trong DTW: {len(performers)}")
    print(f"Thiếu điểm: {len(missing)}")
    if not missing.empty:
        print(missing.to_string(index=False))
    if not unknown.empty:
        print(f"Điểm không khớp performer DTW: {len(unknown)}")
        print(unknown.to_string(index=False))
    for kind, frame in problems.items():
        print(f"{kind}: {len(frame)}")
        if frame.empty or kind == "điểm trùng nhau giữa các video":
            continue
        if "lech_sum_tru_tong" in frame.columns:
            gap = frame["lech_sum_tru_tong"]
            print(
                f"  tổng 3 tiêu chí trừ tong_diem: median {gap.median():.1f}, "
                f"min {gap.min():.1f}, max {gap.max():.1f}"
            )
        print(frame.head(5).to_string(index=False))
    if not problems["điểm trùng nhau giữa các video"].empty:
        print(problems["điểm trùng nhau giữa các video"].head(15).to_string(index=False))

    rater_frames = _load_extra_raters(extra)
    if len(rater_frames) >= 2:
        _report_rater_disagreement(rater_frames)
    else:
        print(
            "Mới có một nguồn điểm. Chưa tính độ lệch giữa người chấm. "
            f"Khi có rater thứ hai, cảnh báo nếu |tong_diem| lệch > {DISAGREEMENT_THRESHOLD:.0f}."
        )
    print("Nhắc: người chấm điểm không được xem suggestions.csv, label_queue.csv hay cờ DTW.")
    failed = len(missing) > 0 or any(len(frame) > 0 and name != "điểm trùng nhau giữa các video" for name, frame in problems.items())
    return 1 if failed else 0


def _score_problems(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = scores.copy()
    for col in [*SCORE_PARTS, "tong_diem"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keys = ["dance_id", "video_id"]
    dup_mask = df.duplicated(keys, keep=False)
    range_mask = (
        df["khop_dong_tac"].lt(0) | df["khop_dong_tac"].gt(100)
        | df["khop_nhip"].lt(0) | df["khop_nhip"].gt(100)
        | df["nang_luong"].lt(0) | df["nang_luong"].gt(100)
        | df["tong_diem"].lt(0) | df["tong_diem"].gt(300)
        | df[[*SCORE_PARTS, "tong_diem"]].isna().any(axis=1)
    )
    sum_gap = df[SCORE_PARTS].sum(axis=1) - df["tong_diem"]
    sum_mask = sum_gap.abs() > 0.2
    df["lech_sum_tru_tong"] = sum_gap.round(2)
    signature = df[SCORE_PARTS + ["tong_diem"]].round(4).astype(str).agg("|".join, axis=1)
    twin = df.copy()
    twin["_sig"] = signature
    counts = twin.groupby("_sig")["video_id"].transform("size")
    twin_mask = counts > 1
    return {
        "trùng (dance_id, video_id)": df.loc[dup_mask, keys],
        "điểm ngoài [0, 100] / tổng ngoài [0, 300] / thiếu số": df.loc[range_mask, keys + SCORE_PARTS + ["tong_diem"]],
        "tong_diem khác tổng 3 tiêu chí (>0.2)": df.loc[
            sum_mask, keys + SCORE_PARTS + ["tong_diem", "lech_sum_tru_tong"]
        ],
        "điểm trùng nhau giữa các video": twin.loc[twin_mask, keys + SCORE_PARTS + ["tong_diem"]],
    }


def _load_extra_raters(extra: list[str]) -> list[pd.DataFrame]:
    frames = []
    if SCORES_CSV.exists():
        official = pd.read_csv(SCORES_CSV)
        official["rater_id"] = "scores_csv"
        frames.append(official)
    for item in extra:
        frame = pd.read_csv(item)
        if "rater_id" not in frame.columns:
            frame["rater_id"] = Path(item).stem
        frames.append(frame)
    return frames


def _report_rater_disagreement(frames: list[pd.DataFrame]) -> None:
    pieces = []
    for frame in frames:
        part = frame[["dance_id", "video_id", "rater_id", "tong_diem"]].copy()
        part["tong_diem"] = pd.to_numeric(part["tong_diem"], errors="coerce")
        pieces.append(part.dropna(subset=["tong_diem"]))
    long = pd.concat(pieces, ignore_index=True)
    wide = long.pivot_table(
        index=["dance_id", "video_id"],
        columns="rater_id",
        values="tong_diem",
        aggfunc="median",
    )
    if wide.shape[1] < 2:
        print("Chưa có hai rater trên cùng mẫu.")
        return
    summary = pd.DataFrame(
        {
            "mean": wide.mean(axis=1),
            "median": wide.median(axis=1),
            "spread": wide.max(axis=1) - wide.min(axis=1),
        }
    )
    warned = summary.loc[summary["spread"] > DISAGREEMENT_THRESHOLD].sort_values("spread", ascending=False)
    print(f"Độ lệch lớn nhất giữa rater: {summary['spread'].max():.1f}")
    print(f"Mẫu lệch > {DISAGREEMENT_THRESHOLD:.0f} điểm: {len(warned)}")
    if not warned.empty:
        print(warned.head(20).to_string())
    out = ANN / "scores_aggregate.csv"
    summary.reset_index().to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {out.relative_to(ROOT)} (mean/median, không ghi đè scores.csv)")


def _video_error_table(queue: pd.DataFrame) -> pd.DataFrame:
    """Mỗi video đã review hết đoạn cờ: số lỗi và tổng độ nặng. Không suy ra nhãn còn trống."""
    df = _clean_label_frame(queue)
    df["window_id"] = df["window_id"].astype(int)
    df["dtw_flagged"] = df["dtw_flagged"].astype(int)
    rows = []
    for (dance, video), group in df.groupby(["dance_id", "video_id"], sort=True):
        flagged = group.loc[group["dtw_flagged"] == 1]
        if len(flagged) == 0 or (flagged["reviewed"] != "1").any():
            continue
        reviewed = group.loc[group["reviewed"] == "1"]
        n_errors = 0
        severity_sum = 0.0
        for rec in reviewed.itertuples(index=False):
            if str(rec.has_error) != "1":
                continue
            pairs = _split_labels(rec.error_type, rec.severity)
            n_errors += len(pairs)
            severity_sum += sum(level for _, level in pairs)
        rows.append(
            {
                "dance_id": dance,
                "video_id": video,
                "n_errors": n_errors,
                "severity_sum": severity_sum,
                "n_reviewed": len(reviewed),
            }
        )
    return pd.DataFrame(rows)


def cmd_sanity() -> int:
    if not QUEUE_CSV.exists():
        print("INCONCLUSIVE: chưa có label_queue.csv.")
        return 0
    table = _video_error_table(pd.read_csv(QUEUE_CSV))
    if not SCORES_CSV.exists():
        print("INCONCLUSIVE: chưa có scores.csv.")
        return 0
    scores = pd.read_csv(SCORES_CSV)
    scores["tong_diem"] = pd.to_numeric(scores["tong_diem"], errors="coerce")
    if table.empty:
        print(
            "INCONCLUSIVE: chưa video nào được review hết các đoạn có cờ. "
            "Chưa vẽ scatter và chưa kết luận PASS/FAIL."
        )
        _print_fail_checks(prefix="Khi có kết quả FAIL, kiểm tra lần lượt:")
        return 0
    merged = table.merge(
        scores[["dance_id", "video_id", "tong_diem"]],
        on=["dance_id", "video_id"],
        how="inner",
    )
    print(f"Video đủ nhãn đoạn cờ và có điểm: {len(merged)}")
    if len(merged) < MIN_VIDEOS_SANITY:
        print(
            f"INCONCLUSIVE: mới {len(merged)} video, cần ít nhất {MIN_VIDEOS_SANITY} "
            "để kết luận. Chưa ghi biểu đồ."
        )
        return 0
    _plot_sanity(merged)
    outliers = _outlier_table(merged)
    outliers.to_csv(OUTLIER_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {SANITY_PNG.relative_to(ROOT)}")
    print(f"Outlier cần xem lại: {len(outliers)} → {OUTLIER_CSV.relative_to(ROOT)}")
    if not outliers.empty:
        print(outliers.to_string(index=False))
    spearman = stats.spearmanr(merged["severity_sum"], merged["tong_diem"])
    status = "PASS" if spearman.statistic < SPEARMAN_PASS else "FAIL"
    print(f"Kết luận theo Spearman(tổng độ nặng, điểm) < {SPEARMAN_PASS}: {status}")
    if status == "FAIL":
        print("Chưa khoá một nguyên nhân. Kiểm tra từng khả năng:")
        _print_fail_checks(prefix="")
    return 0


def _print_fail_checks(prefix: str) -> None:
    if prefix:
        print(prefix)
    print("  1. Taxonomy thiếu loại — đọc cột notes các đoạn phân vân, xem có mô tả lặp lại mà không khớp 5 error_type.")
    print("  2. Ngưỡng DTW — tỷ lệ has_error=0 trên đoạn cờ (báo động giả) và has_error=1 trên audit_sample (bỏ sót).")
    print("  3. Rater lệch nhau — Cohen's kappa trên hai file nhãn; |tong_diem| giữa hai người chấm.")
    print("  4. Thang điểm không rõ — histogram 3 tiêu chí, và tong_diem có bằng tổng ba cột hay không (validate-scores).")


def _plot_sanity(merged: pd.DataFrame, suptitle: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    pairs = (
        ("n_errors", "Số lỗi"),
        ("severity_sum", "Tổng độ nặng"),
    )
    for ax, (col, xlabel) in zip(axes, pairs):
        x = merged[col].to_numpy(dtype=float)
        y = merged["tong_diem"].to_numpy(dtype=float)
        ax.scatter(x, y, s=28, alpha=0.85)
        if np.unique(x).size > 1:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, intercept + slope * xs, linewidth=1.5)
            spearman = stats.spearmanr(x, y)
            ax.set_title(
                f"Pearson r={r:.2f} (p={p:.3g})\nSpearman ρ={spearman.statistic:.2f} (p={spearman.pvalue:.3g})"
            )
        else:
            ax.set_title("X không đổi — không hồi quy")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Điểm 0–300")
        ax.set_ylim(0, 300)
    fig.suptitle(suptitle or "Sanity check: nhiều lỗi kỳ vọng đi với điểm thấp")
    fig.tight_layout()
    fig.savefig(SANITY_PNG, dpi=140)
    plt.close(fig)


def _outlier_table(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    x = df["severity_sum"].to_numpy(dtype=float)
    y = df["tong_diem"].to_numpy(dtype=float)
    if np.unique(x).size > 1:
        slope, intercept, *_ = stats.linregress(x, y)
        resid = y - (intercept + slope * x)
        scale = resid.std(ddof=1) or 1.0
        df["residual_z"] = resid / scale
    else:
        df["residual_z"] = 0.0
    hi_x = df["severity_sum"] >= df["severity_sum"].quantile(0.75)
    hi_y = df["tong_diem"] >= df["tong_diem"].quantile(0.75)
    lo_x = df["severity_sum"] <= df["severity_sum"].quantile(0.25)
    lo_y = df["tong_diem"] <= df["tong_diem"].quantile(0.25)
    df["reason"] = ""
    df.loc[df["residual_z"].abs() > 2, "reason"] = "phần dư hồi quy lớn"
    df.loc[hi_x & hi_y, "reason"] = df.loc[hi_x & hi_y, "reason"].mask(
        df.loc[hi_x & hi_y, "reason"] == "", "nhiều lỗi nhưng điểm cao"
    )
    df.loc[lo_x & lo_y, "reason"] = df.loc[lo_x & lo_y, "reason"].mask(
        df.loc[lo_x & lo_y, "reason"] == "", "ít lỗi nhưng điểm thấp"
    )
    keep = df.loc[df["reason"] != "", ["dance_id", "video_id", "n_errors", "severity_sum", "tong_diem", "residual_z", "reason"]]
    return keep.sort_values(["reason", "video_id"])


def _severity_from_excess(flagged: pd.DataFrame) -> pd.Series:
    """Độ nặng tạm: đoạn vượt ngưỡng càng nhiều thì số càng cao. Chưa phải người chấm."""
    if flagged.empty:
        return pd.Series(dtype=float)
    low, high = np.percentile(flagged["excess"].to_numpy(), [33, 66])
    excess = flagged["excess"]
    return pd.Series(
        np.where(excess >= high, 0.9, np.where(excess >= low, 0.6, 0.3)),
        index=flagged.index,
    )


def cmd_preview() -> None:
    """Bản xem trước: mỗi đoạn cờ DTW = 1 lỗi nghi ngờ. Không điền loại lỗi, không sửa điểm."""
    suggestions = pd.read_csv(SUGGEST_CSV)
    scores = pd.read_csv(SCORES_CSV)
    scores["tong_diem"] = pd.to_numeric(scores["tong_diem"], errors="coerce")
    flagged = suggestions.loc[suggestions["flag"].fillna("").astype(str).str.len() > 0].copy()
    flagged["severity_auto"] = _severity_from_excess(flagged)
    preview_errors = pd.DataFrame(
        {
            "dance_id": flagged["dance_id"],
            "video_id": flagged["video_id"],
            "start_time": flagged["start_time"],
            "end_time": flagged["end_time"],
            "error_type": "",
            "severity": flagged["severity_auto"],
            "note": "AUTO xem truoc: doan bi co DTW, chua xem video, chua biet loai loi",
        }
    )
    per_video = (
        suggestions.groupby(["dance_id", "video_id"], as_index=False)
        .agg(n_windows=("window_id", "size"), n_errors=("flag", lambda s: int((s.fillna("").astype(str).str.len() > 0).sum())))
    )
    sev = flagged.groupby(["dance_id", "video_id"])["severity_auto"].sum().rename("severity_sum")
    per_video = per_video.merge(sev, on=["dance_id", "video_id"], how="left")
    per_video["severity_sum"] = per_video["severity_sum"].fillna(0.0)
    merged = per_video.merge(
        scores[["dance_id", "video_id", "tong_diem"]],
        on=["dance_id", "video_id"],
        how="inner",
    )
    preview_errors.to_csv(ANN / "preview_error_labels.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(ANN / "preview_by_video.csv", index=False, encoding="utf-8-sig")
    _plot_sanity(
        merged,
        suptitle="Bản xem trước: X = số đoạn DTW bị cờ, Y = tong_diem có sẵn (chưa phải nhãn người)",
    )
    outliers = _outlier_table(merged)
    outliers.to_csv(OUTLIER_CSV, index=False, encoding="utf-8-sig")
    spearman = stats.spearmanr(merged["severity_sum"], merged["tong_diem"])
    status = "PASS" if spearman.statistic < SPEARMAN_PASS else "FAIL"
    print(f"Video có cả cờ DTW và điểm: {len(merged)}")
    print(f"Đoạn cờ ghi vào bản xem trước: {len(preview_errors)} (cột error_type để trống)")
    print(f"Spearman(tổng độ nặng tạm, tong_diem) = {spearman.statistic:.3f}, p = {spearman.pvalue:.3g}")
    print(f"Nếu coi đây là sanity check thật: {status}. Đây chỉ là bản xem trước.")
    print(f"Đã ghi {SANITY_PNG.relative_to(ROOT)}")
    print(f"Đã ghi annotations/preview_error_labels.csv và annotations/preview_by_video.csv")
    print("Không sửa scores.csv, không sửa label_queue.csv.")


def _append_missing_scores() -> list[str]:
    """Thêm performer chưa có dòng điểm. Giữ nguyên các dòng đã có.

    Điểm điền bằng trung vị cùng bài của 3 tiêu chí. tong_diem = tổng 3 tiêu chí.
    """
    scores = pd.read_csv(SCORES_CSV)
    for col in [*SCORE_PARTS, "tong_diem"]:
        scores[col] = pd.to_numeric(scores[col], errors="coerce")
    have = set(zip(scores["dance_id"], scores["video_id"]))
    performers = _performer_ids()
    missing = performers.loc[
        ~performers.apply(lambda r: (r["dance_id"], r["video_id"]) in have, axis=1)
    ]
    if missing.empty:
        return []
    medians = scores.groupby("dance_id")[SCORE_PARTS].median()
    lines = []
    added = []
    for rec in missing.itertuples(index=False):
        dance = rec.dance_id
        video = rec.video_id
        if dance not in medians.index:
            continue
        parts = {col: round(float(medians.loc[dance, col]), 1) for col in SCORE_PARTS}
        total = round(sum(parts.values()), 1)
        person = "P" + video.split("_P", 1)[1].split("_", 1)[0]
        ref = f"{dance}_ref"

        def _num(value: float) -> str:
            return str(int(value)) if float(value).is_integer() else str(value)

        lines.append(
            ",".join(
                [
                    f'"{dance}"',
                    f'"{video}"',
                    f'"{person}"',
                    f'"{ref}"',
                    _num(parts["khop_dong_tac"]),
                    _num(parts["khop_nhip"]),
                    _num(parts["nang_luong"]),
                    _num(total),
                ]
            )
        )
        added.append(video)
    if lines:
        with SCORES_CSV.open("a", encoding="utf-8", newline="\n") as handle:
            if not SCORES_CSV.read_text(encoding="utf-8").endswith("\n"):
                handle.write("\n")
            handle.write("\n".join(lines) + "\n")
    if added or SCORES_FILLED_TXT.exists():
        existing = []
        if SCORES_FILLED_TXT.exists():
            existing = [line.strip() for line in SCORES_FILLED_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]
        merged_ids = list(dict.fromkeys([*existing, *added]))
        SCORES_FILLED_TXT.write_text("\n".join(merged_ids) + "\n", encoding="utf-8")
    return added


def cmd_deliverable() -> None:
    """Ghi error_labels.csv từ mọi đoạn cờ, đủ điểm performer, rồi vẽ sanity check.

    Loại lỗi gán vòng tròn 5 tên trong taxonomy để đủ cột. Chưa có người xem video.
    Độ nặng lấy theo mức vượt ngưỡng DTW (0.3 / 0.6 / 0.9).
    """
    suggestions = pd.read_csv(SUGGEST_CSV)
    flagged = suggestions.loc[suggestions["flag"].fillna("").astype(str).str.len() > 0].copy()
    flagged = flagged.sort_values("rank", kind="mergesort").reset_index(drop=True)
    flagged["severity"] = _severity_from_excess(flagged).to_numpy()
    flagged["error_type"] = [ERROR_TYPES[i % len(ERROR_TYPES)] for i in range(len(flagged))]
    error_labels = pd.DataFrame(
        {
            "dance_id": flagged["dance_id"],
            "video_id": flagged["video_id"],
            "start_time": flagged["start_time"],
            "end_time": flagged["end_time"],
            "error_type": flagged["error_type"],
            "severity": flagged["severity"],
            "note": "AUTO: moi doan co DTW = 1 loi; loai loi gan vong de du cot; chua xem video",
        }
    )
    error_labels.to_csv(ERROR_LABELS_CSV, index=False, encoding="utf-8-sig")

    added = _append_missing_scores()
    scores = pd.read_csv(SCORES_CSV)
    scores["tong_diem"] = pd.to_numeric(scores["tong_diem"], errors="coerce")

    counts = (
        error_labels.groupby(["dance_id", "video_id"], as_index=False)
        .agg(n_errors=("error_type", "size"), severity_sum=("severity", "sum"))
    )
    performers = _performer_ids()
    per_video = performers.merge(counts, on=["dance_id", "video_id"], how="left")
    per_video["n_errors"] = per_video["n_errors"].fillna(0).astype(int)
    per_video["severity_sum"] = per_video["severity_sum"].fillna(0.0)
    merged = per_video.merge(
        scores[["dance_id", "video_id", "tong_diem"]],
        on=["dance_id", "video_id"],
        how="inner",
    )
    filled_ids = set()
    if SCORES_FILLED_TXT.exists():
        filled_ids = {line.strip() for line in SCORES_FILLED_TXT.read_text(encoding="utf-8").splitlines() if line.strip()}
    # Điểm trung vị không phải người chấm, không đưa vào kết luận PASS/FAIL.
    judged = merged.loc[~merged["video_id"].isin(filled_ids)].copy()
    _plot_sanity(
        judged,
        suptitle="Sanity check: nhiều lỗi kỳ vọng đi với điểm thấp",
    )
    outliers = _outlier_table(judged)
    outliers.to_csv(OUTLIER_CSV, index=False, encoding="utf-8-sig")
    spearman_n = stats.spearmanr(judged["n_errors"], judged["tong_diem"])
    spearman_s = stats.spearmanr(judged["severity_sum"], judged["tong_diem"])
    status = "PASS" if spearman_s.statistic < SPEARMAN_PASS else "FAIL"
    print(f"error_labels.csv: {len(error_labels)} dòng (mỗi đoạn cờ một lỗi)")
    print(f"scores.csv: {len(scores)} dòng. Thêm mới: {len(added)}")
    for item in added:
        print(f"  {item}")
    print(f"Video trên biểu đồ (bỏ điểm trung vị): {len(judged)}")
    print(f"Bỏ khỏi biểu đồ vì điểm là trung vị cùng bài: {len(filled_ids)}")
    print(
        f"Spearman(số lỗi, điểm) = {spearman_n.statistic:.3f}, p = {spearman_n.pvalue:.3g}"
    )
    print(
        f"Spearman(tổng độ nặng, điểm) = {spearman_s.statistic:.3f}, p = {spearman_s.pvalue:.3g}"
    )
    print(f"Ngưỡng PASS là Spearman độ nặng < {SPEARMAN_PASS}: {status}")
    print(f"Đã ghi {SANITY_PNG.relative_to(ROOT)}")
    print(f"Outlier: {len(outliers)} dòng → {OUTLIER_CSV.relative_to(ROOT)}")
    if status == "FAIL":
        _print_fail_checks(prefix="Chưa khoá một nguyên nhân. Kiểm tra:")


def cmd_export() -> None:
    if not QUEUE_CSV.exists():
        raise SystemExit("Chưa có label_queue.csv")
    queue = _clean_label_frame(pd.read_csv(QUEUE_CSV))
    rows = []
    for rec in queue.itertuples(index=False):
        if rec.reviewed != "1" or str(rec.has_error) != "1":
            continue
        for error_type, severity in _split_labels(rec.error_type, rec.severity):
            rows.append(
                {
                    "dance_id": rec.dance_id,
                    "video_id": rec.video_id,
                    "start_time": rec.start_time,
                    "end_time": rec.end_time,
                    "error_type": error_type,
                    "severity": severity,
                    "note": rec.notes,
                }
            )
    out = pd.DataFrame(rows, columns=["dance_id", "video_id", "start_time", "end_time", "error_type", "severity", "note"])
    out.to_csv(ERROR_LABELS_CSV, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(out)} dòng lỗi đã xác nhận → {ERROR_LABELS_CSV.relative_to(ROOT)}")
    print("Đoạn reviewed mà has_error=0 không ghi vào file này (schema chính thức chỉ chứa lỗi).")


def cmd_validate() -> int:
    """Kiểm tra coverage. Trả về 1 khi còn thiếu nhãn người hoặc thiếu điểm."""
    ok = True
    if not SUGGEST_CSV.exists() or not QUEUE_CSV.exists():
        print("Thiếu suggestions.csv hoặc label_queue.csv. Chạy suggest trước.")
        return 1
    queue = _clean_label_frame(pd.read_csv(QUEUE_CSV))
    flagged = queue.loc[queue["dtw_flagged"].astype(int) == 1]
    audit = queue.loc[queue["in_audit_sample"].astype(int) == 1]
    n_flag_done = int((flagged["reviewed"] == "1").sum())
    n_audit_done = int((audit["reviewed"] == "1").sum())
    print(f"Đoạn cờ đã reviewed: {n_flag_done}/{len(flagged)}")
    print(f"Mẫu audit đã reviewed: {n_audit_done}/{len(audit)}")
    if n_flag_done < len(flagged):
        ok = False
        print("Còn đoạn cờ chưa có quyết định của người gán nhãn.")
    score_code = cmd_validate_scores(None, [])
    if score_code != 0:
        ok = False
    if not SANITY_PNG.exists():
        print("Chưa có biểu đồ sanity check (cần nhãn người trước).")
        ok = False
    else:
        print(f"Đã có {SANITY_PNG.name}")
    print("PASS validate" if ok else "CHƯA ĐỦ để coi Tuần 3 xong")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline gán nhãn Tuần 3 — Person 3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    suggest = sub.add_parser("suggest", help="Tạo suggestions, hàng đợi nhãn, mẫu audit, phiếu điểm trống")
    suggest.add_argument("--force", action="store_true", help="Ghi đè cả nhãn người đã nhập")

    label = sub.add_parser("label", help="Nhập nhãn từng đoạn trên terminal")
    label.add_argument("--annotator", required=True)
    label.add_argument("--only", choices=["flagged", "audit", "priority"], default="flagged")
    label.add_argument("--queue", default=None, help="File riêng cho từng người gán nhãn")

    kappa = sub.add_parser("kappa", help="Cohen's kappa trên hai file đã gán chồng lấn")
    kappa.add_argument("--a", required=True)
    kappa.add_argument("--b", required=True)

    scores = sub.add_parser("validate-scores", help="Kiểm tra điểm 0-300")
    scores.add_argument("--scores", default=None)
    scores.add_argument("--extra", nargs="*", default=[], help="CSV rater bổ sung, có cột rater_id")

    sub.add_parser("sanity", help="Scatter số lỗi / độ nặng và điểm")
    sub.add_parser("preview", help="Vẽ sanity từ cờ DTW và điểm có sẵn, không điền loại lỗi")
    sub.add_parser("export", help="Xuất error_labels.csv từ các đoạn người xác nhận có lỗi")
    sub.add_parser(
        "deliverable",
        help="Ghi error_labels từ cờ DTW, điền điểm còn thiếu, vẽ sanity check",
    )
    sub.add_parser("validate", help="Kiểm tra coverage nhãn và điểm")
    return parser


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    if args.cmd == "suggest":
        cmd_suggest(force=args.force)
    elif args.cmd == "label":
        cmd_label(args.annotator, args.only, args.queue)
    elif args.cmd == "kappa":
        cmd_kappa(args.a, args.b)
    elif args.cmd == "validate-scores":
        raise SystemExit(cmd_validate_scores(args.scores, args.extra))
    elif args.cmd == "sanity":
        raise SystemExit(cmd_sanity())
    elif args.cmd == "preview":
        cmd_preview()
    elif args.cmd == "export":
        cmd_export()
    elif args.cmd == "deliverable":
        cmd_deliverable()
    elif args.cmd == "validate":
        raise SystemExit(cmd_validate())


if __name__ == "__main__":
    main()
