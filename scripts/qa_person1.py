"""
QA Person 1 — kiểm tra pose + preprocess + diff có đúng hướng không.

Chạy:
  python scripts/qa_person1.py
  python scripts/qa_person1.py --dance dance_001 --visualize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess import (  # noqa: E402
    build_diff_sequence,
    get_hip_center,
    mean_abs_diff,
    preprocess_pipeline,
)


def _pose_path(poses: Path, dance_id: str, video_id: str) -> Path:
    return poses / dance_id / f"{video_id}.npy"


def check_coverage(data: Path, poses: Path) -> dict:
    videos = list(data.rglob("*.mp4"))
    missing = []
    for v in videos:
        dance_id = v.parent.parent.name
        p = poses / dance_id / f"{v.stem}.npy"
        if not p.exists():
            missing.append(f"{dance_id}/{v.stem}")
    return {"n_videos": len(videos), "n_missing": len(missing), "missing": missing}


def check_preprocess_invariants(pose: np.ndarray) -> dict:
    out = preprocess_pipeline(pose, target_frames=150)
    hip = get_hip_center(out)
    hip_err = float(np.abs(hip).max())
    finite = bool(np.isfinite(out[:, :, :3]).all())
    # Spine length ~ 1 after mean-spine scale (approx; after smooth may drift a bit)
    ls, rs = out[:, 11, :3], out[:, 12, :3]
    spine = np.linalg.norm((ls + rs) * 0.5, axis=1)
    spine_mean = float(np.mean(spine))
    return {
        "shape": tuple(out.shape),
        "hip_max_abs": hip_err,
        "hip_ok": hip_err < 1e-3,
        "finite": finite,
        "spine_mean": spine_mean,
        "spine_ok": 0.5 < spine_mean < 1.5,
    }


def check_diff_sanity(perf: np.ndarray, ref: np.ndarray) -> dict:
    self_d = build_diff_sequence(perf, perf, target_frames=150)
    vs_ref = build_diff_sequence(perf, ref, target_frames=150)
    # Shape perturbation should increase diff
    bad = perf.copy()
    bad[:, 15, :3] += 0.1
    bad[:, 16, :3] -= 0.08
    vs_bad = build_diff_sequence(perf, bad, target_frames=150)
    return {
        "self_mad": mean_abs_diff(self_d),
        "vs_ref_mad": mean_abs_diff(vs_ref),
        "vs_shape_error_mad": mean_abs_diff(vs_bad),
        "self_ok": mean_abs_diff(self_d) < 1e-5,
        "monotonic_ok": mean_abs_diff(vs_bad) > mean_abs_diff(self_d),
    }


def correlate_diff_vs_score(
    scores_csv: Path,
    poses: Path,
    dance_id: str | None = None,
    max_n: int = 40,
) -> dict:
    df = pd.read_csv(scores_csv)
    if dance_id:
        df = df[df["dance_id"] == dance_id]
    df = df.head(max_n)

    mads, scores = [], []
    skipped = 0
    for _, row in df.iterrows():
        d = str(row["dance_id"])
        vid = str(row["video_id"])
        ref_id = str(row["ref_id"]) if "ref_id" in row and pd.notna(row["ref_id"]) else f"{d}_ref"
        p_path = _pose_path(poses, d, vid)
        r_path = _pose_path(poses, d, ref_id)
        if not p_path.exists() or not r_path.exists():
            skipped += 1
            continue
        perf = np.load(p_path)
        ref = np.load(r_path)
        mad = mean_abs_diff(build_diff_sequence(perf, ref, target_frames=150))
        mads.append(mad)
        scores.append(float(row["tong_diem"]))

    if len(mads) < 3:
        return {"n": len(mads), "skipped": skipped, "pearson": None, "ok": False}

    m = np.asarray(mads)
    s = np.asarray(scores)
    # Kỳ vọng: MAD cao ↔ điểm thấp → correlation âm
    pearson = float(np.corrcoef(m, s)[0, 1])
    return {
        "n": len(mads),
        "skipped": skipped,
        "mad_mean": float(m.mean()),
        "mad_std": float(m.std()),
        "score_mean": float(s.mean()),
        "pearson_mad_vs_score": pearson,
        "ok": pearson < -0.15,  # hướng đúng; dataset auto-label nên không đòi cao
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dance", default=None, help="vd. dance_001 — giới hạn correlation")
    ap.add_argument("--sample", default="dance_001/D01_P003_T01", help="video_id mẫu để check invariants")
    args = ap.parse_args()

    data = ROOT / "data" / "dances"
    poses = ROOT / "poses"
    scores = ROOT / "annotations" / "scores.csv"

    print("=" * 60)
    print("1) COVERAGE (tuần 2 deliverable)")
    cov = check_coverage(data, poses)
    print(f"   videos={cov['n_videos']} missing={cov['n_missing']}")
    print("   →", "PASS" if cov["n_missing"] == 0 else f"FAIL {cov['missing'][:5]}")

    print("\n2) PREPROCESS INVARIANTS")
    dance_id, vid = args.sample.split("/", 1) if "/" in args.sample else ("dance_001", args.sample)
    sample_path = _pose_path(poses, dance_id, vid)
    if not sample_path.exists():
        # fallback: first performer
        sample_path = next(poses.joinpath("dance_001").glob("D*_P*.npy"))
        dance_id = "dance_001"
        vid = sample_path.stem
    pose = np.load(sample_path)
    inv = check_preprocess_invariants(pose)
    print(f"   sample={dance_id}/{vid} raw={pose.shape} → {inv['shape']}")
    print(f"   hip_max_abs={inv['hip_max_abs']:.2e} (kỳ vọng ≈0) → {'PASS' if inv['hip_ok'] else 'FAIL'}")
    print(f"   spine_mean={inv['spine_mean']:.3f} (kỳ vọng ~1) → {'PASS' if inv['spine_ok'] else 'WARN'}")
    print(f"   finite={inv['finite']} → {'PASS' if inv['finite'] else 'FAIL'}")

    print("\n3) DIFF SANITY (self≈0, lệch hình > self)")
    ref_path = _pose_path(poses, dance_id, f"{dance_id}_ref")
    ref = np.load(ref_path)
    ds = check_diff_sanity(pose, ref)
    print(f"   self_mad={ds['self_mad']:.2e} → {'PASS' if ds['self_ok'] else 'FAIL'}")
    print(f"   vs_ref_mad={ds['vs_ref_mad']:.4f}")
    print(f"   vs_shape_error_mad={ds['vs_shape_error_mad']:.4f} → {'PASS' if ds['monotonic_ok'] else 'FAIL'}")

    print("\n4) CORRELATION mean|diff| ↔ tong_diem (hướng âm = đúng)")
    if scores.exists():
        corr = correlate_diff_vs_score(scores, poses, dance_id=args.dance, max_n=60)
        print(f"   n={corr['n']} skipped={corr.get('skipped', 0)}")
        if corr.get("pearson_mad_vs_score") is not None:
            print(
                f"   pearson={corr['pearson_mad_vs_score']:.3f} "
                f"(mad_mean={corr['mad_mean']:.4f}) → "
                f"{'PASS (hướng đúng)' if corr['ok'] else 'WARN (yếu / nhiễu label)'}"
            )
            print("   Ghi chú: nhãn Demo là auto-label → |r| thường chỉ vừa phải.")
        else:
            print("   → SKIP (không đủ mẫu)")
    else:
        print("   → SKIP (không có annotations/scores.csv)")

    print("\n5) CÁCH XEM BẰNG MẮT")
    print("   python src/visualize.py poses/dance_001/D01_P003_T01.npy")
    print("   python src/test_pipeline.py poses/dance_001/D01_P003_T01.npy poses/dance_001/dance_001_ref.npy")
    print("   python src/test_pipeline.py --model   # thêm test Spatial DL (torch)")
    print("   python -m src.preprocessing.build_diffs --scores annotations/scores.csv --poses poses --out poses/diffs")
    print("=" * 60)

    hard_fail = (
        cov["n_missing"] > 0
        or not inv["hip_ok"]
        or not inv["finite"]
        or not ds["self_ok"]
        or not ds["monotonic_ok"]
    )
    print("OVERALL:", "PASS" if not hard_fail else "FAIL")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
