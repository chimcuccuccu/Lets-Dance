"""Validate pose extraction coverage/quality after extract_all."""
from pathlib import Path

import numpy as np

root = Path(r"D:\Study\Đồ án\Lets-Dance")
videos = sorted(root.joinpath("data/dances").rglob("*.mp4"))
poses_root = root / "poses"

missing = []
bad_shape = []
nan_files = []
zero_lower = []
rows = []

newly = {
    "dance_001/D01_P012_T01",
    "dance_003/D03_P007_T01",
    "dance_004/D04_P007_T01",
    "dance_006/D06_P002_T01",
    "dance_006/D06_P012_T01",
    "dance_007/D07_P007_T01",
    "dance_009/D09_P007_T01",
    "dance_009/D09_P012_T01",
    "dance_010/D10_P007_T01",
    "dance_011/D11_P002_T01",
    "dance_012/D12_P002_T01",
    "dance_012/D12_P007_T01",
}

for v in videos:
    dance_id = v.parent.parent.name
    stem = v.stem
    p = poses_root / dance_id / f"{stem}.npy"
    if not p.exists():
        missing.append(f"{dance_id}/{stem}")
        continue
    arr = np.load(p)
    if arr.ndim != 3 or arr.shape[1:] != (33, 4):
        bad_shape.append((str(p), arr.shape))
        continue
    if np.isnan(arr).any():
        nan_files.append(str(p))
    lower = arr[:, 25:33, :3]
    mag = float(np.linalg.norm(lower, axis=-1).mean())
    vis = float(arr[:, :, 3].mean())
    lower_vis = float(arr[:, 25:33, 3].mean())
    if mag < 1e-4:
        zero_lower.append(f"{dance_id}/{stem}")
    key = f"{dance_id}/{stem}"
    if key in newly:
        rows.append(
            {
                "id": key,
                "T": arr.shape[0],
                "vis_mean": vis,
                "lower_vis": lower_vis,
                "lower_mag": mag,
            }
        )

print("=== COVERAGE ===")
print(
    f"videos={len(videos)} missing_pose={len(missing)} "
    f"bad_shape={len(bad_shape)} nan={len(nan_files)} zero_lower={len(zero_lower)}"
)
if missing:
    print("MISSING:", missing)
if bad_shape:
    print("BAD SHAPE:", bad_shape[:5])
if zero_lower:
    print("ZERO LOWER:", zero_lower)

print("\n=== 12 NEWLY EXTRACTED ===")
for r in sorted(rows, key=lambda x: x["id"]):
    flag = []
    if r["lower_mag"] < 0.01:
        flag.append("SUSPECT_LOWER")
    if r["vis_mean"] < 0.2:
        flag.append("LOW_VIS")
    print(
        f"{r['id']}: T={r['T']:4d} vis={r['vis_mean']:.3f} "
        f"lower_vis={r['lower_vis']:.3f} lower_mag={r['lower_mag']:.4f} "
        f"{' '.join(flag)}"
    )

print("\n=== POSES PER DANCE ===")
for d in sorted(poses_root.glob("dance_*")):
    n_ref = len(list(d.glob("dance_*_ref.npy")))
    n_perf = len(list(d.glob("D*_P*.npy")))
    print(f"{d.name}: ref={n_ref} performers={n_perf}")

ok = not missing and not bad_shape and not zero_lower and len(rows) == 12
print("\nSTATUS:", "PASS" if ok else "ISSUES")
