"""
Download AIST++ keypoints3d (+ ignore list) for Spatial pretrain.

Source: https://github.com/google/aistplusplus_dataset/releases/tag/v1.0
Only keypoints3d (~836MB) is required — no videos / SMPL model files.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/google/aistplusplus_dataset/releases/download/v1.0"
ASSETS = {
    "keypoints3d.zip": f"{RELEASE}/keypoints3d.zip",
    "ignore_list.txt": f"{RELEASE}/ignore_list.txt",
}


def _download(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_pct = [-1]

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = int(done * 100 / total_size)
        if pct == last_pct[0] or (pct % 5 and pct != 100):
            return
        last_pct[0] = pct
        print(f"  {pct:3d}%  {done / 1e6:.1f}/{total_size / 1e6:.1f} MB")

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    tmp.replace(dest)
    return dest


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"[extract] {zip_path} → {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    # macOS resource forks sometimes ship in the release zip
    junk = out_dir / "__MACOSX"
    if junk.is_dir():
        import shutil

        shutil.rmtree(junk)
        print("[cleanup] removed __MACOSX")


def main():
    ap = argparse.ArgumentParser(description="Download AIST++ keypoints3d")
    ap.add_argument("--out", default="data/aistpp", help="Annotation root")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--keep-zip",
        action="store_true",
        help="Giữ file zip sau khi giải nén",
    )
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "_raw"
    raw_dir.mkdir(exist_ok=True)

    ignore_path = _download(ASSETS["ignore_list.txt"], root / "ignore_list.txt", args.force)
    print(f"ignore_list → {ignore_path}")

    zip_path = _download(ASSETS["keypoints3d.zip"], raw_dir / "keypoints3d.zip", args.force)
    kp_dir = root / "keypoints3d"
    if kp_dir.is_dir() and any(kp_dir.glob("*.pkl")) and not args.force:
        n = len(list(kp_dir.glob("*.pkl")))
        print(f"[skip] {kp_dir} already has {n} .pkl files")
    else:
        _extract_zip(zip_path, root)
        # Handle zip that nests keypoints3d/ or dumps pkl at root
        if not kp_dir.is_dir():
            nested = list(root.glob("**/keypoints3d"))
            if nested:
                pass  # already correct if extract created it
            else:
                pkls = list(root.glob("*.pkl"))
                if pkls:
                    kp_dir.mkdir(exist_ok=True)
                    for p in pkls:
                        p.rename(kp_dir / p.name)

    n = len(list(kp_dir.glob("*.pkl"))) if kp_dir.is_dir() else 0
    if n == 0:
        print("ERROR: no keypoints3d/*.pkl after extract — check zip layout.", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {n} sequences in {kp_dir}")
    print("Next:")
    print("  python -m src.spatial_dl.pretrain --epochs 20")
    print("  python -m src.spatial_dl.train --official --pretrained checkpoints/spatial_pretrained.pt")

    if not args.keep_zip and zip_path.is_file():
        # Keep zip by default if huge re-download is painful — only delete with explicit flag absence... 
        # Actually default keep zip for safety; user can delete manually.
        pass

    # Tiny fingerprint for sanity
    sample = next(kp_dir.glob("*.pkl"))
    h = hashlib.md5(sample.read_bytes()[:4096]).hexdigest()[:8]
    print(f"Sample {sample.name} md5[:8]={h}")


if __name__ == "__main__":
    main()
