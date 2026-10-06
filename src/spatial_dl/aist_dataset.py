"""
AIST++ → MediaPipe pose dataset for Spatial pretrain.

Expects annotations layout (after ``scripts/download_aistpp.py``)::

    data/aistpp/
      keypoints3d/*.pkl      # COCO-17, keys keypoints3d / keypoints3d_optim
      ignore_list.txt        # optional
      mediapipe_cache/       # auto-built (T,33,4) preprocessed clips

Proxy samples are synthetic ``diff = pose_a − pose_b`` (same genre when possible)
so the encoder sees the same domain as fine-tuning on performer−reference diffs.
"""
from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.preprocess import normalize_length, preprocess_pipeline
from src.spatial_dl.dataset import mirror_pose_diff
from src.spatial_dl.joints_convert import convert_to_mediapipe

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_AIST_ROOT = "data/aistpp"
DEFAULT_SHARED_BASIC = os.getenv('SHARED_DATA_AIST')
DEFAULT_TARGET_FRAMES = 150
DEFAULT_WINDOW_FRAMES = 180  # ~3s @ 60fps before resample

# AIST++ sequence tag → genre id (10 genres)
GENRE_TAGS: Tuple[str, ...] = (
    "gBR", "gPO", "gLO", "gMH", "gLH", "gHO", "gWA", "gKR", "gJS", "gJB",
)
GENRE_TO_ID: Dict[str, int] = {g: i for i, g in enumerate(GENRE_TAGS)}

_SEQ_RE = re.compile(
    r"^(?P<genre>g[A-Z]{2})_.*_d(?P<dancer>\d+)_",
)


def parse_aist_seq_name(seq_name: str) -> Tuple[str, str]:
    """Return (genre_tag, dancer_id). Fallback genre='unk', dancer=seq_name."""
    stem = Path(seq_name).stem
    m = _SEQ_RE.match(stem)
    if not m:
        # Try loose genre prefix
        tag = stem.split("_", 1)[0]
        genre = tag if tag in GENRE_TO_ID else "unk"
        return genre, stem
    return m.group("genre"), m.group("dancer")


def load_ignore_list(aist_root: str | Path) -> set[str]:
    path = Path(aist_root) / "ignore_list.txt"
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.add(Path(name).stem)
    return names


def list_keypoint3d_sequences(aist_root: str | Path) -> List[str]:
    root = Path(aist_root)
    kp_dir = root / "keypoints3d"
    if not kp_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {kp_dir}. Run: python scripts/download_aistpp.py"
        )
    ignore = load_ignore_list(root)
    seqs = sorted(
        p.stem
        for p in kp_dir.glob("*.pkl")
        if p.stem not in ignore and not p.name.startswith("._")
    )
    if not seqs:
        raise FileNotFoundError(f"No .pkl sequences in {kp_dir}")
    return seqs


def load_aist_keypoint3d(
    aist_root: str | Path,
    seq_name: str,
    *,
    use_optim: bool = True,
) -> np.ndarray:
    """Load COCO-17 keypoints (T, 17, 3) from official AIST++ pickle."""
    path = Path(aist_root) / "keypoints3d" / f"{Path(seq_name).stem}.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    key = "keypoints3d_optim" if use_optim and "keypoints3d_optim" in data else "keypoints3d"
    kp = np.asarray(data[key], dtype=np.float32)
    if kp.ndim != 3 or kp.shape[1] != 17:
        raise ValueError(f"{path}: expected (T,17,3), got {kp.shape}")
    return kp[:, :, :3]


def aist_to_mediapipe_pose(
    coco_joints: np.ndarray,
    *,
    target_frames: int = -1,
    smooth_window: int = 7,
) -> np.ndarray:
    """COCO-17 → MediaPipe (T,33,4) → Person-1 preprocess (optional resample)."""
    pose = convert_to_mediapipe(coco_joints, fmt="coco17")
    return preprocess_pipeline(
        pose,
        target_frames=target_frames,
        visibility_threshold=0.5,
        scale_method="spine",
        smooth_window=smooth_window,
        apply_confidence_filter=True,
    )


class AISTPoseCache:
    """Lazy convert+preprocess cache on disk (one .npy per sequence)."""

    def __init__(
        self,
        aist_root: str = DEFAULT_AIST_ROOT,
        cache_dir: str = "",
        use_optim: bool = True,
        smooth_window: int = 7,
    ):
        self.aist_root = str(aist_root)
        self.cache_dir = cache_dir or os.path.join(self.aist_root, "mediapipe_cache")
        self.use_optim = use_optim
        self.smooth_window = smooth_window
        os.makedirs(self.cache_dir, exist_ok=True)

    def path_for(self, seq_name: str) -> str:
        return os.path.join(self.cache_dir, f"{Path(seq_name).stem}.npy")

    def get(self, seq_name: str) -> np.ndarray:
        out = self.path_for(seq_name)
        if os.path.isfile(out):
            return np.load(out).astype(np.float32)

        coco = load_aist_keypoint3d(
            self.aist_root, seq_name, use_optim=self.use_optim
        )
        pose = aist_to_mediapipe_pose(
            coco, target_frames=-1, smooth_window=self.smooth_window
        )
        np.save(out, pose.astype(np.float32))
        return pose.astype(np.float32)

    def build_all(self, sequences: Optional[Sequence[str]] = None) -> int:
        seqs = list(sequences) if sequences is not None else list_keypoint3d_sequences(
            self.aist_root
        )
        for i, name in enumerate(seqs, 1):
            self.get(name)
            if i % 50 == 0 or i == len(seqs):
                logger.info("Cached %d/%d MediaPipe poses", i, len(seqs))
        return len(seqs)


def _sample_window(
    pose: np.ndarray,
    window_frames: int,
    target_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    xyz = pose[:, :, :3].astype(np.float32)
    T = xyz.shape[0]
    if T <= 1:
        return normalize_length(xyz, target_frames)
    win = min(window_frames, T)
    start = int(rng.integers(0, T - win + 1)) if T > win else 0
    return normalize_length(xyz[start : start + win], target_frames)


class AISTPretrainDataset(Dataset):
    """
    Modes
    -----
    synthetic_diff : diff of two windows (same genre preferred) — default
    pose_ae        : single window (autoencoder reconstructs pose)
    """

    def __init__(
        self,
        aist_root: str = DEFAULT_AIST_ROOT,
        sequences: Optional[Sequence[str]] = None,
        mode: str = "synthetic_diff",
        target_frames: int = DEFAULT_TARGET_FRAMES,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        use_optim: bool = True,
        augment: bool = True,
        seed: int = 42,
        max_sequences: int = 0,
        preloaded_poses: Optional[Sequence[np.ndarray]] = None,
    ):
        if mode not in ("synthetic_diff", "pose_ae"):
            raise ValueError(f"Unsupported mode={mode!r}")

        self.mode = mode
        self.target_frames = target_frames
        self.window_frames = window_frames
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.cache = AISTPoseCache(aist_root=aist_root, use_optim=use_optim)
        seqs = list(sequences) if sequences is not None else list_keypoint3d_sequences(
            aist_root
        )
        if max_sequences and max_sequences > 0:
            seqs = seqs[:max_sequences]

        self.sequences = seqs
        self.genres = [parse_aist_seq_name(s)[0] for s in seqs]
        self.dancers = [parse_aist_seq_name(s)[1] for s in seqs]

        self._genre_to_indices: Dict[str, List[int]] = {}
        for i, g in enumerate(self.genres):
            self._genre_to_indices.setdefault(g, []).append(i)

        if preloaded_poses is not None:
            if len(preloaded_poses) != len(self.sequences):
                raise ValueError(
                    f"preloaded_poses ({len(preloaded_poses)}) != sequences ({len(self.sequences)})"
                )
            self._poses = [np.asarray(p, dtype=np.float32) for p in preloaded_poses]
        else:
            # Keep poses in RAM — avoids re-reading long .npy every __getitem__
            self._poses = [self.cache.get(s) for s in self.sequences]

        logger.info(
            "AISTPretrainDataset: n=%d mode=%s genres=%d root=%s (RAM cached)",
            len(self.sequences),
            mode,
            len(self._genre_to_indices),
            aist_root,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def _aug(self, x: np.ndarray) -> np.ndarray:
        if not self.augment:
            return x
        out = x.astype(np.float32).copy()
        t = out.shape[0]
        if self.rng.random() < 0.5:
            out = mirror_pose_diff(out)
        if t > 8 and self.rng.random() < 0.5:
            shift = int(self.rng.integers(-max(1, t // 10), max(2, t // 10)))
            out = np.roll(out, shift, axis=0)
        if self.rng.random() < 0.5:
            out = out * float(self.rng.uniform(0.9, 1.1))
        if self.rng.random() < 0.6:
            out = out + self.rng.normal(0.0, 0.015, size=out.shape).astype(np.float32)
        if self.rng.random() < 0.35:
            n_j = int(self.rng.integers(1, 5))
            joints = self.rng.choice(out.shape[1], size=n_j, replace=False)
            a = int(self.rng.integers(0, max(1, t - 2)))
            b = int(self.rng.integers(a + 1, t))
            out[a:b, joints] = 0.0
        return out.astype(np.float32)

    def _partner_index(self, idx: int) -> int:
        g = self.genres[idx]
        pool = self._genre_to_indices.get(g, [])
        if len(pool) >= 2:
            choices = [j for j in pool if j != idx]
            return int(choices[self.rng.integers(0, len(choices))])
        # Fallback: any other sequence
        if len(self.sequences) == 1:
            return idx
        j = idx
        while j == idx:
            j = int(self.rng.integers(0, len(self.sequences)))
        return j

    def __getitem__(self, idx: int):
        pose_a = self._poses[idx]
        win_a = _sample_window(
            pose_a, self.window_frames, self.target_frames, self.rng
        )

        if self.mode == "pose_ae":
            x = self._aug(win_a)
            return torch.from_numpy(x), torch.from_numpy(x.copy())

        pose_b = self._poses[self._partner_index(idx)]
        win_b = _sample_window(
            pose_b, self.window_frames, self.target_frames, self.rng
        )
        diff = (win_a - win_b).astype(np.float32)
        diff = self._aug(diff)
        return torch.from_numpy(diff), torch.from_numpy(diff.copy())


def split_sequences_by_dancer(
    sequences: Sequence[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Hold out whole dancers — closer to person_id split used in fine-tune."""
    rng = np.random.default_rng(seed)
    dancer_to_seqs: Dict[str, List[str]] = {}
    for s in sequences:
        _, dancer = parse_aist_seq_name(s)
        dancer_to_seqs.setdefault(dancer, []).append(s)

    dancers = sorted(dancer_to_seqs)
    rng.shuffle(dancers)
    n_val = max(1, int(round(len(dancers) * val_ratio))) if len(dancers) > 1 else 0
    val_dancers = set(dancers[:n_val])
    train, val = [], []
    for d, seqs in dancer_to_seqs.items():
        (val if d in val_dancers else train).extend(seqs)
    if not train or not val:
        cut = max(1, int(len(sequences) * (1 - val_ratio)))
        train = list(sequences[:cut])
        val = list(sequences[cut:]) or list(sequences[-1:])
    return train, val


def _pack_xy_vis(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(T,33,2) + (T,33) → (T,33,4) with z=0."""
    t = xy.shape[0]
    if vis.shape != (t, 33):
        vis = np.ones((t, 33), dtype=np.float32)
    z0 = np.zeros((t, 33, 1), dtype=np.float32)
    return np.concatenate(
        [xy[:, :, :2].astype(np.float32), z0, vis[:, :, None].astype(np.float32)],
        axis=2,
    )


def npz_to_mediapipe_pose(path: str | Path) -> np.ndarray:
    """
    dance_coach ``shared_data`` npz → MediaPipe (T, 33, 4).

    Supported layouts:
      - ``xy`` + ``vis`` — normalized image coords
      - ``tensor`` (T,33,6) + ``conf`` — take first 2 channels as xy
      - ``pose`` (T,33,4+) — first 4 channels
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)

        if "pose" in files:
            pose = np.asarray(z["pose"], dtype=np.float32)
            if pose.ndim != 3 or pose.shape[1] != 33:
                raise ValueError(f"{path}: bad pose shape {pose.shape}")
            if pose.shape[2] >= 4:
                return pose[:, :, :4].copy()
            if pose.shape[2] == 3:
                vis = np.ones((pose.shape[0], 33, 1), dtype=np.float32)
                return np.concatenate([pose, vis], axis=2)
            if pose.shape[2] == 2:
                vis = np.ones((pose.shape[0], 33), dtype=np.float32)
                return _pack_xy_vis(pose, vis)
            raise ValueError(f"{path}: unsupported pose C={pose.shape[2]}")

        if "xy" in files:
            xy = np.asarray(z["xy"], dtype=np.float32)
            if xy.ndim != 3 or xy.shape[1] != 33 or xy.shape[2] < 2:
                raise ValueError(f"{path}: bad xy shape {xy.shape}")
            vis = (
                np.asarray(z["vis"], dtype=np.float32)
                if "vis" in files
                else np.ones((xy.shape[0], 33), dtype=np.float32)
            )
            return _pack_xy_vis(xy, vis)

        if "tensor" in files:
            # Older export: (T, 33, 6) — channels 0:2 are xy (image or world-ish)
            ten = np.asarray(z["tensor"], dtype=np.float32)
            if ten.ndim != 3 or ten.shape[1] != 33 or ten.shape[2] < 2:
                raise ValueError(f"{path}: bad tensor shape {ten.shape}")
            xy = ten[:, :, :2]
            if ten.shape[2] >= 3 and float(np.abs(ten[:, :, 2]).mean()) > 1e-6:
                # Real z available — pack xyz + conf/vis
                zc = ten[:, :, 2:3]
                vis = (
                    np.asarray(z["conf"], dtype=np.float32)
                    if "conf" in files
                    else np.ones((ten.shape[0], 33), dtype=np.float32)
                )
                if vis.shape != (ten.shape[0], 33):
                    vis = np.ones((ten.shape[0], 33), dtype=np.float32)
                return np.concatenate([xy, zc, vis[:, :, None]], axis=2)
            vis = (
                np.asarray(z["conf"], dtype=np.float32)
                if "conf" in files
                else np.ones((ten.shape[0], 33), dtype=np.float32)
            )
            return _pack_xy_vis(xy, vis)

        raise ValueError(f"{path}: need keys xy/vis, tensor/conf, or pose; got {z.files}")


def list_shared_aist_npz(
    shared_dir: str | Path = DEFAULT_SHARED_BASIC,
    *,
    camera: str = "all",
    skip_numbered_variants: bool = True,
) -> List[Path]:
    """
    List AIST npz under dance_coach ``shared_data/.../Basic Dance``.

    camera: 'all' | 'c01' | 'c02' | ...
    skip_numbered_variants: drop ``*.1.npz`` duplicate exports
    """
    root = Path(shared_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"shared_data AIST dir không tồn tại: {root}"
        )
    files = sorted(root.glob("*.npz"))
    out: List[Path] = []
    for p in files:
        name = p.name
        if not re.match(r"^g[A-Z]{2}_", name):
            continue
        if skip_numbered_variants and re.search(r"\.\d+\.npz$", name):
            continue
        if camera != "all":
            tag = camera if camera.startswith("c") else f"c{camera}"
            if f"_{tag}_" not in name:
                continue
        out.append(p)
    if not out:
        raise FileNotFoundError(f"Không thấy AIST npz trong {root} (camera={camera})")
    return out


class SharedAISTPoseCache:
    """Preprocess shared_data npz → local npy cache (không ghi vào shared_data)."""

    def __init__(
        self,
        shared_dir: str = DEFAULT_SHARED_BASIC,
        cache_dir: str = "",
        smooth_window: int = 7,
    ):
        self.shared_dir = str(shared_dir)
        self.cache_dir = cache_dir or os.path.join(
            DEFAULT_AIST_ROOT, "shared_basic_cache"
        )
        self.smooth_window = smooth_window
        os.makedirs(self.cache_dir, exist_ok=True)

    def path_for(self, npz_path: str | Path) -> str:
        return os.path.join(self.cache_dir, f"{Path(npz_path).stem}.npy")

    def get(self, npz_path: str | Path) -> np.ndarray:
        out = self.path_for(npz_path)
        if os.path.isfile(out):
            return np.load(out).astype(np.float32)
        raw = npz_to_mediapipe_pose(npz_path)
        # Drop empty / failed extractions
        if not np.isfinite(raw[:, :, :2]).any() or float(np.abs(raw[:, :, :2]).mean()) < 1e-8:
            raise ValueError(f"{npz_path}: empty pose")
        pose = preprocess_pipeline(
            raw,
            target_frames=-1,
            visibility_threshold=0.5,
            scale_method="spine",
            smooth_window=self.smooth_window,
            apply_confidence_filter=True,
        )
        np.save(out, pose.astype(np.float32))
        return pose.astype(np.float32)

    def build_all(self, paths: Sequence[str | Path]) -> List[Path]:
        """Cache convertible files; skip corrupt/unsupported. Returns ok paths."""
        ok: List[Path] = []
        skipped = 0
        for i, p in enumerate(paths, 1):
            p = Path(p)
            try:
                self.get(p)
                ok.append(p)
            except Exception as exc:  # noqa: BLE001 — keep going over mixed exports
                skipped += 1
                if skipped <= 8:
                    logger.warning("Skip %s (%s)", p.name, exc)
            if i % 100 == 0 or i == len(paths):
                logger.info("Shared cache %d/%d (ok=%d skip=%d)", i, len(paths), len(ok), skipped)
        if not ok:
            raise RuntimeError(f"Không cache được file nào từ {self.shared_dir}")
        if skipped:
            logger.warning("Skipped %d/%d shared npz", skipped, len(paths))
        return ok


def get_aist_dataloaders(
    aist_root: str = DEFAULT_AIST_ROOT,
    batch_size: int = 32,
    mode: str = "synthetic_diff",
    target_frames: int = DEFAULT_TARGET_FRAMES,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    max_sequences: int = 0,
    build_cache: bool = True,
    source: str = "keypoints3d",
    shared_dir: str = DEFAULT_SHARED_BASIC,
    camera: str = "all",
):
    """
    source:
      - ``keypoints3d``: official AIST++ COCO-17 under ``aist_root``
      - ``shared``: dance_coach ``shared_data/.../Basic Dance`` MediaPipe npz
    """
    source = source.lower().strip()

    def _load_shared_bundle(apply_max: bool = True):
        paths = list_shared_aist_npz(shared_dir, camera=camera)
        if apply_max and max_sequences and max_sequences > 0:
            paths = paths[:max_sequences]
        cache = SharedAISTPoseCache(shared_dir=shared_dir)
        if build_cache:
            paths = cache.build_all(paths)
        else:
            kept = []
            for p in paths:
                try:
                    cache.get(p)
                    kept.append(Path(p))
                except Exception:  # noqa: BLE001
                    continue
            paths = kept
        names = [f"shared::{p.stem}" for p in paths]
        poses = [cache.get(p) for p in paths]
        return names, poses

    def _load_kpt3d_bundle():
        seqs = list_keypoint3d_sequences(aist_root)
        if max_sequences and max_sequences > 0:
            seqs = seqs[:max_sequences]
        cache = AISTPoseCache(aist_root=aist_root)
        if build_cache:
            cache.build_all(seqs)
        names = [f"kpt3d::{s}" for s in seqs]
        poses = [cache.get(s) for s in seqs]
        return names, poses

    if source in ("shared", "shared_npz", "shared_data"):
        seq_names, all_poses = _load_shared_bundle()
        logger.info("Shared AIST: dir=%s camera=%s n=%d", shared_dir, camera, len(seq_names))
    elif source in ("mix", "mixed", "shared+keypoints3d"):
        # Mix: shared (khuyến nghị --camera c01) + keypoints3d 3D
        n1, p1 = _load_shared_bundle(apply_max=False)
        seqs = list_keypoint3d_sequences(aist_root)
        cache3 = AISTPoseCache(aist_root=aist_root)
        if build_cache:
            cache3.build_all(seqs)
        n2 = [f"kpt3d::{s}" for s in seqs]
        p2 = [cache3.get(s) for s in seqs]
        seq_names = n1 + n2
        all_poses = p1 + p2
        if max_sequences and max_sequences > 0:
            seq_names = seq_names[:max_sequences]
            all_poses = all_poses[:max_sequences]
        logger.info(
            "Mix AIST: shared=%d kpt3d=%d total=%d camera=%s",
            len(n1),
            len(n2),
            len(seq_names),
            camera,
        )
    else:
        seq_names, all_poses = _load_kpt3d_bundle()

    pose_by_name = dict(zip(seq_names, all_poses))
    train_seqs, val_seqs = split_sequences_by_dancer(
        seq_names, val_ratio=val_ratio, seed=seed
    )

    def _make(seqs: List[str], augment: bool) -> "AISTPretrainDataset":
        # strip prefix for genre/dancer parse — parse_aist_seq_name uses stem
        plain = [s.split("::", 1)[-1] for s in seqs]
        poses = [pose_by_name[s] for s in seqs]
        return AISTPretrainDataset(
            aist_root=aist_root,
            sequences=plain,
            mode=mode,
            target_frames=target_frames,
            window_frames=window_frames,
            augment=augment,
            seed=seed,
            preloaded_poses=poses,
        )

    train_ds = _make(train_seqs, True)
    val_ds = _make(val_seqs, False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    logger.info(
        "AIST loaders: train=%d val=%d batch=%d source=%s",
        len(train_ds),
        len(val_ds),
        batch_size,
        source,
    )
    return train_loader, val_loader
