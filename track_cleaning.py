"""
track_cleaning.py

Active-learning pipeline for Andaman reef fish identification.

Combines CLI/GUI path selection, per-track mass labeling, hierarchical
multi-head classification (species / genus / family), on-disk crop
storage, and periodic GPU retraining on a background thread — all
wrapped in a Qt-based viewer with zoom/pan for fine-grained ID.

Features:
  • CLI args (-c/-l/-f) with Qt file/folder dialogs as fallback for any
    input not supplied on the command line.
  • Loads a TaxonomicMultiHead checkpoint (produced by train.py or by a
    previous session) and refines it in place. Predictions from frame one.
  • Label one representative frame per track; the label is propagated to
    every unlabeled row sharing that Track ID.
  • A `true_id` identity column: rows sharing a `true_id` are the same
    physical fish. Merge broken tracks by editing True ID in the UI;
    split switched tracks via the Split-here button or automatic review.
  • Automatic detection of probable ID swaps using per-track motion /
    size discontinuity scoring, plus a review pass that walks the user
    through the top candidates with side-by-side crop previews.
  • Auto-suggestion from the current multi-head model with softmax
    confidences shown in the UI. Press Enter on an empty field to accept,
    type a name to override.
  • Zoomable, pannable viewer (scroll wheel to zoom under cursor, drag to
    pan, double-click to toggle fit ↔ 100%, Ctrl+0 / Ctrl+1 / Ctrl+= /
    Ctrl+- shortcuts) for resolving ambiguous fish.
  • Crops of every confirmed label are written to disk under
    ./output/labeled_fish_crops/<species>/ for use as future training data.
  • Every RETRAIN_INTERVAL confirmations, and once more at session end,
    the model is retrained on the on-disk crop corpus. Training runs on a
    QThread worker so the UI stays responsive and streams epoch/loss
    progress to the status bar. Label-map indices are preserved across
    retrains, so trained head rows are never silently reassigned.
  • Durable checkpointing: label maps are persisted to a JSON sidecar
    (independent of the .pth) and every save rotates the previous
    generation into ./models/checkpoints/backups/ before overwriting.
    Writes are atomic (tmp + fsync + os.replace) so a crash mid-save
    cannot corrupt the live checkpoint.
  • Labeled CSV is flushed at every milestone and on exit, preserving the
    input's subfolder structure under ./output/tracks/.
  • Family resolution uses the same layered resolver as train.py: checklist,
    optional secondary CSV, then GBIF, with a shared on-disk cache so the
    second run is offline-instant.

Usage:
    python track_cleaning.py
    python track_cleaning.py --csv tracks.csv --checklist checklist.csv --frames-dir frames/

    Any of --csv/-c, --checklist/-l, --frames-dir/-f omitted on the command
    line falls back to a Qt file/folder selection dialog.

Inputs:
    --csv/-c                   Tracking CSV (frame, id, x1, y1, x2, y2).
    --checklist/-l             Andaman checklist CSV (species, genus, family).
    --frames-dir/-f            Parent folder containing the video's frames.
    --secondary-taxonomy       Optional broader taxonomy CSV for families.
    --gbif-cache               JSON cache for GBIF results (shared with train.py).

Outputs:
    ./output/tracks/...           Labeled CSV, mirroring the annotated_videos
                                  subfolder layout of the input.
    ./output/labeled_fish_crops/  Per-species crop folders used as the training set.
    ./models/checkpoints/         Multi-head checkpoint (fish-classifier-1.pth)
                                  plus its label-map sidecar and rotated backups.

Requires:
    pip install PySide6
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut, QPainter
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QLabel,
        QLineEdit,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QStatusBar,
        QFileDialog,
        QGraphicsView,
        QGraphicsScene,
        QGraphicsPixmapItem,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PySide6 is required for the GUI. Install it with: pip install PySide6"
    ) from exc

from model import TaxonomicMultiHead
from taxonomy import TaxonomyResolver, canonical_species

# ==========================================
# Configuration
# ==========================================
OUTPUT_CROP_DIR = "./output/labeled_fish_crops"
CHECKPOINT_DIR = "./models/checkpoints"
MULTIHEAD_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "fish-classifier-1.pth")
LABEL_MAP_PATH = os.path.join(CHECKPOINT_DIR, "fish-classifier-labelmaps.json")
BACKUP_DIR = os.path.join(CHECKPOINT_DIR, "backups")
KEEP_BACKUPS = 10
BACKBONE_PATH = "./models/fish-classifier-0.pth"
CSV_OUT = "./output/tracks"
DEFAULT_GBIF_CACHE = "./output/.gbif_cache.json"

RETRAIN_INTERVAL = 150
BATCH_SIZE = 32
EPOCHS_PER_RETRAIN = 5
LR = 1e-4
LAMBDA_GENUS = 0.3
LAMBDA_FAMILY = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ==========================================
# Path Resolution Helper
# ==========================================
def get_output_csv_path(input_csv_path, output_base_dir):
    """
    Output path inside output_base_dir preserving the subfolder structure
    starting after 'annotated_videos'.
    """
    abs_input = Path(input_csv_path).resolve()
    parts = abs_input.parts
    if "annotated_videos" in parts:
        idx = parts.index("annotated_videos")
        relative_path = Path(*parts[idx + 1 :])
    else:
        relative_path = Path(abs_input.name)
    target_path = Path(output_base_dir) / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


# ==========================================
# 1. CLI args / Qt path selection
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Active-learning pipeline for Andaman reef fish identification. "
        "Any input not given on the command line falls back to a file/folder dialog."
    )
    parser.add_argument(
        "-c",
        "--csv",
        dest="csv_path",
        type=str,
        default=None,
        help="Path to the tracking CSV file.",
    )
    parser.add_argument(
        "-l",
        "--checklist",
        dest="checklist_path",
        type=str,
        default=None,
        help="Path to the Andaman checklist CSV.",
    )
    parser.add_argument(
        "-f",
        "--frames-dir",
        dest="root_data_dir",
        type=str,
        default=None,
        help="Path to the parent folder containing the frames.",
    )
    parser.add_argument(
        "--secondary-taxonomy",
        default=None,
        help="Optional broader taxonomy CSV used before GBIF.",
    )
    parser.add_argument(
        "--gbif-cache",
        default=DEFAULT_GBIF_CACHE,
        help="JSON cache for GBIF results. Share with train.py.",
    )
    return parser.parse_args()


def get_user_paths(csv_path=None, checklist_path=None, root_data_dir=None):
    """
    Resolves the three required inputs. Any value already supplied (e.g. via
    CLI args) is used as-is after validation; anything missing is prompted
    for via a Qt file/folder dialog. QApplication must exist before calling.
    """
    if csv_path:
        if not os.path.isfile(csv_path):
            raise SystemExit(f"CSV not found: {csv_path}")
    else:
        csv_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Tracking CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_path:
            raise SystemExit("No CSV selected.")

    if checklist_path:
        if not os.path.isfile(checklist_path):
            raise SystemExit(f"Checklist not found: {checklist_path}")
    else:
        checklist_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Andaman Checklist CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not checklist_path:
            raise SystemExit("No checklist selected.")

    if root_data_dir:
        if not os.path.isdir(root_data_dir):
            raise SystemExit(f"Frames directory not found: {root_data_dir}")
    else:
        root_data_dir = QFileDialog.getExistingDirectory(
            None, "Select Parent Folder Containing Sites / Frames Data"
        )
        if not root_data_dir:
            raise SystemExit("No frames directory selected.")

    return csv_path, checklist_path, root_data_dir


# ==========================================
# 2. Helpers
# ==========================================
def build_frame_index(root_dir):
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                index[f] = os.path.join(dirpath, f)
    return index


def locate_frame_path(index, frame_num):
    try:
        n = int(frame_num)
    except (ValueError, TypeError):
        return None
    for candidate in (f"frame{n:06d}.jpg", f"frame{n}.jpg", f"_{n:04d}.jpg"):
        if candidate in index:
            return index[candidate]
    for name, path in index.items():
        if name.endswith(f"frame{n:06d}.jpg") or name.endswith(f"frame{n}.jpg"):
            return path
    return None


def prewarm_resolver_from_existing_crops(resolver):
    """
    Resolve every genus already present under OUTPUT_CROP_DIR before the GUI
    opens, so any GBIF lookups happen during startup (which is already slow)
    rather than mid-labeling on the Qt main thread.  Usually a no-op because
    the cache was written at the end of the previous session.
    """
    root = Path(OUTPUT_CROP_DIR)
    if not root.exists():
        return
    genera = set()
    for d in root.iterdir():
        if not d.is_dir():
            continue
        _, genus, _ = resolver.resolve(d.name)  # may hit GBIF; populates cache
        if genus and genus.lower() not in ("unknown", "unidentified"):
            genera.add(genus)
    if genera:
        resolver.prewarm(genera, batch_log_every=25)


# ==========================================
# 3. On-disk crop dataset
# ==========================================
class FishCropDataset(Dataset):
    def __init__(self, crop_dir, resolver, sp_to_idx, gn_to_idx, fa_to_idx, transform):
        self.transform = transform
        self.samples = []
        root = Path(crop_dir)
        if not root.exists():
            return
        for species_dir in sorted(root.iterdir()):
            if not species_dir.is_dir():
                continue
            sp_key, genus, family = resolver.resolve(species_dir.name)
            if (
                sp_key not in sp_to_idx
                or genus not in gn_to_idx
                or family not in fa_to_idx
            ):
                continue
            for img_path in species_dir.glob("*.jpg"):
                self.samples.append(
                    (img_path, sp_to_idx[sp_key], gn_to_idx[genus], fa_to_idx[family])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sp, gn, fa = self.samples[idx]
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            transformed_img = self.transform(img_rgb)
        return transformed_img, sp, gn, fa


def build_label_maps(crop_dir, resolver, prev_sp=None, prev_gn=None, prev_fa=None):
    """
    Scan `crop_dir` and produce the three label maps.

    If previous maps are supplied, ALL previously-known classes keep their
    indices (even those with no crops on disk right now), and new classes
    found in `crop_dir` are appended alphabetically.  A naive re-sort would
    reassign species → index between retrains, silently pointing the head
    weights at the wrong species.
    """
    sp_set, gn_set, fa_set = set(), set(), set()
    root = Path(crop_dir)
    if root.exists():
        for species_dir in sorted(root.iterdir()):
            if not species_dir.is_dir():
                continue
            sp_key, genus, family = resolver.resolve(species_dir.name)
            if not sp_key:
                continue
            sp_set.add(sp_key)
            gn_set.add(genus)
            fa_set.add(family)

    def stable(existing, keys):
        existing = existing or {}
        out = {}
        for k, i in sorted(existing.items(), key=lambda kv: kv[1]):
            out[k] = i
        next_idx = (max(out.values()) + 1) if out else 0
        for k in sorted(keys):
            if k not in out:
                out[k] = next_idx
                next_idx += 1
        return out

    return (stable(prev_sp, sp_set), stable(prev_gn, gn_set), stable(prev_fa, fa_set))


def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(total / (num_classes * max(c, 1)))
    w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    return torch.clamp(w, max=10.0)


# ==========================================
# 3b. Track discontinuity detection (probable ID swaps)
# ==========================================
def _box_center(row):
    return ((row["x1"] + row["x2"]) / 2.0, (row["y1"] + row["y2"]) / 2.0)


def detect_track_discontinuities(df, *, velocity_window=5, max_pairs=500):
    """
    For every consecutive-row pair within the same track `id`, compute a
    continuity score. A high score means the transition looks like a jump —
    an occlusion-triggered identity swap, a track break, or a merged
    detection — rather than a fish smoothly continuing to swim.

    Returns a DataFrame sorted by score desc, capped at `max_pairs`, with
    columns: id, prev_idx, curr_idx, prev_frame, curr_frame, frame_gap,
    pos_jump_pf, vel_angle_deg, area_ratio, speed_ratio, score,
    true_id_prev, true_id_curr.
    """
    rows = []
    for track_id, grp in df.groupby("id", sort=False):
        grp = grp.sort_values("frame")
        idxs = grp.index.tolist()
        if len(idxs) < 2:
            continue

        cxs, cys, areas, frames = [], [], [], []
        for i in idxs:
            r = df.loc[i]
            cx, cy = _box_center(r)
            cxs.append(cx)
            cys.append(cy)
            areas.append(max(1.0, float((r["x2"] - r["x1"]) * (r["y2"] - r["y1"]))))
            frames.append(int(r["frame"]))

        for k in range(1, len(idxs)):
            prev_idx, curr_idx = idxs[k - 1], idxs[k]
            f_prev, f_curr = frames[k - 1], frames[k]
            f_gap = max(1, f_curr - f_prev)
            dx = cxs[k] - cxs[k - 1]
            dy = cys[k] - cys[k - 1]
            dist = (dx * dx + dy * dy) ** 0.5
            pos_jump_pf = dist / f_gap

            # Recent velocity from up to `velocity_window` prior transitions.
            lo = max(0, k - velocity_window)
            hist = []
            for j in range(lo + 1, k):
                fj = max(1, frames[j] - frames[j - 1])
                hx = (cxs[j] - cxs[j - 1]) / fj
                hy = (cys[j] - cys[j - 1]) / fj
                hist.append((hx, hy))
            if hist:
                vx_prev = sum(h[0] for h in hist) / len(hist)
                vy_prev = sum(h[1] for h in hist) / len(hist)
            else:
                vx_prev = vy_prev = 0.0

            vx_curr = dx / f_gap
            vy_curr = dy / f_gap

            # Direction change (deg, 0 = perfectly consistent).
            n1 = (vx_prev**2 + vy_prev**2) ** 0.5
            n2 = (vx_curr**2 + vy_curr**2) ** 0.5
            if n1 > 1e-6 and n2 > 1e-6:
                cos_a = max(
                    -1.0, min(1.0, (vx_prev * vx_curr + vy_prev * vy_curr) / (n1 * n2))
                )
                vel_angle_deg = float(np.degrees(np.arccos(cos_a)))
            else:
                vel_angle_deg = 0.0

            area_ratio = areas[k] / areas[k - 1]
            speed_ratio = n2 / (n1 + 1e-6)

            jump_score = pos_jump_pf
            angle_score = vel_angle_deg / 180.0
            area_score = abs(float(np.log(max(area_ratio, 1e-6))))
            speed_score = (
                abs(float(np.log(max(speed_ratio, 1e-6)))) if n1 > 1e-3 else 0.0
            )
            # Longer gaps are riskier: identity may have been re-acquired on a
            # different fish after an occlusion. Small weight, but it lifts
            # long-gap candidates above trivially small jitter.
            gap_score = float(np.log1p(f_gap)) / 5.0

            score = (
                1.0 * jump_score
                + 15.0 * angle_score
                + 8.0 * area_score
                + 4.0 * speed_score
                + 5.0 * gap_score
            )

            rows.append(
                {
                    "id": int(track_id),
                    "prev_idx": int(prev_idx),
                    "curr_idx": int(curr_idx),
                    "prev_frame": int(f_prev),
                    "curr_frame": int(f_curr),
                    "frame_gap": int(f_gap),
                    "pos_jump_pf": float(pos_jump_pf),
                    "vel_angle_deg": float(vel_angle_deg),
                    "area_ratio": float(area_ratio),
                    "speed_ratio": float(speed_ratio),
                    "true_id_prev": df.at[prev_idx, "true_id"],
                    "true_id_curr": df.at[curr_idx, "true_id"],
                    "score": float(score),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values("score", ascending=False).head(max_pairs).reset_index(drop=True)
    )


# ==========================================
# 4. Retraining  (runs on a worker thread)
# ==========================================
def _assert_no_class_shrink(prev, new, name):
    """
    Belt-and-suspenders guard: a retrain must never drop a class that the
    previous label map already knew about.  If this fires, the caller passed
    an empty/incomplete prev_* map and a head would silently be truncated.
    """
    if not prev:
        return
    missing = set(prev) - set(new)
    if missing:
        raise RuntimeError(
            f"{name} label map lost {len(missing)} classes during retrain "
            f"(e.g. {sorted(missing)[:5]}). Refusing to shrink the head."
        )


def retrain(
    model,
    resolver,
    epochs=EPOCHS_PER_RETRAIN,
    log=print,
    prev_sp=None,
    prev_gn=None,
    prev_fa=None,
):
    """
    Full retrain pass.  `log` is a callable that receives status strings so
    the caller (Qt worker) can route them into the UI instead of stdout.

    `prev_*` maps preserve class indices across retrains so a growing label
    set never reshuffles the head weights.  Samples whose family is
    'Unknown_Family' are excluded from the family loss — the head still has
    a slot for that class, but it's never trained to predict it.

    NOTE: model.update_heads() must *grow* the existing Linear layers in
    place (preserving weights for already-known indices) rather than
    rebuilding them.  See model.py for the corresponding patch.
    """
    sp_to_idx, gn_to_idx, fa_to_idx = build_label_maps(
        OUTPUT_CROP_DIR,
        resolver,
        prev_sp=prev_sp,
        prev_gn=prev_gn,
        prev_fa=prev_fa,
    )

    # Refuse to proceed if anything vanished from the historical maps.
    _assert_no_class_shrink(prev_sp or {}, sp_to_idx, "species")
    _assert_no_class_shrink(prev_gn or {}, gn_to_idx, "genus")
    _assert_no_class_shrink(prev_fa or {}, fa_to_idx, "family")

    n_sp, n_gn, n_fa = len(sp_to_idx), len(gn_to_idx), len(fa_to_idx)

    if min(n_sp, n_gn, n_fa) < 2:
        log(
            f"⏸ Skipping retrain — need ≥2 classes at every level "
            f"(sp={n_sp}, gn={n_gn}, fa={n_fa})."
        )
        return None, None, None

    ds = FishCropDataset(
        OUTPUT_CROP_DIR, resolver, sp_to_idx, gn_to_idx, fa_to_idx, TRAIN_TRANSFORM
    )
    if len(ds) < 4:
        log(f"⏸ Skipping retrain — only {len(ds)} crops on disk.")
        return None, None, None

    ignore_family_idx = fa_to_idx.get("Unknown_Family")

    model.update_heads(n_sp, n_gn, n_fa)
    model.to(DEVICE)
    model.train()

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    w_sp = compute_class_weights([s for _, s, _, _ in ds.samples], n_sp)
    w_gn = compute_class_weights([g for _, _, g, _ in ds.samples], n_gn)
    w_fa = compute_class_weights([f for _, _, _, f in ds.samples], n_fa)

    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, y_sp, y_gn, y_fa in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            y_sp = y_sp.to(DEVICE)
            y_gn = y_gn.to(DEVICE)
            y_fa = y_fa.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits_sp, logits_gn, logits_fa = model(imgs)
                loss_sp = F.cross_entropy(logits_sp, y_sp, weight=w_sp)
                loss_gn = F.cross_entropy(logits_gn, y_gn, weight=w_gn)
                if ignore_family_idx is not None:
                    mask = y_fa != ignore_family_idx
                    if mask.any():
                        loss_fa = F.cross_entropy(
                            logits_fa[mask], y_fa[mask], weight=w_fa
                        )
                    else:
                        loss_fa = logits_fa.sum() * 0.0
                else:
                    loss_fa = F.cross_entropy(logits_fa, y_fa, weight=w_fa)
                loss = loss_sp + LAMBDA_GENUS * loss_gn + LAMBDA_FAMILY * loss_fa

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)

        avg = total_loss / max(len(ds), 1)
        log(f"epoch {epoch + 1}/{epochs} | loss {avg:.4f}")

    return sp_to_idx, gn_to_idx, fa_to_idx


# ==========================================
# 4a. Durable, atomic checkpoint + label-map persistence
# ==========================================
def _atomic_torch_save(obj, path):
    """Write `obj` to `path` atomically: tmp file, fsync, then os.replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    # fsync so a power loss between replace and flush can't truncate the file
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_json_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _rotate_backups(timestamp):
    """
    Snapshot the current live checkpoint + label-map into backups/ under a
    shared timestamp, then prune to KEEP_BACKUPS generations.
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for src, suffix in (
        (MULTIHEAD_CHECKPOINT, "pth"),
        (LABEL_MAP_PATH, "labelmaps.json"),
    ):
        if os.path.exists(src):
            dst = os.path.join(BACKUP_DIR, f"{timestamp}.{suffix}")
            shutil.copy2(src, dst)

    generations = sorted({f.split(".", 1)[0] for f in os.listdir(BACKUP_DIR)})
    for old in generations[:-KEEP_BACKUPS]:
        for suffix in ("pth", "labelmaps.json"):
            p = os.path.join(BACKUP_DIR, f"{old}.{suffix}")
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def save_label_maps(sp_to_idx, gn_to_idx, fa_to_idx):
    """Persist the label maps independent of the .pth, so a checkpoint
    without sp_to_idx (e.g. one produced by train.py) can still be
    reconstructed into a full historical map."""
    _atomic_json_save(
        {
            "sp_to_idx": sp_to_idx,
            "gn_to_idx": gn_to_idx,
            "fa_to_idx": fa_to_idx,
        },
        LABEL_MAP_PATH,
    )


def load_label_maps():
    """Returns ({}, {}, {}) on any failure — missing file, corrupt JSON, etc."""
    if not os.path.exists(LABEL_MAP_PATH):
        return {}, {}, {}
    try:
        with open(LABEL_MAP_PATH, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}, {}, {}
    return (
        d.get("sp_to_idx", {}) or {},
        d.get("gn_to_idx", {}) or {},
        d.get("fa_to_idx", {}) or {},
    )


def save_checkpoint(model, sp_to_idx, gn_to_idx, fa_to_idx):
    """
    Rotate backups, then atomically write the label-map sidecar and the
    checkpoint itself.  Order matters: the sidecar is written first, so if
    the process dies between the two writes we still have a consistent
    map that can repair the (older) .pth on next load.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    _rotate_backups(timestamp)

    save_label_maps(sp_to_idx, gn_to_idx, fa_to_idx)

    _atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "sp_to_idx": sp_to_idx,
            "gn_to_idx": gn_to_idx,
            "fa_to_idx": fa_to_idx,
            "num_species": model.num_species,
            "num_genera": model.num_genera,
            "num_families": model.num_families,
        },
        MULTIHEAD_CHECKPOINT,
    )
    print(f" 💾 Checkpoint saved to {MULTIHEAD_CHECKPOINT} (backup: {timestamp})")


# ==========================================
# 4b. Robust checkpoint loading
# ==========================================
def _infer_head_sizes_from_state_dict(state_dict):
    """
    Recover (num_species, num_genera, num_families) from a checkpoint whose
    explicit size keys are missing (older save format).  Reads dim 0 of any
    state_dict tensor whose key ends in a head-weight suffix.
    """

    def _find(suffixes):
        for k, v in state_dict.items():
            for suf in suffixes:
                if k.endswith(suf) and hasattr(v, "shape") and v.dim() == 2:
                    return int(v.shape[0])
        return None

    n_sp = _find(("fc_species.weight", "species_head.weight", "sp_head.weight"))
    n_gn = _find(("fc_genus.weight", "genus_head.weight", "gn_head.weight"))
    n_fa = _find(("fc_family.weight", "family_head.weight", "fa_head.weight"))
    return n_sp, n_gn, n_fa


def _merge_label_maps(primary, sidecar):
    """
    Union of two label maps.  `sidecar` wins on conflicts (it is the
    authoritative historical record); primary supplies anything the
    sidecar lacks (e.g. a checkpoint that was saved by train.py and never
    touched by this session).
    """
    out = dict(sidecar or {})
    for k, v in (primary or {}).items():
        out.setdefault(k, v)
    return out


def build_or_load_model(backbone_path=None):
    if os.path.exists(MULTIHEAD_CHECKPOINT):
        ckpt = torch.load(MULTIHEAD_CHECKPOINT, map_location=DEVICE)
        state = ckpt["model_state"]

        n_sp = ckpt.get("num_species")
        n_gn = ckpt.get("num_genera")
        n_fa = ckpt.get("num_families")

        ckpt_sp = ckpt.get("sp_to_idx") or {}
        ckpt_gn = ckpt.get("gn_to_idx") or {}
        ckpt_fa = ckpt.get("fa_to_idx") or {}

        # --- Merge in the sidecar so a checkpoint without maps is repaired ---
        sidecar_sp, sidecar_gn, sidecar_fa = load_label_maps()
        sp_to_idx = _merge_label_maps(ckpt_sp, sidecar_sp)
        gn_to_idx = _merge_label_maps(ckpt_gn, sidecar_gn)
        fa_to_idx = _merge_label_maps(ckpt_fa, sidecar_fa)

        # Fallback: explicit keys → merged label-map lengths → tensor shapes.
        if n_sp is None and sp_to_idx:
            n_sp = len(sp_to_idx)
        if n_gn is None and gn_to_idx:
            n_gn = len(gn_to_idx)
        if n_fa is None and fa_to_idx:
            n_fa = len(fa_to_idx)

        if None in (n_sp, n_gn, n_fa):
            i_sp, i_gn, i_fa = _infer_head_sizes_from_state_dict(state)
            n_sp = n_sp or i_sp
            n_gn = n_gn or i_gn
            n_fa = n_fa or i_fa

        if None in (n_sp, n_gn, n_fa):
            raise RuntimeError(
                f"Could not determine head sizes from {MULTIHEAD_CHECKPOINT}. "
                f"Top-level keys: {sorted(ckpt.keys())}; "
                f"first state_dict keys: {list(state.keys())[:8]}"
            )

        model = TaxonomicMultiHead(
            backbone_path=None,
            num_species=n_sp,
            num_genera=n_gn,
            num_families=n_fa,
        ).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        print(
            f" ➔ Loaded checkpoint {MULTIHEAD_CHECKPOINT} "
            f"(sp={n_sp}, gn={n_gn}, fa={n_fa}; "
            f"label-map sp={len(sp_to_idx)} gn={len(gn_to_idx)} fa={len(fa_to_idx)})"
        )
        return model, sp_to_idx, gn_to_idx, fa_to_idx, True

    if backbone_path and os.path.exists(backbone_path):
        print(f" ➔ Using pretrained backbone {backbone_path}")
        model = TaxonomicMultiHead(
            backbone_path=backbone_path,
            num_species=2,
            num_genera=2,
            num_families=2,
        ).to(DEVICE)
    else:
        if backbone_path:
            print(f" ⚠ Backbone {backbone_path} not found — starting cold.")
        model = TaxonomicMultiHead(
            backbone_path=None,
            num_species=2,
            num_genera=2,
            num_families=2,
        ).to(DEVICE)
        print(" ➔ Initialized cold multi-head model.")

    # Even on a cold start, recover any label maps that exist from a prior
    # session whose .pth has since been deleted.
    sidecar_sp, sidecar_gn, sidecar_fa = load_label_maps()
    return model, sidecar_sp, sidecar_gn, sidecar_fa, False


def predict(model, img_tensor, sp_inv, gn_inv, fa_inv):
    model.eval()
    with torch.no_grad():
        sp, gn, fa = model(img_tensor.to(DEVICE))
        sp_p = F.softmax(sp, dim=1)
        gn_p = F.softmax(gn, dim=1)
        fa_p = F.softmax(fa, dim=1)
        sp_conf, sp_i = sp_p.max(dim=1)
        gn_conf, gn_i = gn_p.max(dim=1)
        fa_conf, fa_i = fa_p.max(dim=1)
    return (
        sp_inv.get(sp_i.item(), "?"),
        gn_inv.get(gn_i.item(), "?"),
        fa_inv.get(fa_i.item(), "?"),
        sp_conf.item(),
        gn_conf.item(),
        fa_conf.item(),
    )


# ==========================================
# 5. Qt background worker for retraining
# ==========================================
class RetrainWorker(QThread):
    """
    Runs retrain() off the Qt main thread.  While this worker is alive, do
    NOT call predict()/model(...) from the main thread — the worker mutates
    the same module in place.
    """

    progress = Signal(str)
    finished_with_result = Signal(object, object, object)

    def __init__(self, model, resolver, prev_sp=None, prev_gn=None, prev_fa=None):
        super().__init__()
        self.model = model
        self.resolver = resolver
        self.prev_sp = prev_sp or {}
        self.prev_gn = prev_gn or {}
        self.prev_fa = prev_fa or {}

    def run(self):
        try:
            result = retrain(
                self.model,
                self.resolver,
                log=self.progress.emit,
                prev_sp=self.prev_sp,
                prev_gn=self.prev_gn,
                prev_fa=self.prev_fa,
            )
        except Exception as exc:  # noqa: BLE001
            self.progress.emit(f"❌ Retrain failed: {exc}")
            result = (None, None, None)
        self.finished_with_result.emit(*result)


# ==========================================
# 6. Zoomable / pannable image view
# ==========================================
class ZoomableImageView(QGraphicsView):
    """
    Image viewer with:
      • mouse-wheel zoom anchored under the cursor
      • click-and-drag to pan
      • double-click to toggle fit ↔ 100%
      • Ctrl+= / Ctrl+- / Ctrl+0 / Ctrl+1 keyboard shortcuts
      • preserves zoom level across window resizes
    """

    MIN_SCALE = 0.05
    MAX_SCALE = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.black)
        self.setMinimumSize(600, 400)

        self._current_scale = 1.0
        self._fit_on_next_resize = True

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()

    def fit_to_view(self):
        if self._pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._current_scale = self.transform().m11()
        self._fit_on_next_resize = True

    def zoom_to_100(self):
        if self._pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self._current_scale = 1.0
        self.centerOn(self._pixmap_item)
        self._fit_on_next_resize = False

    def zoom_in(self):
        self._zoom_center(1.25)

    def zoom_out(self):
        self._zoom_center(1 / 1.25)

    def _zoom_at(self, factor, view_pos):
        new_scale = self._current_scale * factor
        if not (self.MIN_SCALE <= new_scale <= self.MAX_SCALE):
            return
        old_pos = self.mapToScene(view_pos)
        self.scale(factor, factor)
        new_pos = self.mapToScene(view_pos)
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        self._current_scale = self.transform().m11()
        self._fit_on_next_resize = False

    def _zoom_center(self, factor):
        new_scale = self._current_scale * factor
        if not (self.MIN_SCALE <= new_scale <= self.MAX_SCALE):
            return
        self.scale(factor, factor)
        self._current_scale = self.transform().m11()
        self._fit_on_next_resize = False

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        self._zoom_at(factor, event.position().toPoint())
        event.accept()

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if abs(self._current_scale - 1.0) < 1e-3:
            self.fit_to_view()
        else:
            self.zoom_to_100()
        event.accept()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._fit_on_next_resize:
            self.fit_to_view()


# ==========================================
# 7. Qt main window
# ==========================================
class LabelerWindow(QMainWindow):
    def __init__(
        self,
        df,
        frame_index,
        resolver,
        model,
        sp_to_idx,
        gn_to_idx,
        fa_to_idx,
        is_trained,
        out_csv_path,
    ):
        super().__init__()

        self.df = df
        self.frame_index = frame_index
        self.resolver = resolver
        self.model = model
        self.sp_to_idx = sp_to_idx
        self.gn_to_idx = gn_to_idx
        self.fa_to_idx = fa_to_idx
        self.sp_inv = {i: s for s, i in sp_to_idx.items()}
        self.gn_inv = {i: g for g, i in gn_to_idx.items()}
        self.fa_inv = {i: f for f, i in fa_to_idx.items()}
        self.is_trained = is_trained
        self.out_csv_path = out_csv_path

        self.action_counter = 0
        self.current_idx = -1
        self.current_row = None
        self.current_cropped = None
        self.current_suggestion = None
        self.current_model_genus = None
        self.current_model_family = None

        self._current_pixmap = None
        self.retrain_worker = None
        self._closing = False

        # frame-nav state inside the current track
        self._track_row_indices = []
        self._track_pos = 0

        # review mode state
        self.review_queue = []
        self.review_pos = -1
        self.review_mode = False
        self.review_stats = {"n": 0, "splits": 0, "merges": 0}

        self._build_ui()
        QTimer.singleShot(0, self._advance)

    # ---------- UI construction ----------
    def _build_ui(self):
        self.setWindowTitle("Active Classification Workspace")
        self.resize(1200, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.image_view = ZoomableImageView()
        layout.addWidget(self.image_view, stretch=1)

        self.frame_label = QLabel("")
        self.frame_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.frame_label)

        self.reco_label = QLabel("")
        self.reco_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.reco_label)

        # Review-mode preview strip (hidden unless a candidate is active).
        self.preview_label = QLabel("")
        self.preview_label.setMinimumHeight(220)
        self.preview_label.setStyleSheet("background:#111; color:#eee;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Review-mode status + button row.
        review_row = QHBoxLayout()
        self.review_btn = QPushButton("🔎 Review suspect ID swaps")
        self.review_btn.setToolTip("Ctrl+R — scan tracks for probable identity swaps")
        self.review_btn.clicked.connect(self._enter_review_mode)
        review_row.addWidget(self.review_btn)

        self.review_info = QLabel("")
        self.review_info.setStyleSheet("font-size: 12px; color:#ccc;")
        review_row.addWidget(self.review_info, stretch=1)
        layout.addLayout(review_row)

        # Species entry.
        input_row = QHBoxLayout()
        self.entry = QLineEdit()
        self.entry.setPlaceholderText(
            "Enter species tag — blank accepts recommendation, 'exit' finishes"
        )
        self.entry.returnPressed.connect(self._on_submit)
        input_row.addWidget(self.entry, stretch=1)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.clicked.connect(self._on_submit)
        input_row.addWidget(self.submit_btn)

        layout.addLayout(input_row)

        # Identity / frame-nav row.
        identity_row = QHBoxLayout()
        identity_row.addWidget(QLabel("True ID:"))

        self.true_id_entry = QLineEdit()
        self.true_id_entry.setMaximumWidth(120)
        self.true_id_entry.setPlaceholderText("identity group (default = track id)")
        self.true_id_entry.returnPressed.connect(self.entry.setFocus)
        identity_row.addWidget(self.true_id_entry)

        self.new_id_btn = QPushButton("New ID")
        self.new_id_btn.setToolTip(
            "Fill the field with a fresh true_id (applied on Submit)"
        )
        self.new_id_btn.clicked.connect(self._assign_new_id)
        identity_row.addWidget(self.new_id_btn)

        self.split_btn = QPushButton("Split here")
        self.split_btn.setToolTip("Start a new true_id from the current frame onwards")
        self.split_btn.clicked.connect(self._split_here)
        identity_row.addWidget(self.split_btn)

        identity_row.addStretch(1)

        self.prev_btn = QPushButton("◀ Prev frame")
        self.prev_btn.clicked.connect(lambda: self._nav_frame(-1))
        identity_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next frame ▶")
        self.next_btn.clicked.connect(lambda: self._nav_frame(+1))
        identity_row.addWidget(self.next_btn)

        layout.addLayout(identity_row)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready. Type a species name, leave blank to accept, 'exit' to finish. "
            "Ctrl+R reviews suspect ID swaps."
        )

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+="), self, activated=self.image_view.zoom_in)
        QShortcut(QKeySequence("Ctrl++"), self, activated=self.image_view.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.image_view.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.image_view.fit_to_view)
        QShortcut(QKeySequence("Ctrl+1"), self, activated=self.image_view.zoom_to_100)
        QShortcut(QKeySequence("Alt+Left"), self, activated=lambda: self._nav_frame(-1))
        QShortcut(
            QKeySequence("Alt+Right"), self, activated=lambda: self._nav_frame(+1)
        )
        QShortcut(QKeySequence("Alt+n"), self, activated=self._assign_new_id)
        QShortcut(QKeySequence("Alt+s"), self, activated=self._split_here)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._enter_review_mode)
        QShortcut(QKeySequence("Space"), self, activated=self._review_skip)
        QShortcut(QKeySequence("S"), self, activated=self._review_split)
        QShortcut(QKeySequence("Escape"), self, activated=self._review_exit)

    # ---------- image handling ----------
    def _set_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self.image_view.set_pixmap(self._current_pixmap)

    # ---------- navigation ----------
    def _find_next_unlabeled(self, start):
        for i in range(start, len(self.df)):
            if pd.isna(self.df.at[i, "assigned_species"]):
                return i
        return -1

    def _advance(self):
        # Scan from row 0 so splits that leave an unlabeled prefix behind
        # are still picked up.
        next_idx = self._find_next_unlabeled(0)
        if next_idx == -1:
            self.statusBar().showMessage("✅ All tracks labeled. Finalizing…")
            self._set_input_enabled(False)
            QTimer.singleShot(400, self.close)
            return
        self._show_index(next_idx)

    def _show_index(self, idx):
        row = self.df.iloc[idx]
        img_path = locate_frame_path(self.frame_index, row["frame"])
        if not img_path or not os.path.exists(img_path):
            self.current_idx = idx
            QTimer.singleShot(0, self._advance)
            return

        img = cv2.imread(img_path)
        if img is None:
            self.current_idx = idx
            QTimer.singleShot(0, self._advance)
            return

        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            self.current_idx = idx
            QTimer.singleShot(0, self._advance)
            return

        self.current_idx = idx
        self.current_row = row
        self.current_cropped = cropped

        # Frame navigation state across the whole track.
        track_idxs = self.df.index[self.df["id"] == row["id"]].tolist()
        track_idxs.sort(key=lambda i: self.df.at[i, "frame"])
        self._track_row_indices = track_idxs
        try:
            self._track_pos = track_idxs.index(idx)
        except ValueError:
            self._track_pos = 0

        suggestion, source_type = None, ""
        model_genus = model_family = None
        if self.is_trained:
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = EVAL_TRANSFORM(pil).unsqueeze(0)
            sp, gn, fa, c_sp, c_gn, c_fa = predict(
                self.model, tensor, self.sp_inv, self.gn_inv, self.fa_inv
            )
            suggestion = canonical_species(sp)
            model_genus = gn
            model_family = fa
            source_type = (
                f"🤖 sp={c_sp * 100:.0f}% gn={c_gn * 100:.0f}% fa={c_fa * 100:.0f}%"
            )

        self.current_suggestion = suggestion
        self.current_model_genus = model_genus
        self.current_model_family = model_family

        display = img.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"Frame: {int(row['frame'])} | Track ID: {int(row['id'])} | "
            f"True ID: {row['true_id']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        self._set_image(display)

        self.frame_label.setText(
            f"Index: {idx} | Frame: {os.path.basename(img_path)} | "
            f"Track ID: {int(row['id'])} | True ID: {row['true_id']} | "
            f"Frame {self._track_pos + 1}/{len(track_idxs)}"
        )
        if suggestion:
            self.reco_label.setText(f"Recommendation: {suggestion}  |  {source_type}")
        else:
            self.reco_label.setText("Recommendation: (model not yet trained)")

        self.true_id_entry.setText(str(row["true_id"]))
        self.entry.clear()
        self.entry.setFocus()

    # ---------- true_id / frame navigation ----------
    def _parse_true_id(self, text):
        text = str(text).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text

    def _next_true_id(self):
        ids = pd.to_numeric(self.df["true_id"], errors="coerce")
        if ids.notna().any():
            return int(ids.max()) + 1
        return f"t{len(self.df) + 1}"

    def _assign_new_id(self):
        """Fill the field with a fresh id; the change is applied on Submit."""
        self.true_id_entry.setText(str(self._next_true_id()))
        self.true_id_entry.setFocus()
        self.true_id_entry.selectAll()

    def _split_here(self):
        """Create a new true_id for the suffix of the current segment."""
        if self.current_row is None:
            return
        current_id = self.current_row["id"]
        current_true_id = self.current_row["true_id"]
        current_frame = self.current_row["frame"]
        new_true_id = self._next_true_id()
        seg_mask = (
            (self.df["id"] == current_id)
            & (self.df["true_id"] == current_true_id)
            & (self.df["frame"] >= current_frame)
        )
        n = int(seg_mask.sum())
        if n == 0:
            return
        self.df.loc[seg_mask, "true_id"] = new_true_id
        self.statusBar().showMessage(
            f"✂ Split {n} row(s) at frame {current_frame} → true_id {new_true_id}"
        )
        self._show_index(self.current_idx)

    def _nav_frame(self, delta):
        if not self._track_row_indices:
            return
        new_pos = self._track_pos + delta
        if 0 <= new_pos < len(self._track_row_indices):
            self._show_index(self._track_row_indices[new_pos])

    # ---------- input handling ----------
    def _set_input_enabled(self, enabled: bool):
        self.entry.setEnabled(enabled)
        self.submit_btn.setEnabled(enabled)
        self.true_id_entry.setEnabled(enabled)
        self.new_id_btn.setEnabled(enabled)
        self.split_btn.setEnabled(enabled)

    def _on_submit(self):
        if self._closing or (self.retrain_worker and self.retrain_worker.isRunning()):
            return
        if self.current_idx < 0 or self.current_row is None:
            return

        text = self.entry.text().strip().lower()
        if text == "exit":
            self.close()
            return

        if text == "" and self.current_suggestion:
            final_species = canonical_species(self.current_suggestion)
        elif text != "":
            final_species = canonical_species(text)
        else:
            self.reco_label.setText(
                "❌ Input required — no valid recommendation exists yet."
            )
            return

        row = self.current_row
        final_species, final_genus, final_family = self.resolver.resolve(final_species)
        if final_family == "Unknown_Family" and self.current_model_family:
            final_family = self.current_model_family
        if final_genus == "Unknown" and self.current_model_genus:
            final_genus = self.current_model_genus

        # --- Resolve true_id: merge/rename the current segment if changed ---
        current_id = row["id"]
        old_true_id = row["true_id"]
        typed = self.true_id_entry.text().strip()
        new_true_id = self._parse_true_id(typed) if typed else None

        if new_true_id is not None and new_true_id != old_true_id:
            seg_mask = (self.df["id"] == current_id) & (
                self.df["true_id"] == old_true_id
            )
            self.df.loc[seg_mask, "true_id"] = new_true_id
            effective_true_id = new_true_id
        else:
            effective_true_id = old_true_id

        # --- Propagate the label to the whole identity group ---
        group_mask = (self.df["true_id"] == effective_true_id) & (
            self.df["assigned_species"].isna()
        )
        n_matched = int(group_mask.sum())
        self.df.loc[group_mask, "assigned_species"] = final_species
        self.df.loc[group_mask, "assigned_genus"] = final_genus
        self.df.loc[group_mask, "assigned_family"] = final_family

        species_dir = os.path.join(OUTPUT_CROP_DIR, final_species)
        os.makedirs(species_dir, exist_ok=True)
        crop_path = os.path.join(
            species_dir,
            f"crop_idx{self.current_idx}_true{effective_true_id}.jpg",
        )
        cv2.imwrite(crop_path, self.current_cropped)

        self.action_counter += 1
        self.statusBar().showMessage(
            f"💾 Applied labels to {n_matched} frames for true_id "
            f"{effective_true_id}. Actions: {self.action_counter}"
        )

        if self.action_counter % RETRAIN_INTERVAL == 0:
            self._start_retrain()
        else:
            self._advance()

    # ---------- review mode ----------
    def _enter_review_mode(self):
        if self.review_mode:
            return
        self.statusBar().showMessage("🔎 Scanning tracks for suspicious jumps…")
        QApplication.processEvents()

        disc = detect_track_discontinuities(self.df, max_pairs=500)
        if disc.empty:
            self.statusBar().showMessage("No candidates found.")
            return

        # Only surface pairs whose two sides still share the same true_id.
        disc = disc[disc["true_id_prev"] == disc["true_id_curr"]].reset_index(drop=True)
        if disc.empty:
            self.statusBar().showMessage("No unresolved candidates.")
            return

        self.review_queue = disc.to_dict("records")
        self.review_pos = -1
        self.review_mode = True
        self.review_stats = {"n": 0, "splits": 0, "merges": 0}
        self.statusBar().showMessage(
            f"🔎 {len(self.review_queue)} candidates. "
            f"Space=not a swap, S=split here, Esc=exit."
        )
        self._next_review()

    def _next_review(self):
        self.review_pos += 1
        if self.review_pos >= len(self.review_queue):
            msg = (
                f"✅ Review complete — {self.review_stats['n']} checked, "
                f"{self.review_stats['splits']} split, "
                f"{self.review_stats['merges']} merged."
            )
            self.statusBar().showMessage(msg)
            self.review_mode = False
            self.preview_label.clear()
            self.review_info.clear()
            self.df.to_csv(self.out_csv_path, index=False)
            self._advance()
            return

        item = self.review_queue[self.review_pos]
        self._show_review_pair(item)

    def _show_review_pair(self, item):
        curr_idx = item["curr_idx"]
        # Repoint the main view to the current row so the box shown is the one
        # after the jump.
        self._show_index(curr_idx)

        composite = self._compose_pair_crops(item["prev_idx"], curr_idx)
        if composite is not None:
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))
        else:
            self.preview_label.setText("(no preview)")

        self.review_info.setText(
            f"[{self.review_pos + 1}/{len(self.review_queue)}] "
            f"id={item['id']} frames {item['prev_frame']}→{item['curr_frame']} "
            f"(gap {item['frame_gap']}) | "
            f"jump {item['pos_jump_pf']:.2f} px/f | "
            f"Δdir {item['vel_angle_deg']:.0f}° | "
            f"area×{item['area_ratio']:.2f} | "
            f"score {item['score']:.1f}"
        )
        self.true_id_entry.setText(str(self.df.at[curr_idx, "true_id"]))
        self.entry.clear()

    def _compose_pair_crops(self, prev_idx, curr_idx):
        """Return a BGR image with the prev-frame crop beside the curr-frame crop."""

        def _load_crop(idx):
            r = self.df.iloc[idx]
            img_path = locate_frame_path(self.frame_index, r["frame"])
            if not img_path:
                return None
            img = cv2.imread(img_path)
            if img is None:
                return None
            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            crop = img[y1:y2, x1:x2]
            return crop if crop.size else None

        prev_crop = _load_crop(prev_idx)
        curr_crop = _load_crop(curr_idx)
        if prev_crop is None and curr_crop is None:
            return None

        H = 260

        def _fit(img):
            h, w = img.shape[:2]
            s = H / max(h, 1)
            return cv2.resize(img, (max(1, int(w * s)), H))

        tiles = []
        for crop, tag in ((prev_crop, "PREV"), (curr_crop, "CURR")):
            if crop is None:
                tiles.append(np.full((H, 120, 3), 40, dtype=np.uint8))
                continue
            t = _fit(crop)
            cv2.putText(t, tag, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(t)

        sep = np.full((H, 20, 3), 90, dtype=np.uint8)
        return np.hstack([tiles[0], sep, tiles[1]])

    def _review_skip(self):
        if not self.review_mode:
            return
        self.review_stats["n"] += 1
        self._next_review()

    def _review_exit(self):
        if not self.review_mode:
            return
        self.review_mode = False
        self.preview_label.clear()
        self.review_info.clear()
        self.statusBar().showMessage("Exited review mode.")
        self._advance()

    def _review_split(self):
        """User decided the pair IS a swap: new true_id for the current segment."""
        if not self.review_mode or self.review_pos < 0:
            return
        item = self.review_queue[self.review_pos]
        curr_idx = item["curr_idx"]

        track_id = self.df.at[curr_idx, "id"]
        old_true = self.df.at[curr_idx, "true_id"]
        curr_frm = self.df.at[curr_idx, "frame"]
        new_true = self._next_true_id()

        mask = (
            (self.df["id"] == track_id)
            & (self.df["true_id"] == old_true)
            & (self.df["frame"] >= curr_frm)
        )
        n = int(mask.sum())
        if n == 0:
            self._next_review()
            return

        self.df.loc[mask, "true_id"] = new_true
        # Clear existing labels so the new segment gets relabeled with the
        # correct species on the next pass.
        for col in ("assigned_species", "assigned_genus", "assigned_family"):
            self.df.loc[mask, col] = pd.NA

        self.review_stats["n"] += 1
        self.review_stats["splits"] += 1
        self.df.to_csv(self.out_csv_path, index=False)
        self.statusBar().showMessage(
            f"✂ Split {n} rows at frame {curr_frm} → true_id {new_true}. "
            f"Exiting review to label the new segment."
        )
        self.review_mode = False
        self.preview_label.clear()
        self.review_info.clear()
        self._show_index(curr_idx)  # user types the species and hits Enter

    # ---------- retraining ----------
    def _start_retrain(self):
        self._set_input_enabled(False)
        self.statusBar().showMessage(
            f"🔄 Milestone ({self.action_counter} actions). Retraining…"
        )
        self.df.to_csv(self.out_csv_path, index=False)

        self.retrain_worker = RetrainWorker(
            self.model,
            self.resolver,
            prev_sp=self.sp_to_idx,
            prev_gn=self.gn_to_idx,
            prev_fa=self.fa_to_idx,
        )
        self.retrain_worker.progress.connect(self._on_retrain_progress)
        self.retrain_worker.finished_with_result.connect(self._on_retrain_done)
        self.retrain_worker.start()

    def _on_retrain_progress(self, msg: str):
        self.statusBar().showMessage(msg)

    def _on_retrain_done(self, sp_to_idx, gn_to_idx, fa_to_idx):
        if sp_to_idx is not None:
            self.sp_to_idx = sp_to_idx
            self.gn_to_idx = gn_to_idx
            self.fa_to_idx = fa_to_idx
            self.sp_inv = {i: s for s, i in sp_to_idx.items()}
            self.gn_inv = {i: g for g, i in gn_to_idx.items()}
            self.fa_inv = {i: f for f, i in fa_to_idx.items()}
            self.is_trained = True
            save_checkpoint(self.model, sp_to_idx, gn_to_idx, fa_to_idx)
            self.statusBar().showMessage("✅ Retrain complete. Resuming…")
        else:
            self.statusBar().showMessage("⏸ Retrain skipped (not enough data).")

        self.df.to_csv(self.out_csv_path, index=False)
        self.retrain_worker = None
        self._set_input_enabled(True)
        self._advance()

    # ---------- shutdown ----------
    def closeEvent(self, event):  # noqa: N802
        if self._closing:
            event.accept()
            return

        if self.action_counter == 0:
            self.df.to_csv(self.out_csv_path, index=False)
            event.accept()
            return

        event.ignore()
        self._closing = True
        self._set_input_enabled(False)
        self.df.to_csv(self.out_csv_path, index=False)

        self.statusBar().showMessage("🏁 Session ending — running final retrain…")
        self.retrain_worker = RetrainWorker(
            self.model,
            self.resolver,
            prev_sp=self.sp_to_idx,
            prev_gn=self.gn_to_idx,
            prev_fa=self.fa_to_idx,
        )
        self.retrain_worker.progress.connect(self._on_retrain_progress)
        self.retrain_worker.finished_with_result.connect(self._on_final_retrain_done)
        self.retrain_worker.start()

    def _on_final_retrain_done(self, sp_to_idx, gn_to_idx, fa_to_idx):
        if sp_to_idx is not None:
            self.sp_to_idx = sp_to_idx
            self.gn_to_idx = gn_to_idx
            self.fa_to_idx = fa_to_idx
        save_checkpoint(self.model, self.sp_to_idx, self.gn_to_idx, self.fa_to_idx)
        self.df.to_csv(self.out_csv_path, index=False)

        # Persist any GBIF results from this session so the next run is offline.
        try:
            self.resolver.finalize()
        except Exception as exc:  # noqa: BLE001
            print(f" ⚠ Could not finalize resolver cache: {exc}")

        print(f"🏁 Session closed. Progress saved to: {self.out_csv_path}")
        self.retrain_worker = None
        self.close()


# ==========================================
# 8. Entry point
# ==========================================
def execute_pipeline():
    args = parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    csv_path, checklist_path, root_data_dir = get_user_paths(
        csv_path=args.csv_path,
        checklist_path=args.checklist_path,
        root_data_dir=args.root_data_dir,
    )

    out_csv_path = str(get_output_csv_path(csv_path, CSV_OUT))

    # Layered resolver: checklist → secondary CSV → GBIF → cache.
    resolver = TaxonomyResolver(
        checklist_path,
        secondary_csv=args.secondary_taxonomy,
        cache_path=args.gbif_cache,
        verbose=True,
    )

    print(f"\nCUDA status: {torch.cuda.is_available()} | Device: {DEVICE}")
    print(f"Master sheet (In) : {csv_path}")
    print(f"Master sheet (Out): {out_csv_path}")
    print(f"Checklist:          {checklist_path}")
    print(f"Frames folder:      {root_data_dir}")
    print(f"GBIF cache:         {args.gbif_cache}")
    print(f"Checkpoint:         {MULTIHEAD_CHECKPOINT}")
    print(f"Label maps:         {LABEL_MAP_PATH}")
    print(f"Backups:            {BACKUP_DIR}")

    os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

    # Prewarm before the GUI opens: any GBIF work happens here, not during
    # labeling.  Usually a no-op because the cache was written last session.
    print("\n📚 Resolving taxonomy for existing crop folders…")
    prewarm_resolver_from_existing_crops(resolver)

    print("\n📂 Indexing frames...")
    frame_index = build_frame_index(root_data_dir)
    print(f"   Found {len(frame_index)} valid frame files.")

    df = pd.read_csv(csv_path).reset_index(drop=True)

    for col in ("assigned_species", "assigned_genus", "assigned_family"):
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype="object")

    # Identity column: rows sharing a true_id are the same physical fish.
    if "true_id" not in df.columns:
        df["true_id"] = df["id"]
    else:
        missing = df["true_id"].isna()
        df.loc[missing, "true_id"] = df.loc[missing, "id"]

    model, sp_to_idx, gn_to_idx, fa_to_idx, is_trained = build_or_load_model(
        BACKBONE_PATH
    )

    print("\n=======================================================")
    print("      🚀 ACTIVE LEARNING SYSTEM INITIALIZED 🚀")
    print("=======================================================")
    print("Instructions:")
    print("  ➔ Type the true species name and press Enter.")
    print("  ➔ Leave blank to accept the auto-suggestion.")
    print("  ➔ Edit 'True ID' to merge a broken track into an existing fish.")
    print("  ➔ Press 'Split here' to start a new identity at the current frame.")
    print("  ➔ Ctrl+R reviews probable ID swaps automatically.")
    print("  ➔ Type 'exit' (or close the window) to retrain & save.")
    print("=======================================================\n")

    window = LabelerWindow(
        df=df,
        frame_index=frame_index,
        resolver=resolver,
        model=model,
        sp_to_idx=sp_to_idx,
        gn_to_idx=gn_to_idx,
        fa_to_idx=fa_to_idx,
        is_trained=is_trained,
        out_csv_path=out_csv_path,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    execute_pipeline()
