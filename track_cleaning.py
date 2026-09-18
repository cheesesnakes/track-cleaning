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
  • Label one representative frame per track; the label is propagated to
    every unlabeled row sharing that Track ID.
  • Auto-suggestion from the current multi-head model (species / genus /
    family) with softmax confidences shown in the UI. Press Enter on an
    empty field to accept, type a name to override.
  • Zoomable, pannable viewer (scroll wheel to zoom under cursor, drag to
    pan, double-click to toggle fit ↔ 100%, Ctrl+0 / Ctrl+1 / Ctrl+= /
    Ctrl+- shortcuts) for resolving ambiguous fish.
  • Crops of every confirmed label are written to disk under
    ./output/labeled_fish_crops/<species>/ for use as future training data.
  • Every RETRAIN_INTERVAL confirmations, and once more at session end,
    the model is retrained on the on-disk crop corpus. Training runs on a
    QThread worker so the UI stays responsive and streams epoch/loss
    progress to the status bar.
  • Labeled CSV is flushed at every milestone and on exit, preserving the
    input's subfolder structure under ./output/tracks/.

Usage:
    python track_cleaning.py
    python track_cleaning.py --csv tracks.csv --checklist checklist.csv --frames-dir frames/

    Any of --csv/-c, --checklist/-l, --frames-dir/-f omitted on the command
    line falls back to a Qt file/folder selection dialog.

Inputs:
    --csv/-c          Tracking CSV (needs at least: frame, id, x1, y1, x2, y2).
    --checklist/-l    Andaman checklist CSV with columns: species, genus, family.
    --frames-dir/-f   Parent folder containing the video's frame images.

Outputs:
    ./output/tracks/...           Labeled CSV, mirroring the annotated_videos
                                  subfolder layout of the input.
    ./output/labeled_fish_crops/  Per-species crop folders used as the training set.
    ./models/checkpoints/         Multi-head checkpoint (fish-classifier-1.pth).

Requires:
    pip install PySide6
"""

import argparse
import os
import re
import sys
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
from taxonomy import TaxonomyResolver

# ==========================================
# Configuration
# ==========================================
OUTPUT_CROP_DIR = "./output/labeled_fish_crops"
CHECKPOINT_DIR = "./models"
MULTIHEAD_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "fish-classifier-1.pth")
CSV_OUT = "./output/tracks"

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
    Computes output path inside output_base_dir preserving the subfolder
    structure starting after 'annotated_videos'.

    Example:
      Input:  /data/projects/annotated_videos/site_1/cam_A/labels.csv
      Output: ./output/videos/site_1/cam_A/labels.csv
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
        help="Path to the Andaman checklist CSV (columns: species, genus, family).",
    )
    parser.add_argument(
        "-f",
        "--frames-dir",
        dest="root_data_dir",
        type=str,
        default=None,
        help="Path to the parent folder containing the frames for this video.",
    )
    return parser.parse_args()


def get_user_paths(csv_path=None, checklist_path=None, root_data_dir=None):
    """
    Resolves the three required inputs. Any value already supplied (e.g. via
    CLI args) is used as-is after validation; anything missing is prompted
    for via a Qt file/folder dialog.

    QApplication must already exist before calling this.
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
            "Select Andaman Checklist CSV (columns: species, genus, family)",
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
def clean_label_string(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9_]", "", text.strip().replace(" ", "_").lower())


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
            species_raw = species_dir.name
            sp_key, genus, family = resolver.resolve(species_raw)
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


def build_label_maps(crop_dir, resolver):
    species_set, genus_set, family_set = set(), set(), set()
    root = Path(crop_dir)
    if not root.exists():
        return {}, {}, {}

    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir():
            continue
        sp_key, genus, family = resolver.resolve(species_dir.name)
        species_set.add(sp_key)
        genus_set.add(genus)
        family_set.add(family)

    sp_to_idx = {s: i for i, s in enumerate(sorted(species_set))}
    gn_to_idx = {g: i for i, g in enumerate(sorted(genus_set))}
    fa_to_idx = {f: i for i, f in enumerate(sorted(family_set))}
    return sp_to_idx, gn_to_idx, fa_to_idx


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
# 4. Retraining  (runs on a worker thread)
# ==========================================
def retrain(model, resolver, epochs=EPOCHS_PER_RETRAIN, log=print):
    """
    Full retrain pass.  `log` is a callable that receives status strings so
    the caller (Qt worker) can route them into the UI instead of stdout.
    """
    sp_to_idx, gn_to_idx, fa_to_idx = build_label_maps(OUTPUT_CROP_DIR, resolver)
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
                loss = (
                    F.cross_entropy(logits_sp, y_sp, weight=w_sp)
                    + LAMBDA_GENUS * F.cross_entropy(logits_gn, y_gn, weight=w_gn)
                    + LAMBDA_FAMILY * F.cross_entropy(logits_fa, y_fa, weight=w_fa)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)

        avg = total_loss / max(len(ds), 1)
        log(f"epoch {epoch + 1}/{epochs} | loss {avg:.4f}")

    return sp_to_idx, gn_to_idx, fa_to_idx


def save_checkpoint(model, sp_to_idx, gn_to_idx, fa_to_idx):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
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
    print(f" 💾 Checkpoint saved to {MULTIHEAD_CHECKPOINT}")


def build_or_load_model(backbone_path=None):
    if os.path.exists(MULTIHEAD_CHECKPOINT):
        ckpt = torch.load(MULTIHEAD_CHECKPOINT, map_location=DEVICE)
        model = TaxonomicMultiHead(
            backbone_path=None,
            num_species=ckpt["num_species"],
            num_genera=ckpt["num_genera"],
            num_families=ckpt["num_families"],
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(
            f" ➔ Loaded checkpoint {MULTIHEAD_CHECKPOINT} "
            f"(sp={ckpt['num_species']}, gn={ckpt['num_genera']}, fa={ckpt['num_families']})"
        )
        return model, ckpt["sp_to_idx"], ckpt["gn_to_idx"], ckpt["fa_to_idx"], True

    model = TaxonomicMultiHead(
        backbone_path=backbone_path, num_species=2, num_genera=2, num_families=2
    ).to(DEVICE)
    print(" ➔ Initialized cold/warm multi-head model.")
    return model, {}, {}, {}, False


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
    Runs retrain() off the Qt main thread.

    Emits `progress(str)` for status messages and
    `finished_with_result(sp_to_idx, gn_to_idx, fa_to_idx)` when done.
    On skip/failure the three payload objects are None.

    While a worker is running, do NOT call predict()/model(...) from the
    main thread — the worker mutates the same module in place.
    """

    progress = Signal(str)
    finished_with_result = Signal(object, object, object)

    def __init__(self, model, resolver):
        super().__init__()
        self.model = model
        self.resolver = resolver

    def run(self):
        try:
            result = retrain(self.model, self.resolver, log=self.progress.emit)
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
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
        # We do all anchoring manually so the resize handler doesn't fight us.
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.black)
        self.setMinimumSize(600, 400)

        self._current_scale = 1.0
        self._fit_on_next_resize = True

    # ---------- public API ----------
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

    # ---------- internals ----------
    def _zoom_at(self, factor: float, view_pos):
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

    def _zoom_center(self, factor: float):
        new_scale = self._current_scale * factor
        if not (self.MIN_SCALE <= new_scale <= self.MAX_SCALE):
            return
        self.scale(factor, factor)
        self._current_scale = self.transform().m11()
        self._fit_on_next_resize = False

    # ---------- Qt events ----------
    def wheelEvent(self, event):  # noqa: N802 - Qt API
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        self._zoom_at(factor, event.position().toPoint())
        event.accept()

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt API
        # Toggle between fit-to-window and 1:1
        if abs(self._current_scale - 1.0) < 1e-3:
            self.fit_to_view()
        else:
            self.zoom_to_100()
        event.accept()

    def resizeEvent(self, event):  # noqa: N802 - Qt API
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

        self._build_ui()
        # kick off the first item on the next event-loop tick
        QTimer.singleShot(0, self._advance)

    # ---------- UI construction ----------
    def _build_ui(self):
        self.setWindowTitle("Active Classification Workspace")
        self.resize(1200, 800)

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

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready. Type a species name, leave blank to accept, 'exit' to finish."
        )

        # Convenience shortcut: Ctrl+Q quits through the same finalize path.
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        # Zoom / pan shortcuts. Ctrl-prefixed so they don't fight the entry field.
        QShortcut(QKeySequence("Ctrl+="), self, activated=self.image_view.zoom_in)
        QShortcut(QKeySequence("Ctrl++"), self, activated=self.image_view.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.image_view.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.image_view.fit_to_view)
        QShortcut(QKeySequence("Ctrl+1"), self, activated=self.image_view.zoom_to_100)

    # ---------- image handling ----------
    def _set_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self.image_view.set_pixmap(self._current_pixmap)

    def _rescale_pixmap(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._rescale_pixmap()

    # ---------- navigation ----------
    def _find_next_unlabeled(self, start):
        for i in range(start, len(self.df)):
            if pd.isna(self.df.at[i, "assigned_species"]):
                return i
        return -1

    def _advance(self):
        next_idx = self._find_next_unlabeled(self.current_idx + 1)
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

        # Inference (safe: retrain worker is never alive here)
        suggestion, source_type = None, ""
        model_genus = model_family = None
        if self.is_trained:
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = EVAL_TRANSFORM(pil).unsqueeze(0)
            sp, gn, fa, c_sp, c_gn, c_fa = predict(
                self.model, tensor, self.sp_inv, self.gn_inv, self.fa_inv
            )
            suggestion = clean_label_string(sp)
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
            f"Frame: {int(row['frame'])} | Track ID: {int(row['id'])}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        self._set_image(display)

        self.frame_label.setText(
            f"Index: {idx} | Frame: {os.path.basename(img_path)} | "
            f"Track ID: {int(row['id'])}"
        )
        if suggestion:
            self.reco_label.setText(f"Recommendation: {suggestion}  |  {source_type}")
        else:
            self.reco_label.setText("Recommendation: (model not yet trained)")

        self.entry.clear()
        self.entry.setFocus()

    # ---------- input handling ----------
    def _set_input_enabled(self, enabled: bool):
        self.entry.setEnabled(enabled)
        self.submit_btn.setEnabled(enabled)

    def _on_submit(self):
        # Ignore input while a retrain worker owns the model or during shutdown.
        if self._closing or (self.retrain_worker and self.retrain_worker.isRunning()):
            return
        if self.current_idx < 0 or self.current_row is None:
            return

        text = self.entry.text().strip().lower()

        if text == "exit":
            self.close()
            return

        if text == "" and self.current_suggestion:
            final_species = clean_label_string(self.current_suggestion)
        elif text != "":
            final_species = clean_label_string(text)
        else:
            self.reco_label.setText(
                "❌ Input required — no valid recommendation exists yet."
            )
            return

        row = self.current_row
        _, final_genus, final_family = self.resolver.resolve(final_species)

        if final_family == "Unknown_Family" and self.current_model_family:
            final_family = self.current_model_family
        if final_genus == "Unknown" and self.current_model_genus:
            final_genus = self.current_model_genus

        track_mask = (self.df["id"] == row["id"]) & (self.df["assigned_species"].isna())
        n_matched = int(track_mask.sum())
        self.df.loc[track_mask, "assigned_species"] = final_species
        self.df.loc[track_mask, "assigned_genus"] = final_genus
        self.df.loc[track_mask, "assigned_family"] = final_family

        species_dir = os.path.join(OUTPUT_CROP_DIR, final_species)
        os.makedirs(species_dir, exist_ok=True)
        crop_path = os.path.join(
            species_dir, f"crop_idx{self.current_idx}_id{int(row['id'])}.jpg"
        )
        cv2.imwrite(crop_path, self.current_cropped)

        self.action_counter += 1
        self.statusBar().showMessage(
            f"💾 Applied labels to {n_matched} frames for Track ID {int(row['id'])}. "
            f"Actions: {self.action_counter}"
        )

        if self.action_counter % RETRAIN_INTERVAL == 0:
            self._start_retrain()
        else:
            self._advance()

    # ---------- retraining ----------
    def _start_retrain(self):
        self._set_input_enabled(False)
        self.statusBar().showMessage(
            f"🔄 Milestone ({self.action_counter} actions). Retraining…"
        )
        # Flush current progress before the worker starts
        self.df.to_csv(self.out_csv_path, index=False)

        self.retrain_worker = RetrainWorker(self.model, self.resolver)
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
    def closeEvent(self, event):  # noqa: N802 - Qt API
        # If we've already gone through finalize, just close.
        if self._closing:
            event.accept()
            return

        # Nothing was labeled — no need for a final retrain.
        if self.action_counter == 0:
            self.df.to_csv(self.out_csv_path, index=False)
            event.accept()
            return

        # Otherwise: ignore the event, run final retrain, then re-close.
        event.ignore()
        self._closing = True
        self._set_input_enabled(False)
        self.df.to_csv(self.out_csv_path, index=False)

        self.statusBar().showMessage("🏁 Session ending — running final retrain…")
        self.retrain_worker = RetrainWorker(self.model, self.resolver)
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
        print(f"🏁 Session closed. Progress saved to: {self.out_csv_path}")
        self.retrain_worker = None
        self.close()  # _closing is True → accepted immediately


# ==========================================
# 7. Entry point
# ==========================================
def execute_pipeline():
    args = parse_args()

    # QApplication must exist before any QFileDialog / QMainWindow is created.
    app = QApplication.instance() or QApplication(sys.argv)

    csv_path, checklist_path, root_data_dir = get_user_paths(
        csv_path=args.csv_path,
        checklist_path=args.checklist_path,
        root_data_dir=args.root_data_dir,
    )

    out_csv_path = str(get_output_csv_path(csv_path, CSV_OUT))

    resolver = TaxonomyResolver(checklist_path)

    print(f"\nCUDA status: {torch.cuda.is_available()} | Device: {DEVICE}")
    print(f"Master sheet (In) : {csv_path}")
    print(f"Master sheet (Out): {out_csv_path}")
    print(f"Checklist:          {checklist_path}")
    print(f"Frames folder:      {root_data_dir}")

    os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

    print("\n📂 Indexing frames...")
    frame_index = build_frame_index(root_data_dir)
    print(f"   Found {len(frame_index)} valid frame files.")

    df = pd.read_csv(csv_path)
    for col in ("assigned_species", "assigned_genus", "assigned_family"):
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype="object")

    backbone_path = "/models/fish-classifier-0.pth"
    backbone_arg = backbone_path if os.path.exists(backbone_path) else None
    model, sp_to_idx, gn_to_idx, fa_to_idx, is_trained = build_or_load_model(
        backbone_arg
    )

    print("\n=======================================================")
    print("      🚀 ACTIVE LEARNING SYSTEM INITIALIZED 🚀")
    print("=======================================================")
    print("Instructions:")
    print("  ➔ Type the true species name and press Enter.")
    print("  ➔ Leave blank to accept the auto-suggestion.")
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
