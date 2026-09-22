"""
track_cleaning.py

Active-learning pipeline for Andaman reef fish identification.

Combines CLI/GUI path selection, per-track mass labeling, hierarchical
multi-head classification (species / genus / family), on-disk crop
storage, and a Qt-based viewer with zoom/pan for fine-grained ID.

Features:
  • CLI args (-c/-l/-f) with Qt file/folder dialogs as fallback for any
    input not supplied on the command line.
  • Loads a TaxonomicMultiHead checkpoint (produced by train.py) and uses
    it to suggest labels from frame one.
  • Label one representative frame per track; the label is propagated to
    every unlabeled row sharing that Track ID.
  • A `true_id` identity column: rows sharing a `true_id` are the same
    physical fish. Merge broken tracks by editing True ID in the UI;
    split switched tracks via the Split-here button.
  • Auto-suggestion from the loaded multi-head model with softmax
    confidences shown in the UI. Press Enter on an empty field to accept,
    type a name to override.
  • Top-3 species suggestions, each annotated with its resolved family,
    plus the family head's top-3. Family lookups are memoized per session.
  • Corrections overwrite cleanly: typing a species for an already-labeled
    True ID (or merging two identities together) rewrites every row in
    that identity group and removes the stale crop(s) filed under the
    previous species, so a fixed mistake does not linger in the training
    corpus.  The species entry pre-fills with the current label so a fix
    is an edit rather than a retype.
  • Zoomable, pannable viewer (scroll wheel to zoom under cursor, drag to
    pan, double-click to toggle fit ↔ 100%, Ctrl+0 / Ctrl+1 / Ctrl+= /
    Ctrl+- shortcuts) for resolving ambiguous fish.
  • Free-scroll frame navigation (Prev/Next and Alt+←/→) walks every
    detection in the dataset, not just the current Track ID, so you can
    audit or relabel any row without hunting for its track.
  • Crops of every confirmed label are written to disk under
    ./output/labeled_fish_crops/<species>/ for use as future training data.
  • Labeled CSV is flushed at every milestone and on exit, preserving the
    input's subfolder structure under ./output/tracks/.
  • Family resolution uses the same layered resolver as train.py: checklist,
    optional secondary CSV, then GBIF, with a shared on-disk cache so the
    second run is offline-instant.

Note:
  This script does NOT train or fine-tune models. It only produces labeled
  CSVs and a crop corpus; use train.py (or a later run of train.py) to fit
  a new model on the accumulated crops.

  Probable ID-swap detection, correction, and swap-frame recovery are
  handled by a separate companion script, check_id_swaps.py. That script
  scans a tracking CSV for motion/size discontinuities, splits swapped
  segments onto a fresh true_id, and pulls the swap frame out of the
  source video (offline) so the normal labeling pass here can pick it up.
  Run it before — or between sessions of — this tool.

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

Reads:
    ./models/checkpoints/fish-classifier-1.pth         Multi-head checkpoint.
    ./models/checkpoints/fish-classifier-labelmaps.json  Label-map sidecar.

Requires:
    pip install PySide6
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from PySide6.QtCore import Qt, Signal, QTimer
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
BACKBONE_PATH = "./models/fish-classifier-0.pth"
CSV_OUT = "./output/tracks"
DEFAULT_GBIF_CACHE = "./output/.gbif_cache.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 3. Robust checkpoint loading
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


def predict(model, img_tensor, sp_inv, gn_inv, fa_inv, topk=3):
    """Return top-`topk` species, genus, and family predictions with confidences.

    Returns
    -------
    sp_list : list[(species_name, confidence)]   # length <= topk
    gn_list : list[(genus_name,   confidence)]   # length <= topk
    fa_list : list[(family_name,  confidence)]   # length <= topk
    """
    model.eval()
    with torch.no_grad():
        sp, gn, fa = model(img_tensor.to(DEVICE))
        sp_p = F.softmax(sp, dim=1)
        gn_p = F.softmax(gn, dim=1)
        fa_p = F.softmax(fa, dim=1)

        k_sp = min(topk, sp_p.size(1))
        k_gn = min(topk, gn_p.size(1))
        k_fa = min(topk, fa_p.size(1))

        sp_conf, sp_i = sp_p.topk(k_sp, dim=1)
        gn_conf, gn_i = gn_p.topk(k_gn, dim=1)
        fa_conf, fa_i = fa_p.topk(k_fa, dim=1)

    def _pack(conf, idx, inv):
        return [
            (inv.get(idx[0, k].item(), "?"), float(conf[0, k].item()))
            for k in range(idx.size(1))
        ]

    return (
        _pack(sp_conf, sp_i, sp_inv),
        _pack(gn_conf, gn_i, gn_inv),
        _pack(fa_conf, fa_i, fa_inv),
    )


# ==========================================
# 4. Zoomable / pannable image view
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
# 5. Qt main window
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
        self._closing = False

        # Session-level memoization for species → family resolution, so the
        # top-3 display doesn't trigger repeated GBIF / resolver lookups.
        self._species_family_cache = {}

        # frame-nav state inside the current track
        self._track_row_indices = []
        self._track_pos = 0

        # free-scroll state across every detection, sorted by frame
        self._all_row_indices = (
            self.df.assign(_f=pd.to_numeric(self.df["frame"], errors="coerce"))
            .sort_values(["_f", "id"], kind="stable")
            .index.tolist()
        )
        self._global_pos = 0

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
        self.reco_label.setWordWrap(True)
        layout.addWidget(self.reco_label)

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
        self.prev_btn.setToolTip(
            "Previous detection in the whole dataset (Alt+Left). "
            "Skips to the prior frame if the current track has no match."
        )
        self.prev_btn.clicked.connect(lambda: self._nav_frame(-1))
        identity_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next frame ▶")
        self.next_btn.setToolTip(
            "Next detection in the whole dataset (Alt+Right). "
            "Skips to the next frame if the current track has no match."
        )
        self.next_btn.clicked.connect(lambda: self._nav_frame(+1))
        identity_row.addWidget(self.next_btn)

        layout.addLayout(identity_row)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready. Type a species name, leave blank to accept, 'exit' to finish."
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

    # ---------- image handling ----------
    def _set_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self.image_view.set_pixmap(self._current_pixmap)

    # ---------- taxonomy helper ----------
    def _family_for(self, species):
        """Resolve family for a species name, memoized for the session."""
        if species in self._species_family_cache:
            return self._species_family_cache[species]
        try:
            _, _, fam = self.resolver.resolve(species)
        except Exception:  # noqa: BLE001
            fam = "Unknown"
        self._species_family_cache[species] = fam
        return fam

    # ---------- navigation ----------
    def _load_frame_bgr(self, frame_num):
        """Return the full BGR frame image for `frame_num`, or (None, None)."""
        img_path = locate_frame_path(self.frame_index, frame_num)
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_path
        return None, None

    def _advance(self):
        """Show the first unlabeled row whose frame is loadable."""
        for i in range(len(self.df)):
            if pd.isna(self.df.iloc[i]["assigned_species"]) and self._show_index(i):
                return
        self.statusBar().showMessage("✅ All tracks labeled. Finalizing…")
        self._set_input_enabled(False)
        QTimer.singleShot(400, self.close)

    def _show_index(self, idx):
        """Display row `idx`. Returns True on success, False on missing/unreadable frame."""
        row = self.df.iloc[idx]
        img, img_path = self._load_frame_bgr(row["frame"])
        if img is None:
            return False

        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            return False

        prev_id = self.current_row["id"] if self.current_row is not None else None

        self.current_idx = idx
        self.current_row = row
        self.current_cropped = cropped

        # Per-track navigation state (kept for the "Track X/Y" indicator).
        track_idxs = self.df.index[self.df["id"] == row["id"]].tolist()
        track_idxs.sort(key=lambda i: self.df.at[i, "frame"])
        self._track_row_indices = track_idxs
        try:
            self._track_pos = track_idxs.index(idx)
        except ValueError:
            self._track_pos = 0

        # Global navigation state.
        try:
            self._global_pos = self._all_row_indices.index(idx)
        except ValueError:
            self._global_pos = 0

        suggestion, source_type = None, ""
        model_genus = model_family = None
        sp_suggestions = []  # [(species, conf, family), ...]
        if self.is_trained:
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = EVAL_TRANSFORM(pil).unsqueeze(0)
            sp_list, gn_list, fa_list = predict(
                self.model, tensor, self.sp_inv, self.gn_inv, self.fa_inv, topk=3
            )

            for sp_name, c_sp in sp_list:
                sp_canon = canonical_species(sp_name)
                sp_suggestions.append((sp_canon, c_sp, self._family_for(sp_canon)))

            suggestion = sp_suggestions[0][0] if sp_suggestions else None
            model_genus = gn_list[0][0] if gn_list else None
            model_family = fa_list[0][0] if fa_list else None

            sp_txt = "  |  ".join(
                f"{s} ({c * 100:.0f}% · {fam})" for s, c, fam in sp_suggestions
            )
            gn_txt = f"{gn_list[0][0]} ({gn_list[0][1] * 100:.0f}%)" if gn_list else "?"
            fa_txt = "  |  ".join(f"{f} ({c * 100:.0f}%)" for f, c in fa_list)
            source_type = (
                f"🤖 Species top-3: {sp_txt}\n"
                f"    Genus: {gn_txt}   |   Family top-3: {fa_txt}"
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

        frame_src = os.path.basename(img_path) if img_path else "(decoded from video)"
        self.frame_label.setText(
            f"Index: {idx} | Frame: {frame_src} | "
            f"Track ID: {int(row['id'])} | True ID: {row['true_id']} | "
            f"Track {self._track_pos + 1}/{len(track_idxs)} | "
            f"All {self._global_pos + 1}/{len(self._all_row_indices)}"
        )

        # Surface any label already attached to this row or its True ID
        # group, so free-scrolling shows *what's been decided* rather than
        # only the model's guess.
        existing_label = row["assigned_species"] if "assigned_species" in row else None
        if pd.isna(existing_label) and "true_id" in row:
            grp = self.df[
                (self.df["true_id"] == row["true_id"])
                & (self.df["assigned_species"].notna())
            ]
            if len(grp):
                existing_label = grp["assigned_species"].iloc[0]

        if pd.notna(existing_label):
            rec_txt = f"Current label: {existing_label}"
            if suggestion and canonical_species(existing_label) != suggestion:
                rec_txt += f"\n  model top-1: {suggestion}\n{source_type}"
            self.reco_label.setText(rec_txt)
        elif suggestion:
            self.reco_label.setText(f"Recommendation:\n{source_type}")
        else:
            self.reco_label.setText("Recommendation: (model not yet trained)")

        self.true_id_entry.setText(str(row["true_id"]))

        # Only reset the species entry when we move to a different fish.
        # If we already know a label for this identity, pre-fill it so a
        # correction is an edit to a visible value rather than a retype.
        if prev_id != row["id"]:
            if pd.notna(existing_label):
                self.entry.setText(str(existing_label))
                self.entry.selectAll()
            else:
                self.entry.clear()

        self.entry.setFocus()
        return True

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
        """Step through *all* detections in the dataset, sorted by frame.

        Frame numbers with no detection (or unreadable images) are skipped
        silently so a single Prev/Next press always lands on something
        viewable, regardless of which track you started from.
        """
        if self.current_idx < 0 or not self._all_row_indices:
            return
        try:
            pos = self._all_row_indices.index(self.current_idx)
        except ValueError:
            pos = 0
        step = 1 if delta > 0 else -1
        pos += step
        while 0 <= pos < len(self._all_row_indices):
            if self._show_index(self._all_row_indices[pos]):
                return
            pos += step
        self.statusBar().showMessage("No more frames in either direction.")

    # ---------- on-disk crop maintenance ----------
    def _purge_crops_for_true_id(self, true_id, keep_species=None):
        """Delete on-disk crops filed for `true_id` under the *wrong* species.

        Crop filenames embed both the row index and the true_id, e.g.
        `crop_idx42_true5.jpg`, so a true_id-glob is a precise way to find
        every crop belonging to that identity group regardless of which
        species folder it was written under.

        `keep_species` = folder name that should survive; pass None to
        remove every crop for this true_id (used when a true_id is retired
        by a merge and its crops become orphaned).

        Afterwards, any species folder left empty is removed so downstream
        training doesn't see a class with no examples.
        """
        root = Path(OUTPUT_CROP_DIR)
        if not root.exists():
            return 0

        suffix = f"_true{true_id}.jpg"
        keep_resolved = (root / keep_species).resolve() if keep_species else None

        removed = 0
        for species_dir in root.iterdir():
            if not species_dir.is_dir():
                continue
            try:
                if keep_resolved is not None and species_dir.resolve() == keep_resolved:
                    continue
            except OSError:
                continue
            for crop in species_dir.glob(f"crop_idx*{suffix}"):
                try:
                    crop.unlink()
                    removed += 1
                except OSError:
                    pass

        # Prune any folders we just emptied.
        for species_dir in list(root.iterdir()):
            if not species_dir.is_dir():
                continue
            try:
                if not any(species_dir.iterdir()):
                    species_dir.rmdir()
            except OSError:
                pass

        return removed

    # ---------- input handling ----------
    def _set_input_enabled(self, enabled: bool):
        self.entry.setEnabled(enabled)
        self.submit_btn.setEnabled(enabled)
        self.true_id_entry.setEnabled(enabled)
        self.new_id_btn.setEnabled(enabled)
        self.split_btn.setEnabled(enabled)

    def _on_submit(self):
        if self._closing:
            return
        if self.current_idx < 0 or self.current_row is None:
            return

        text = self.entry.text().strip().lower()
        if text == "exit":
            self.close()
            return

        row = self.current_row
        current_id = row["id"]
        old_true_id = row["true_id"]

        # ---- 1. Decide which True ID we're writing to -------------------
        typed = self.true_id_entry.text().strip()
        new_true_id = self._parse_true_id(typed) if typed else None
        if new_true_id is not None and new_true_id != old_true_id:
            effective_true_id = new_true_id
        else:
            effective_true_id = old_true_id
        is_merge = new_true_id is not None and new_true_id != old_true_id

        # ---- 2. Look for an already-assigned species on this identity ---
        existing = self.df[
            (self.df["true_id"] == effective_true_id)
            & (self.df["assigned_species"].notna())
        ]
        existing_species = (
            existing["assigned_species"].iloc[0] if len(existing) else None
        )

        # ---- 3. Resolve the species ------------------------------------
        if text == "":
            if existing_species is not None:
                final_species = canonical_species(existing_species)
                self.statusBar().showMessage(
                    f"↩ Reused existing label '{final_species}' for "
                    f"true_id {effective_true_id}."
                )
            elif self.current_suggestion:
                final_species = canonical_species(self.current_suggestion)
            else:
                self.reco_label.setText(
                    "❌ Input required — no prior label or recommendation for this ID."
                )
                return
        else:
            final_species = canonical_species(text)

        final_species, final_genus, final_family = self.resolver.resolve(final_species)
        if final_family == "Unknown_Family" and self.current_model_family:
            final_family = self.current_model_family
        if final_genus == "Unknown" and self.current_model_genus:
            final_genus = self.current_model_genus

        # ---- 4. If the True ID changed, apply the segment merge ---------
        if is_merge:
            seg_mask = (self.df["id"] == current_id) & (
                self.df["true_id"] == old_true_id
            )
            self.df.loc[seg_mask, "true_id"] = new_true_id

        # ---- 5. Propagate / overwrite ----------------------------------
        # A typed species is an explicit instruction: overwrite every row
        # in the identity group, so correcting an earlier mistake actually
        # takes effect instead of silently no-op'ing on already-labeled
        # rows.  A merge is treated the same way — otherwise the joined
        # group would end up with a mixture of the two old labels.
        #
        # A blank submission that wasn't a merge only fills NaN rows, so
        # accepting the model suggestion never tramples an existing label.
        overwrite = bool(text) or is_merge
        if overwrite:
            group_mask = self.df["true_id"] == effective_true_id
        else:
            group_mask = (self.df["true_id"] == effective_true_id) & (
                self.df["assigned_species"].isna()
            )
        n_matched = int(group_mask.sum())
        self.df.loc[group_mask, "assigned_species"] = final_species
        self.df.loc[group_mask, "assigned_genus"] = final_genus
        self.df.loc[group_mask, "assigned_family"] = final_family

        # ---- 6. Reconcile the on-disk crop corpus ----------------------
        # If we just merged, the old true_id no longer exists in the
        # DataFrame; any crop filed under it is orphaned.
        n_purged = 0
        if is_merge:
            n_purged += self._purge_crops_for_true_id(old_true_id)

        # Remove any crops for this true_id sitting under a *different*
        # species folder — those are the stale artefacts of the earlier,
        # wrong label and would otherwise keep poisoning the training set.
        n_purged += self._purge_crops_for_true_id(
            effective_true_id, keep_species=final_species
        )

        species_dir = os.path.join(OUTPUT_CROP_DIR, final_species)
        os.makedirs(species_dir, exist_ok=True)
        crop_path = os.path.join(
            species_dir,
            f"crop_idx{self.current_idx}_true{effective_true_id}.jpg",
        )
        cv2.imwrite(crop_path, self.current_cropped)

        self.action_counter += 1
        msg = (
            f"💾 Applied labels to {n_matched} frames for true_id {effective_true_id}."
        )
        if n_purged:
            msg += f" Purged {n_purged} stale crop(s)."
        msg += f" Actions: {self.action_counter}"
        self.statusBar().showMessage(msg)

        # Flush the CSV to disk at every milestone so a crash can't lose
        # more than a handful of labels.
        self.df.to_csv(self.out_csv_path, index=False)

        self._advance()

    # ---------- shutdown ----------
    def closeEvent(self, event):  # noqa: N802
        if self._closing:
            event.accept()
            return
        self._closing = True
        self._set_input_enabled(False)

        self.df.to_csv(self.out_csv_path, index=False)

        # Persist any GBIF results from this session so the next run is offline.
        try:
            self.resolver.finalize()
        except Exception as exc:  # noqa: BLE001
            print(f" ⚠ Could not finalize resolver cache: {exc}")

        print(f"🏁 Session closed. Progress saved to: {self.out_csv_path}")
        event.accept()


# ==========================================
# 6. Entry point
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
    print(
        "  ➔ For probable ID swaps, run check_id_swaps.py against the source "
        "video first; it will split the track and backfill the swap frame."
    )
    print("  ➔ Type 'exit' (or close the window) to save and quit.")
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
