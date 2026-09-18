# Andaman Reef Fish — Active Learning System

Local, privacy-focused pipeline for identifying reef fish from video-derived
frames, with hierarchical (species / genus / family) classification, a Qt
labeling GUI, track-level mass labeling, zoom/pan inspection, and periodic
GPU retraining on an NVIDIA RTX 4070 (or any CUDA device).

---

## Pipeline Overview

```
                 ┌──────────────────────┐
                 │  download_images.py  │  GBIF + iNat + local datasets
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │      train.py        │  Pretrain ResNet18 backbone
                 └──────────┬───────────┘  → fish-classifier-0.pth
                            ▼
    ┌───────────────────────────────────────────────┐
    │            track_cleaning.py                  │
    │  • Qt path dialogs                            │
    │  • Frame index (O(1) lookup)                  │
    │  • Multi-head model (sp / genus / family)     │
    │  • Zoomable / pannable viewer                 │
    │  • Track-level mass assignment                │
    │  • Retrain every 150 actions (worker thread)  │
    │  • Checkpoint to ./models/checkpoints/        │
    └──────────┬────────────────────────────────────┘
               ▼
    ┌──────────────────────┐
    │     validate.py      │  Checklist consistency
    └──────────────────────┘  → validation_report.csv
```

---

## Files

| File | Purpose |
|---|---|
| `taxonomy.py` | Loads Andaman checklist, resolves species → genus/family |
| `model.py` | `TaxonomicMultiHead` — shared ResNet18 + 3 linear heads |
| `track_cleaning.py` | Main active-learning loop (Qt GUI + label + retrain) |
| `train.py` | Pretrain a ResNet18 backbone on reference imagery |
| `validate.py` | Checklist validation with suggested corrections |
| `download_images.py` | GBIF + iNaturalist fetch, plus local dataset ingest |

---

## Setup

```bash
pip install torch torchvision pandas opencv-python numpy pillow requests PySide6
```

For GPU acceleration install the CUDA build of PyTorch from
https://pytorch.org/get-started/locally/.

---

## Data Specification

### Master tracking CSV

Required columns:

| Column | Meaning |
|---|---|
| `frame` | Chronological frame index |
| `id` | Unique track ID (same physical fish across frames) |
| `x1, y1` | Top-left corner of the bounding box |
| `x2, y2` | Bottom-right corner of the bounding box |

The pipeline writes back `assigned_species`, `assigned_genus`,
`assigned_family` for each track.

### Andaman checklist CSV

Required: `species`. Optional: `genus`, `family`.

If `genus` and `family` are present, the resolver can auto-fill them for
any labeled species. If not, the pipeline falls back to the model's own
genus/family predictions.

### Frame directory layout

```
parent_directory/
├── site_alpha/
│   └── plot_A/
│       └── gopro_video_01/
│           ├── frame000102.jpg
│           └── frame000104.jpg
└── site_beta/
    └── plot_C/
        └── gopro_video_03/
            └── frame001240.jpg
```

The pipeline indexes all `.jpg` files once at startup, then resolves
frames by filename in O(1).

---

## Workflow

### 1. Fetch reference imagery

```bash
python download_images.py \
    --species-csv species_list.csv \
    --out-dir ./reference_images \
    --max-per-species 40
```

To ingest a local dataset (e.g. FishWIO, WildFish, Fish4Knowledge):

```bash
python download_images.py \
    --species-csv species_list.csv \
    --ingest-dir /path/to/FishWIO \
    --source-name fishwio
```

The ingested dataset should be in ImageFolder layout (one folder per species).

### 2. Pretrain the backbone

```bash
python train.py \
    --data-dir ./reference_images \
    --epochs 15 --batch-size 32 \
    --out models/fish-classifier-0.pth \
    --amp
```

Produces `fish-classifier-0.pth`. Only its conv-layer weights are used
downstream — the `fc` head is discarded.

### 3. Run the active-learning loop

```bash
python track_cleaning.py
# or with explicit paths:
python track_cleaning.py \
    --csv  /data/annotated_videos/site_1/cam_A/tracks.csv \
    --checklist andaman_checklist.csv \
    --frames-dir /data/frames/site_1/cam_A
```

If any of `--csv/-c`, `--checklist/-l`, `--frames-dir/-f` is omitted, a
native Qt file/folder dialog opens for that input. The pipeline then
displays one representative frame per unlabeled track.

### 4. Label

Each unlabeled track appears in the **Active Classification Workspace**:

| Input | Effect |
|---|---|
| **Blank + Enter** | Accepts the model (or CSV) suggestion |
| **species name + Enter** | Overrides, e.g. `lutjanus_decussatus` |
| **exit** (or window close / Ctrl+Q) | Runs a final retrain, saves checkpoints and CSV, closes |

The status bar shows the model's softmax confidence at each taxonomic
level (`sp=…% gn=…% fa=…%`) so you can see when the model is uncertain.

Labels are mass-assigned to **all frames with the same track ID** — one
action covers dozens of frames.

### Viewer controls

| Action | Gesture |
|---|---|
| Zoom in / out | Scroll wheel (zooms under cursor) |
| Pan | Click-and-drag |
| Fit to window | Double-click, or `Ctrl+0` |
| 1:1 pixels | Double-click again, or `Ctrl+1` |
| Fine zoom in | `Ctrl+=` |
| Fine zoom out | `Ctrl+-` |
| Quit + save | `Ctrl+Q`, or the window close button |

Zoom limits are 0.05×–40×, enough to inspect fin rays or jaw shape on a
4K frame.

### 5. Validate against the checklist

```bash
python validate.py \
    --master-csv tracking_master.csv \
    --checklist-csv andaman_checklist.csv \
    --report-out validation_report.csv
```

Reports any species/genus/family not present in the checklist, plus a
`suggested_species` / `suggested_genus` / `suggested_family` column for
flagged rows.

---

## How Retraining Works

Every 150 labeling actions (and once more on exit):

1. All crops in `labeled_fish_crops/` are indexed.
2. Species / genus / family label sets are derived from the crops,
   using `TaxonomyResolver` to map species → genus/family.
3. The multi-head model's three heads are resized to match the current
   class counts. **The backbone is preserved** — previously learned
   features are not lost.
4. Training runs for `EPOCHS_PER_RETRAIN` (default 5) epochs with:
   - Shared ResNet18 backbone
   - Class-weighted cross-entropy at each level
   - Loss = `L_species + 0.3·L_genus + 0.1·L_family`
   - Mixed precision on CUDA
5. Checkpoint saved to `./models/checkpoints/fish-classifier-1.pth`.

### Threading

Retraining runs on a **`QThread` worker**. The Qt event loop keeps
draining during training, so the window stays responsive: you can resize,
pan, or zoom the current image, and the status bar streams progress
(`epoch 1/5 | loss …`). Input is disabled while the worker owns the
model, since it mutates the same `TaxonomicMultiHead` instance used for
inference. When the worker finishes, the updated heads are swapped in and
the next track is shown automatically.

Closing the window (X button), pressing `Ctrl+Q`, or typing `exit` all
route through the same shutdown path: the current CSV is flushed, one
final retrain runs on the worker, the checkpoint is written, and the app
quits cleanly.

### Why multi-head, not three separate models?

- **1/3 the GPU memory** — one ResNet18, not three.
- **Regularization** — genus and family losses act as auxiliary
  supervision, which matters when you have <100 crops per species.
- **Graceful fallback** — if the species head is uncertain, the genus
  prediction is still usable. The display panel shows all three
  confidences at once.
- **Growing label sets** — new species and genera are handled by
  resizing the appropriate head; no model rebuild required.

---

## Configuration

Top of `track_cleaning.py`:

| Constant | Default | Meaning |
|---|---|---|
| `RETRAIN_INTERVAL` | `150` | Label actions between retrains |
| `EPOCHS_PER_RETRAIN` | `5` | Training epochs per retrain |
| `LR` | `1e-4` | Learning rate during active learning |
| `LAMBDA_GENUS` | `0.3` | Genus loss weight |
| `LAMBDA_FAMILY` | `0.1` | Family loss weight |
| `BATCH_SIZE` | `32` | Retrain batch size |
| `MULTIHEAD_CHECKPOINT` | `./models/checkpoints/fish-classifier-1.pth` | Checkpoint path |
| `OUTPUT_CROP_DIR` | `./output/labeled_fish_crops` | Crop corpus root |
| `CSV_OUT` | `./output/tracks` | Output CSV root |

Top of `train.py` (via CLI): `--epochs`, `--batch-size`, `--lr`, `--amp`.

---

## Output Files

| Path | Contents |
|---|---|
| `output/labeled_fish_crops/<species>/*.jpg` | Cropped fish from every labeled track |
| `output/tracks/<...>/tracks.csv` | Labeled CSV, mirroring the input's subfolder layout under `annotated_videos/` |
| `models/checkpoints/fish-classifier-1.pth` | Latest multi-head weights + label maps |
| `models/fish-classifier-0.pth` | ResNet18 conv weights from `train.py` |
| `output/validation_report.csv` | Flagged rows + suggested corrections |
| `output/reference_images/provenance_log.csv` | Source / license for every reference image |

The output CSV is flushed at every retrain milestone and once more on
exit, so an unexpected shutdown loses at most `RETRAIN_INTERVAL` labels.

---

## Notes on External Datasets

For Andaman reef fish, the following datasets are worth ingesting via
`--ingest-dir`, in rough order of regional relevance:

1. **FishWIO** — 114,664 images, 124 Western Indian Ocean reef species.
   Highest regional overlap with the Andaman Sea.
2. **WildFish** — 54,459 images, 1,000 global species. Broad prior; many
   species will not occur in the Andaman region.
3. **Fish4Knowledge** — 27,370 images, 23 Taiwan reef species. Fauna
   differs substantially; use only as auxiliary.
4. **FishNet** — 94,532 images, 17,357 species. Taxonomic hierarchy is
   useful, but the dataset is heavily imbalanced.

Each dataset should be arranged in ImageFolder layout before ingesting.
Species names not present in your checklist will still be added to the
pretraining corpus — this is fine, since only the conv-layer weights
carry over to the active-learning stage.
