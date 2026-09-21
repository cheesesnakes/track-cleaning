# Andaman Reef Fish — Active Learning System

Local, privacy-focused pipeline for identifying reef fish from video-derived
frames, with hierarchical (species / genus / family) classification, a Qt
labeling GUI, track-level mass labeling, true-ID merge/split identity
management, free-scroll frame auditing, automated ID-swap detection, and
periodic GPU retraining on an NVIDIA RTX 4070 (or any CUDA device).

---

## Pipeline Overview

```
             ┌──────────────────────┐
             │  download_images.py  │  GBIF + iNat + local datasets
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │   data_cleaning.py   │  Quarantine unreadable reference images
             └──────────┬───────────┘  (scan_bad_images.py)
                        ▼
             ┌──────────────────────┐
             │      train.py        │  Pretrain ResNet18 backbone
             └──────────┬───────────┘  → fish-classifier-0.pth
                        ▼
             ┌──────────────────────┐
             │    find_swaps.py     │  Batch-detect & split probable ID
             └──────────┬───────────┘  swaps in tracking CSVs (run before,
                        │              or between sessions of, track_cleaning.py)
                        ▼
┌───────────────────────────────────────────────┐
│            track_cleaning.py                  │
│  • Qt path dialogs                            │
│  • Frame index (O(1) lookup)                  │
│  • Multi-head model (sp / genus / family)     │
│  • Zoomable / pannable viewer                 │
│  • Track-level mass assignment                │
│  • True-ID merge / split identity management  │
│  • Free-scroll frame navigation (audit mode)  │
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

| File                 | Purpose                                                                      |
| -------------------- | ----------------------------------------------------------------------------|
| `taxonomy.py`        | Loads Andaman checklist, resolves species → genus/family                    |
| `model.py`           | `TaxonomicMultiHead` — shared ResNet18 + 3 linear heads                     |
| `track_cleaning.py`  | Main active-learning loop (Qt GUI + label + retrain + identity management)  |
| `find_swaps.py`      | Batch ID-swap detector/fixer — splits discontinuous tracks onto fresh IDs   |
| `train.py`           | Pretrain a ResNet18 backbone on reference imagery                           |
| `data_cleaning.py`   | One-time scan that quarantines unreadable/corrupt reference images          |
| `validate.py`        | Checklist validation with suggested corrections                             |
| `download_images.py` | GBIF + iNaturalist fetch, plus local dataset ingest                         |

---

## Setup

```
pip install torch torchvision pandas opencv-python numpy pillow requests PySide6
```

For GPU acceleration install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>.

`find_swaps.py` needs `opencv-python`, `pandas`, and `numpy` (already covered above).

---

## Data Specification

### Master tracking CSV

Required columns:

| Column   | Meaning                                            |
| -------- | -------------------------------------------------- |
| `frame`  | Chronological frame index                          |
| `id`     | Unique track ID (same physical fish across frames) |
| `x1, y1` | Top-left corner of the bounding box                |
| `x2, y2` | Bottom-right corner of the bounding box            |

The pipeline writes back `assigned_species`, `assigned_genus`, `assigned_family` for each track, plus a `true_id` identity column (see [Identity Management](#identity-management)) and a `review_status` column used by `find_swaps.py`.

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

```
python download_images.py \
    --species-csv species_list.csv \
    --out-dir ./reference_images \
    --max-per-species 40
```

To ingest a local dataset (e.g. FishWIO, WildFish, Fish4Knowledge):

```
python download_images.py \
    --species-csv species_list.csv \
    --ingest-dir /path/to/FishWIO \
    --source-name fishwio
```

The ingested dataset should be in ImageFolder layout (one folder per species).

### 2. Quarantine bad reference images

Bulk scraping from GBIF/iNaturalist can leave truncated downloads, HTML
error pages saved with an image extension, or zero-byte files behind.
Run this once before pretraining to catch them:

```
python data_cleaning.py --data-dir ./reference_images
python data_cleaning.py --data-dir ./reference_images --dry-run  # report only
```

Bad files are **moved**, not deleted, to a sibling `<data-dir>_quarantine/<class>/`
folder so `ImageFolder` no longer sees them and you can inspect or restore
them later.

### 3. Pretrain the backbone

```
python train.py \
    --data-dir ./reference_images \
    --epochs 15 --batch-size 32 \
    --out models/fish-classifier-0.pth \
    --amp
```

Produces `fish-classifier-0.pth`. Only its conv-layer weights are used
downstream — the `fc` head is discarded.

### 4. Detect and fix probable ID swaps

Before (or between) labeling sessions, scan tracking CSVs for
discontinuities that look like an occlusion-triggered identity swap — a
sudden position jump, a sharp direction change, or a size jump between
consecutive frames of the same track:

```
# Preview across the whole tree
python find_swaps.py \
    --csv-root data/annotated_videos \
    --frames-root data/annotated_frames \
    --video-root /mnt/videos \
    --dry-run

# Apply
python find_swaps.py \
    --csv-root data/annotated_videos \
    --frames-root data/annotated_frames \
    --video-root /mnt/videos

# CSVs only, no frame backfill
python find_swaps.py --csv-root data/annotated_videos
```

For each candidate above `--threshold` (default 30), the script splits the
suffix of that `(id, true_id)` segment onto a fresh `true_id` — in place,
no prompts — and clears that segment's `assigned_species` /
`assigned_genus` / `assigned_family` so `track_cleaning.py` re-labels it on
the next pass. The `id` column itself is never touched.

If `--frames-root` and `--video-root` are both given, any swap frame not
already on disk is decoded straight from the matching source video and
written into the frames folder using the same `frame%06d.jpg` naming
convention `track_cleaning.py` expects — so the labeling pass has the real
frame in hand even if it was never exported. Videos are matched to a CSV
by filename stem, any extension, mirrored under the same relative path as
`--csv-root`. If either root is omitted, or a video isn't found, the CSV
splits are still applied; the missing frame is just skipped when
`track_cleaning.py` advances to it.

Splits within the same segment are applied in descending frame order so a
later split never invalidates the row range an earlier split already
carved out. Rows already marked `dismissed` or `split` in `review_status`
are skipped on subsequent runs.

### 5. Run the active-learning loop

```
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

Optional flags:

| Flag                    | Meaning                                                          |
| ------------------------| ------------------------------------------------------------------|
| `--secondary-taxonomy`  | Broader taxonomy CSV consulted before falling back to GBIF        |
| `--gbif-cache`          | Path to the JSON GBIF-lookup cache (shared with `train.py`; defaults to `./output/.gbif_cache.json`) |

### 6. Label

Each unlabeled track appears in the **Active Classification Workspace**:

| Input                               | Effect                                                  |
| ------------------------------------| ----------------------------------------------------------|
| **Blank + Enter**                   | Accepts the model (or CSV) suggestion                   |
| **species name + Enter**            | Overrides, e.g. `lutjanus_decussatus`                   |
| **exit** (or window close / Ctrl+Q) | Runs a final retrain, saves checkpoints and CSV, closes |

The status bar shows the model's softmax confidence at each taxonomic
level (`sp=…% gn=…% fa=…%`) so you can see when the model is uncertain.

Labels are mass-assigned to **all frames sharing the same True ID group**
(see below) — one action covers dozens of frames. If you already know a
label for the current identity, the species field is pre-filled with it,
so fixing a mistake is an edit rather than a retype.

### Identity Management

`true_id` is the identity column: rows sharing a `true_id` are treated as
the *same physical fish*, independent of `id` (the raw tracker's track
number, which is never modified by the labeling GUI).

| Control                         | Effect                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------|
| **True ID** field                | Shows/edits the current row's identity group; defaults to the track ID                |
| **New ID** button (`Alt+n`)      | Fills the field with a fresh, unused `true_id`, applied on the next Submit             |
| **Split here** button (`Alt+s`)  | Immediately moves every row of the current segment from the current frame onward onto a new `true_id` — for a track whose ID jumped mid-swim |
| Typing an **existing** True ID into the field | Merges the current row's segment into that identity group on Submit                |

Typing a species for an already-labeled True ID, or merging two
identities together, rewrites every row in that identity group and
deletes the stale on-disk crop(s) filed under the previous species — so a
corrected mistake doesn't linger in the training corpus. Emptied species
folders under `labeled_fish_crops/` are pruned automatically.

### Viewer controls

| Action        | Gesture                              |
| ------------- | ------------------------------------ |
| Zoom in / out | Scroll wheel (zooms under cursor)    |
| Pan           | Click-and-drag                       |
| Fit to window | Double-click, or `Ctrl+0`            |
| 1:1 pixels    | Double-click again, or `Ctrl+1`      |
| Fine zoom in  | `Ctrl+=`                             |
| Fine zoom out | `Ctrl+-`                             |
| Quit + save   | `Ctrl+Q`, or the window close button |

Zoom limits are 0.05×–40×, enough to inspect fin rays or jaw shape on a
4K frame.

### Frame navigation

| Control                       | Effect                                                                                     |
| ------------------------------| ---------------------------------------------------------------------------------------------|
| **Prev / Next frame** buttons, or `Alt+←` / `Alt+→` | Step through *every* detection in the dataset, sorted by frame — not just the current track |

This free-scroll mode lets you audit or relabel any row without hunting
for its track. Frames with no detection, or an image that fails to load,
are skipped silently so a single press always lands on something
viewable. The status bar shows both the current position within the
track (`Track X/Y`) and within the whole dataset (`All X/Y`).

### 7. Validate against the checklist

```
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

| Constant               | Default                                      | Meaning                              |
| ----------------------- | --------------------------------------------- | ------------------------------------- |
| `RETRAIN_INTERVAL`     | `150`                                        | Label actions between retrains       |
| `EPOCHS_PER_RETRAIN`   | `5`                                           | Training epochs per retrain          |
| `LR`                   | `1e-4`                                       | Learning rate during active learning |
| `LAMBDA_GENUS`         | `0.3`                                        | Genus loss weight                    |
| `LAMBDA_FAMILY`        | `0.1`                                        | Family loss weight                   |
| `BATCH_SIZE`           | `32`                                         | Retrain batch size                   |
| `MULTIHEAD_CHECKPOINT` | `./models/checkpoints/fish-classifier-1.pth` | Checkpoint path                      |
| `OUTPUT_CROP_DIR`      | `./output/labeled_fish_crops`                | Crop corpus root                     |
| `CSV_OUT`              | `./output/tracks`                            | Output CSV root                      |
| `DEFAULT_GBIF_CACHE`   | `./output/.gbif_cache.json`                  | GBIF lookup cache (shared with `train.py`) |

Top of `train.py` (via CLI): `--epochs`, `--batch-size`, `--lr`, `--amp`.

Top of `find_swaps.py` (via CLI): `--csv-root` (required), `--frames-root`,
`--video-root`, `--threshold` (default 30.0), `--max-pairs` (default 500),
`--dry-run`.

Top of `data_cleaning.py` (via CLI): `--data-dir` (required), `--dry-run`.

---

## Output Files

| Path                                         | Contents                                                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------|
| `output/labeled_fish_crops/<species>/*.jpg`  | Cropped fish from every labeled track                                         |
| `output/tracks/<...>/tracks.csv`             | Labeled CSV, mirroring the input's subfolder layout under `annotated_videos/` |
| `models/checkpoints/fish-classifier-1.pth`   | Latest multi-head weights + label maps                                        |
| `models/fish-classifier-0.pth`               | ResNet18 conv weights from `train.py`                                         |
| `output/validation_report.csv`               | Flagged rows + suggested corrections                                          |
| `output/reference_images/provenance_log.csv` | Source / license for every reference image                                    |
| `<reference-dir>_quarantine/<class>/*`       | Unreadable reference images moved aside by `data_cleaning.py`                 |

The output CSV is flushed at every retrain milestone and once more on
exit, so an unexpected shutdown loses at most `RETRAIN_INTERVAL` labels.

---

## Notes on External Datasets

For Andaman reef fish, the following datasets are worth ingesting via `--ingest-dir`, in rough order of regional relevance:

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