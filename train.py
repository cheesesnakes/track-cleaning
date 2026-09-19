"""
train.py

Pretrain a TaxonomicMultiHead model on reference imagery (GBIF /
iNaturalist folders from download_images.py, or local copies of FishWIO /
WildFish / Fish4Knowledge arranged as one folder per species) **plus** any
labeled crop corpus produced by track_cleaning.py.

Multiple data sources are supported via repeated --data-dir flags.  Every
directory is expected to have the same layout (one subfolder per species,
images inside), and all of them are scanned and merged into a single
training set with one unified label map.  Example:

    python train.py \
        --data-dir ./reference_images \
        --data-dir ./output/labeled_fish_crops \
        --checklist andaman_checklist.csv \
        --epochs 15 --out models/checkpoints/fish-classifier-1.pth

Produces a checkpoint that is directly loadable by track_cleaning.py: same
state_dict format, same label-map schema, same three-head architecture.
The label-map JSON sidecar (fish-classifier-labelmaps.json) is written
alongside the checkpoint so track_cleaning.py can load class indices
without having to re-derive them.

Family resolution is layered (see taxonomy.py): Andaman checklist first,
optional secondary CSV second, then GBIF.  GBIF is queried at most once
per genus per run and cached on disk, so the second run is offline and
instant.  A prewarm pass resolves every genus across every data source
before image scanning starts, so the file walk never interleaves with
network I/O.

Species folder names are normalized via taxonomy.canonical_species, so
'Lutjanus decussatus', 'Lutjanus_decussatus', and 'lutjanus_decussatus'
all map to the same class — including across different --data-dir roots.

Usage:
    python train.py \
        --data-dir ./reference_images \
        [--data-dir ./output/labeled_fish_crops] \
        --checklist andaman_checklist.csv \
        [--secondary-taxonomy ./output/fishbase_genera.csv] \
        [--gbif-cache ./output/.gbif_cache.json] \
        --epochs 15 --batch-size 32 \
        --out models/checkpoints/fish-classifier-1.pth \
        [--label-map-out models/checkpoints/fish-classifier-labelmaps.json] \
        [--amp] [--backbone /path/to/fish-classifier-0.pth]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from taxonomy import TaxonomyResolver, canonical_species

try:
    from model import TaxonomicMultiHead
except ImportError as exc:
    raise SystemExit(
        f"Could not import TaxonomicMultiHead from model.py: {exc}\n"
        f"Expected signature: __init__(backbone_path, num_species, "
        f"num_genera, num_families), forward(x) -> (sp, gn, fa) logits."
    ) from exc

# Match track_cleaning.py's loss weights so the two stages optimize the same objective.
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ==========================================
# Scan
# ==========================================
def collect_folder_names(root_dir):
    """Return sorted list of immediate subdirectory names under root_dir."""
    root = Path(root_dir)
    if not root.is_dir():
        raise SystemExit(f"Data directory not found: {root}")
    return [d.name for d in sorted(root.iterdir()) if d.is_dir()]


def prewarm_gbif_for_missing_genera(folder_names, resolver):
    """
    Derive a genus from each folder name the same way resolve() would,
    filter out those already covered by the checklist or cache, and prewarm
    GBIF for the rest.  Runs before image scanning so training never blocks
    on network I/O.  Folder names may come from any number of data dirs —
    duplicates are collapsed by the set.
    """
    missing = set()
    for name in folder_names:
        sp_key = canonical_species(name)
        sp_space = sp_key.replace("_", " ")
        if not sp_space:
            continue
        genus = sp_space.split()[0].title()
        g_lc = genus.lower()
        if not g_lc or g_lc in ("unknown", "unidentified"):
            continue
        if g_lc in resolver.genus_to_family:
            continue
        missing.add(genus)

    if not missing:
        if resolver.verbose:
            print(" ➔ All reference genera already resolved by checklist/cache.")
        return

    resolver.prewarm(missing, batch_log_every=25)


def scan_one_data_dir(root_dir, resolver):
    """
    Walk a single data directory (one subfolder per species), normalize
    folder names via canonical_species, and return a flat list of
    (img_path, sp_key, genus, family) records.

    Raises SystemExit only if `root_dir` itself is missing — an individual
    species folder with no images is silently skipped, because the crop
    corpus can legitimately contain species the model has never seen and
    empty folders from a crashed session.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise SystemExit(f"Data directory not found: {root}")

    records = []
    skipped_dirs = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        sp_key, genus, family = resolver.resolve(d.name)
        if not sp_key:
            skipped_dirs += 1
            continue
        imgs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if not imgs:
            skipped_dirs += 1
            continue
        for p in imgs:
            records.append((str(p), sp_key, genus, family))

    return records, skipped_dirs


def scan_all_data_dirs(data_dirs, resolver):
    """
    Merge every --data-dir root into a single (samples, sp_to_idx, gn_to_idx,
    fa_to_idx) tuple.  Species with the same canonical name across two roots
    collapse into one class; anything new in the crop corpus that isn't in
    the reference set becomes an additional class.

    Returns (samples, sp_to_idx, gn_to_idx, fa_to_idx).
    """
    all_records = []
    for d in data_dirs:
        recs, skipped = scan_one_data_dir(d, resolver)
        note = f" ({skipped} empty/unresolvable folder(s) skipped)" if skipped else ""
        print(f"   {d}: {len(recs)} image(s){note}")
        all_records.extend(recs)

    if not all_records:
        raise SystemExit("No images found across any --data-dir.")

    sp_set = {r[1] for r in all_records}
    gn_set = {r[2] for r in all_records}
    fa_set = {r[3] for r in all_records}

    if len(sp_set) < 2:
        raise SystemExit(
            f"Need ≥2 distinct species across all --data-dir roots; "
            f"found {len(sp_set)}."
        )

    sp_to_idx = {s: i for i, s in enumerate(sorted(sp_set))}
    gn_to_idx = {g: i for i, g in enumerate(sorted(gn_set))}
    fa_to_idx = {f: i for i, f in enumerate(sorted(fa_set))}

    samples = [
        (path, sp_to_idx[sp], gn_to_idx[gn], fa_to_idx[fa])
        for path, sp, gn, fa in all_records
    ]
    return samples, sp_to_idx, gn_to_idx, fa_to_idx


# ==========================================
# Dataset
# ==========================================
class MultiHeadFishDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sp, gn, fa = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, sp, gn, fa


# ==========================================
# Split
# ==========================================
def stratified_split(samples, val_fraction, seed=42):
    """
    Train/val split stratified by species index.

    A plain random split is dangerous once the crop corpus is merged in:
    a species with only a handful of crops (say three Andaman fish) could
    otherwise land entirely in train or entirely in val, making the val
    score either falsely optimistic or meaningless.  Stratifying keeps
    every species represented in val whenever it has ≥2 samples, and
    drops rare single-sample species into train only.

    Returns (train_samples, val_samples), both shuffled.
    """
    by_species = defaultdict(list)
    for s in samples:
        by_species[s[1]].append(s)

    rng = torch.Generator().manual_seed(seed)
    train, val = [], []
    for sp_idx in sorted(by_species):
        items = by_species[sp_idx]
        perm = torch.randperm(len(items), generator=rng).tolist()
        shuffled = [items[i] for i in perm]
        if len(shuffled) == 1:
            train.extend(shuffled)
            continue
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)  # always keep ≥1 in train
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    # Re-shuffle both sets so batches aren't species-ordered.
    train_perm = torch.randperm(len(train), generator=rng).tolist()
    val_perm = torch.randperm(len(val), generator=rng).tolist()
    return [train[i] for i in train_perm], [val[i] for i in val_perm]


# ==========================================
# Training
# ==========================================
def class_weights(labels, n_classes):
    """Inverse-frequency weights, clamped, on DEVICE — matches track_cleaning."""
    counts = torch.bincount(
        torch.tensor(labels, dtype=torch.long), minlength=n_classes
    ).float()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.clamp(w.to(DEVICE), max=10.0)


def run_epoch(
    model, loader, optimizer, weights, scaler, train, use_amp, ignore_family_idx=None
):
    """
    One pass over `loader`.  `ignore_family_idx`, when set, excludes that
    family class from the family loss — used to keep the 'Unknown_Family'
    bucket from dominating and to avoid training the head to predict it.
    """
    w_sp, w_gn, w_fa = weights
    model.train() if train else model.eval()

    total_loss, total_seen = 0.0, 0
    sp_ok = gn_ok = fa_ok = fa_seen = 0

    with torch.set_grad_enabled(train):
        for imgs, y_sp, y_gn, y_fa in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            y_sp = y_sp.to(DEVICE)
            y_gn = y_gn.to(DEVICE)
            y_fa = y_fa.to(DEVICE)

            if train:
                optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits_sp, logits_gn, logits_fa = model(imgs)
                loss_sp = nn.functional.cross_entropy(logits_sp, y_sp, weight=w_sp)
                loss_gn = nn.functional.cross_entropy(logits_gn, y_gn, weight=w_gn)
                if ignore_family_idx is not None:
                    mask = y_fa != ignore_family_idx
                    if mask.any():
                        loss_fa = nn.functional.cross_entropy(
                            logits_fa[mask], y_fa[mask], weight=w_fa
                        )
                    else:
                        loss_fa = logits_fa.sum() * 0.0
                else:
                    loss_fa = nn.functional.cross_entropy(logits_fa, y_fa, weight=w_fa)
                loss = loss_sp + LAMBDA_GENUS * loss_gn + LAMBDA_FAMILY * loss_fa

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_seen += bs
            sp_ok += (logits_sp.argmax(1) == y_sp).sum().item()
            gn_ok += (logits_gn.argmax(1) == y_gn).sum().item()
            if ignore_family_idx is not None:
                m = y_fa != ignore_family_idx
                fa_ok += ((logits_fa.argmax(1) == y_fa) & m).sum().item()
                fa_seen += m.sum().item()
            else:
                fa_ok += (logits_fa.argmax(1) == y_fa).sum().item()
                fa_seen += bs

    n = max(total_seen, 1)
    fa_n = max(fa_seen, 1)
    return total_loss / n, sp_ok / n, gn_ok / n, fa_ok / fa_n


# ==========================================
# Checkpoint
# ==========================================
def save_multhead_checkpoint(
    model, path, sp_to_idx, gn_to_idx, fa_to_idx, n_sp, n_gn, n_fa
):
    """
    Writes the exact format track_cleaning.save_checkpoint used to produce
    so the two are interchangeable on disk.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "sp_to_idx": sp_to_idx,
            "gn_to_idx": gn_to_idx,
            "fa_to_idx": fa_to_idx,
            "num_species": n_sp,
            "num_genera": n_gn,
            "num_families": n_fa,
        },
        path,
    )


def save_label_map_json(path, sp_to_idx, gn_to_idx, fa_to_idx):
    """
    Persist the label maps as a JSON sidecar next to the checkpoint, in the
    exact schema track_cleaning.load_label_maps() expects.  Written before
    training starts so a crash mid-run still leaves a usable map behind.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "sp_to_idx": sp_to_idx,
                "gn_to_idx": gn_to_idx,
                "fa_to_idx": fa_to_idx,
            },
            fh,
            indent=2,
        )


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a TaxonomicMultiHead fish classifier. "
        "Pass --data-dir multiple times to merge reference imagery with a "
        "track_cleaning.py crop corpus."
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        required=True,
        help="Folder of per-species subfolders of images. May be repeated; "
        "every root is scanned and merged into one training set. Typical "
        "use: --data-dir ./reference_images "
        "--data-dir ./output/labeled_fish_crops",
    )
    parser.add_argument(
        "--checklist",
        required=True,
        help="Andaman checklist CSV (species[, genus, family]).",
    )
    parser.add_argument(
        "--secondary-taxonomy",
        default=None,
        help="Optional broader taxonomy CSV (genus,family or "
        "species,family) consulted before GBIF.",
    )
    parser.add_argument(
        "--gbif-cache",
        default="./output/.gbif_cache.json",
        help="JSON cache for GBIF genus→family results.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out",
        default="models/checkpoints/fish-classifier-1.pth",
        help="Checkpoint path, loadable directly by track_cleaning.py.",
    )
    parser.add_argument(
        "--label-map-out",
        default=None,
        help="Where to write the JSON label-map sidecar. Defaults to "
        "fish-classifier-labelmaps.json in the same directory as --out, "
        "matching what track_cleaning.py loads.",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Optional ImageNet-pretrained state_dict to warm-start "
        "the trunk (e.g. an old fish-classifier-0.pth).",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Taxonomy resolver (checklist + secondary CSV + GBIF cache)
    # ---------------------------------------------------------------
    resolver = TaxonomyResolver(
        args.checklist,
        secondary_csv=args.secondary_taxonomy,
        cache_path=args.gbif_cache,
        verbose=True,
    )

    # ---------------------------------------------------------------
    # Pass 1: folder discovery across every --data-dir root
    # ---------------------------------------------------------------
    print(f"\n📂 Scanning {len(args.data_dir)} data source(s)…")
    all_folder_names = []
    for d in args.data_dir:
        names = collect_folder_names(d)
        print(f"   {d}: {len(names)} species folder(s)")
        all_folder_names.extend(names)

    # ---------------------------------------------------------------
    # Pass 1.5: prewarm GBIF for any genus not already resolved
    # ---------------------------------------------------------------
    prewarm_gbif_for_missing_genera(all_folder_names, resolver)

    # ---------------------------------------------------------------
    # Pass 2: build samples now that every genus is resolvable locally
    # ---------------------------------------------------------------
    samples, sp_to_idx, gn_to_idx, fa_to_idx = scan_all_data_dirs(
        args.data_dir, resolver
    )
    resolver.finalize()

    n_sp, n_gn, n_fa = len(sp_to_idx), len(gn_to_idx), len(fa_to_idx)
    ignore_family_idx = fa_to_idx.get("Unknown_Family")

    unknown_family = (
        sum(1 for _, _, _, fa in samples if fa == ignore_family_idx)
        if ignore_family_idx is not None
        else 0
    )
    if unknown_family:
        pct = 100.0 * unknown_family / max(len(samples), 1)
        print(
            f" ⚠ {unknown_family} image(s) ({pct:.1f}%) still have no family "
            f"after all resolution layers — they train the species and genus "
            f"heads but are excluded from the family loss."
        )

    print(f"\nClasses: sp={n_sp}  gn={n_gn}  fa={n_fa}  |  images: {len(samples)}")

    # ---------------------------------------------------------------
    # Write the label-map sidecar now, before any training happens, so a
    # crash mid-run still leaves track_cleaning.py something to load.
    # ---------------------------------------------------------------
    label_map_path = args.label_map_out or os.path.join(
        os.path.dirname(args.out) or ".",
        "fish-classifier-labelmaps.json",
    )
    save_label_map_json(label_map_path, sp_to_idx, gn_to_idx, fa_to_idx)
    print(f" 💾 Label maps written to {label_map_path}")

    # ---------------------------------------------------------------
    # Stratified train/val split (species-balanced across both sides)
    # ---------------------------------------------------------------
    train_samples, val_samples = stratified_split(samples, args.val_fraction)
    if not val_samples:
        raise SystemExit("Stratified split produced an empty val set.")
    print(
        f" Split: {len(train_samples)} train / {len(val_samples)} val "
        f"({len(train_samples)} seen per epoch)"
    )

    train_loader = DataLoader(
        MultiHeadFishDataset(train_samples, TRAIN_TRANSFORM),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        MultiHeadFishDataset(val_samples, EVAL_TRANSFORM),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = TaxonomicMultiHead(
        backbone_path=args.backbone,
        num_species=n_sp,
        num_genera=n_gn,
        num_families=n_fa,
    ).to(DEVICE)
    print(f" ➔ Device: {DEVICE}")

    w_sp = class_weights([s for _, s, _, _ in train_samples], n_sp)
    w_gn = class_weights([g for _, _, g, _ in train_samples], n_gn)
    w_fa = class_weights([f for _, _, _, f in train_samples], n_fa)
    weights = (w_sp, w_gn, w_fa)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    use_amp = args.amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            scaler,
            train=True,
            use_amp=use_amp,
            ignore_family_idx=ignore_family_idx,
        )
        va = run_epoch(
            model,
            val_loader,
            optimizer,
            weights,
            scaler,
            train=False,
            use_amp=use_amp,
            ignore_family_idx=ignore_family_idx,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tr[0]:.4f} sp {tr[1]:.3f} gn {tr[2]:.3f} fa {tr[3]:.3f} | "
            f"val   loss {va[0]:.4f} sp {va[1]:.3f} gn {va[2]:.3f} fa {va[3]:.3f}"
        )

        score = va[1] + 0.3 * va[2] + 0.1 * va[3]
        if score >= best_val:
            best_val = score
            save_multhead_checkpoint(
                model,
                args.out,
                sp_to_idx,
                gn_to_idx,
                fa_to_idx,
                n_sp,
                n_gn,
                n_fa,
            )
            print(f"  💾 Saved ({args.out}) | composite val score {score:.3f}")

    print(f"\nDone. Best composite val score: {best_val:.3f}")
    print(f"Checkpoint: {args.out}")
    print(f"Label maps: {label_map_path}")
    print("Loadable directly by track_cleaning.py.")


if __name__ == "__main__":
    main()
