"""
train.py

Pretrain a TaxonomicMultiHead model on reference imagery (GBIF /
iNaturalist folders from download_images.py, or local copies of FishWIO /
WildFish / Fish4Knowledge arranged as one folder per species).

Unlike the previous single-head version, this produces a checkpoint that is
*directly loadable* by track_cleaning.py: the same state_dict format, the
same label-map schema, and the same three-head architecture.  The active
learning loop then refines those heads on user-confirmed crops.

Species folder names are normalized via taxonomy.canonical_species, so
'Lutjanus decussatus', 'Lutjanus_decussatus', and 'lutjanus_decussatus'
all map to the same class.  Genus and family are derived from the checklist
via TaxonomyResolver; species absent from the checklist still get a
genus/family estimate (first-token genus, 'Unknown_Family' family) so the
model can learn from regional out-of-checklist imagery.

Usage:
    python train.py \
        --data-dir ./reference_images \
        --checklist andaman_checklist.csv \
        --epochs 15 --batch-size 32 \
        --out models/checkpoints/fish-classifier-1.pth \
        [--amp]
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
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
# Dataset: ImageFolder-lite with canonical labels
# ==========================================
def scan_reference_folders(root_dir, resolver):
    """
    Walk root_dir (one subfolder per species), normalize folder names via
    canonical_species, and derive the three label maps.

    Returns (samples, sp_to_idx, gn_to_idx, fa_to_idx) where samples is a
    list of (img_path, sp_idx, gn_idx, fa_idx).
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise SystemExit(f"Reference directory not found: {root}")

    sp_set, gn_set, fa_set = set(), set(), set()
    records = []  # (path, species_key, genus, family)
    skipped_dirs = 0

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        sp_key, genus, family = resolver.resolve(d.name)
        if not sp_key:
            skipped_dirs += 1
            continue
        n_imgs = 0
        for img in sorted(d.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                records.append((str(img), sp_key, genus, family))
                n_imgs += 1
        if n_imgs == 0:
            skipped_dirs += 1
            continue
        sp_set.add(sp_key)
        gn_set.add(genus)
        fa_set.add(family)

    if len(sp_set) < 2:
        raise SystemExit(
            f"Need ≥2 non-empty species folders under {root}; found {len(sp_set)}."
        )
    if skipped_dirs:
        print(f" ⚠ Skipped {skipped_dirs} folder(s) with no images or empty names.")

    sp_to_idx = {s: i for i, s in enumerate(sorted(sp_set))}
    gn_to_idx = {g: i for i, g in enumerate(sorted(gn_set))}
    fa_to_idx = {f: i for i, f in enumerate(sorted(fa_set))}

    samples = [
        (path, sp_to_idx[sp], gn_to_idx[gn], fa_to_idx[fa])
        for path, sp, gn, fa in records
    ]
    return samples, sp_to_idx, gn_to_idx, fa_to_idx


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


def run_epoch(model, loader, optimizers_and_loss, scaler, train, use_amp):
    optimizer, w_sp, w_gn, w_fa = optimizers_and_loss
    model.train() if train else model.eval()

    total_loss, total_seen = 0.0, 0
    sp_correct = gn_correct = fa_correct = 0

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
                loss = (
                    nn.functional.cross_entropy(logits_sp, y_sp, weight=w_sp)
                    + LAMBDA_GENUS
                    * nn.functional.cross_entropy(logits_gn, y_gn, weight=w_gn)
                    + LAMBDA_FAMILY
                    * nn.functional.cross_entropy(logits_fa, y_fa, weight=w_fa)
                )

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
            sp_correct += (logits_sp.argmax(1) == y_sp).sum().item()
            gn_correct += (logits_gn.argmax(1) == y_gn).sum().item()
            fa_correct += (logits_fa.argmax(1) == y_fa).sum().item()

    n = max(total_seen, 1)
    return total_loss / n, sp_correct / n, gn_correct / n, fa_correct / n


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a TaxonomicMultiHead fish classifier."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder of per-species subfolders of reference images.",
    )
    parser.add_argument(
        "--checklist",
        required=True,
        help="Andaman checklist CSV (species[, genus, family]).",
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
        "--backbone",
        default=None,
        help="Optional ImageNet-pretrained state_dict to warm-start the trunk "
        "(e.g. an old fish-classifier-0.pth). Omit for ImageNet init.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    args = parser.parse_args()

    resolver = TaxonomyResolver(args.checklist)

    print(f"📂 Scanning {args.data_dir}…")
    samples, sp_to_idx, gn_to_idx, fa_to_idx = scan_reference_folders(
        args.data_dir, resolver
    )
    n_sp, n_gn, n_fa = len(sp_to_idx), len(gn_to_idx), len(fa_to_idx)

    # Report any species whose genus/family came from the fallback rather than
    # the checklist — good to know before you trust the genus/family heads.
    unknown_family = sum(
        1 for _, _, _, fa in samples if list(fa_to_idx.keys())[fa] == "Unknown_Family"
    )
    if unknown_family:
        pct = 100.0 * unknown_family / len(samples)
        print(
            f" ⚠ {unknown_family} image(s) ({pct:.1f}%) have no checklist "
            f"family; they will still train the species and genus heads."
        )

    print(f"Classes: sp={n_sp}  gn={n_gn}  fa={n_fa}  |  images: {len(samples)}")

    # Split before wrapping in datasets so train/val see disjoint indices.
    val_size = max(1, int(len(samples) * args.val_fraction))
    train_size = len(samples) - val_size
    if train_size < 1:
        raise SystemExit("Not enough images to split into train/val.")

    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(samples), generator=gen).tolist()
    train_samples = [samples[i] for i in perm[:train_size]]
    val_samples = [samples[i] for i in perm[train_size:]]

    train_ds = MultiHeadFishDataset(train_samples, TRAIN_TRANSFORM)
    val_ds = MultiHeadFishDataset(val_samples, EVAL_TRANSFORM)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(DEVICE.type == "cuda"),
    )

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

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    use_amp = args.amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    bundle = (optimizer, w_sp, w_gn, w_fa)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_sp, tr_gn, tr_fa = run_epoch(
            model, train_loader, bundle, scaler, train=True, use_amp=use_amp
        )
        va_loss, va_sp, va_gn, va_fa = run_epoch(
            model, val_loader, bundle, scaler, train=False, use_amp=use_amp
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} "
            f"sp {tr_sp:.3f} gn {tr_gn:.3f} fa {tr_fa:.3f} | "
            f"val loss {va_loss:.4f} "
            f"sp {va_sp:.3f} gn {va_gn:.3f} fa {va_fa:.3f}"
        )

        # Model selection: species accuracy dominates, genus/family as tiebreakers.
        score = va_sp + 0.3 * va_gn + 0.1 * va_fa
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
    print(
        "This file is directly loadable by track_cleaning.py — run it now "
        "and the GUI will make predictions from frame one."
    )


def save_multhead_checkpoint(
    model, path, sp_to_idx, gn_to_idx, fa_to_idx, n_sp, n_gn, n_fa
):
    """
    Writes the exact format track_cleaning.save_checkpoint produces so the two
    are interchangeable on disk.
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


if __name__ == "__main__":
    main()
