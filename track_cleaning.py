"""
track_cleaning.py

Active-learning pipeline for Andaman reef fish identification.

Combines GUI path selection, per-track mass labeling, hierarchical
multi-head classification (species / genus / family), on-disk crop
storage, and periodic GPU retraining.

Usage:
    python track_cleaning.py
"""

import os
import re
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import TaxonomicMultiHead
from taxonomy import TaxonomyResolver

# ==========================================
# Configuration
# ==========================================
OUTPUT_CROP_DIR = "./output/labeled_fish_crops"
CHECKPOINT_DIR = "./models/checkpoints"
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
        # Extract relative path following 'annotated_videos'
        idx = parts.index("annotated_videos")
        relative_path = Path(*parts[idx + 1 :])
    else:
        # Fallback if 'annotated_videos' is not in the path stem
        relative_path = Path(abs_input.name)

    target_path = Path(output_base_dir) / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


# ==========================================
# 1. GUI path selection
# ==========================================
def get_user_paths():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    csv_path = filedialog.askopenfilename(
        title="Select Master Tracking CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    if not csv_path:
        raise SystemExit("No CSV selected.")

    checklist_path = filedialog.askopenfilename(
        title="Select Andaman Checklist CSV (columns: species, genus, family)",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    if not checklist_path:
        raise SystemExit("No checklist selected.")

    root_data_dir = filedialog.askdirectory(
        title="Select Parent Folder Containing Sites / Frames Data"
    )
    if not root_data_dir:
        raise SystemExit("No frames directory selected.")

    root.destroy()
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
                    (
                        img_path,
                        sp_to_idx[sp_key],
                        gn_to_idx[genus],
                        fa_to_idx[family],
                    )
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
# 4. Retraining
# ==========================================
def retrain(model, resolver, epochs=EPOCHS_PER_RETRAIN):
    sp_to_idx, gn_to_idx, fa_to_idx = build_label_maps(OUTPUT_CROP_DIR, resolver)
    n_sp, n_gn, n_fa = len(sp_to_idx), len(gn_to_idx), len(fa_to_idx)

    if min(n_sp, n_gn, n_fa) < 2:
        print(
            f" ⏸ Skipping retrain — need ≥2 classes at every level "
            f"(sp={n_sp}, gn={n_gn}, fa={n_fa})."
        )
        return None, None, None

    ds = FishCropDataset(
        OUTPUT_CROP_DIR, resolver, sp_to_idx, gn_to_idx, fa_to_idx, TRAIN_TRANSFORM
    )
    if len(ds) < 4:
        print(f" ⏸ Skipping retrain — only {len(ds)} crops on disk.")
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
        print(f"   epoch {epoch + 1}/{epochs} | loss {avg:.4f}")

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
        return (
            model,
            ckpt["sp_to_idx"],
            ckpt["gn_to_idx"],
            ckpt["fa_to_idx"],
            True,
        )

    model = TaxonomicMultiHead(
        backbone_path=backbone_path,
        num_species=2,
        num_genera=2,
        num_families=2,
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
# 5. Main loop
# ==========================================
def execute_pipeline():
    csv_path, checklist_path, root_data_dir = get_user_paths()

    # Calculate output path preserving subfolder structure from annotated_videos
    out_csv_path = get_output_csv_path(csv_path, CSV_OUT)

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
            df[col] = np.nan

    backbone_path = "pretrained_backbone.pth"
    backbone_arg = backbone_path if os.path.exists(backbone_path) else None
    model, sp_to_idx, gn_to_idx, fa_to_idx, is_trained = build_or_load_model(
        backbone_arg
    )
    sp_inv = {i: s for s, i in sp_to_idx.items()}
    gn_inv = {i: g for g, i in gn_to_idx.items()}
    fa_inv = {i: f for f, i in fa_to_idx.items()}

    action_counter = 0

    print("\n=======================================================")
    print("      🚀 ACTIVE LEARNING SYSTEM INITIALIZED 🚀")
    print("=======================================================")
    print("Instructions:")
    print("  ➔ Type the true species name and press Enter.")
    print("  ➔ Leave blank to accept the auto-suggestion.")
    print("  ➔ Type 'exit' to retrain (if needed), save, and close.")
    print("=======================================================\n")

    for idx in range(len(df)):
        if pd.notna(df.at[idx, "assigned_species"]):
            continue

        row = df.iloc[idx]
        img_path = locate_frame_path(frame_index, row["frame"])
        if not img_path or not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        img_tensor = EVAL_TRANSFORM(pil).unsqueeze(0)

        suggestion, source_type, model_genus, model_family = None, "", None, None
        if is_trained:
            sp, gn, fa, c_sp, c_gn, c_fa = predict(
                model, img_tensor, sp_inv, gn_inv, fa_inv
            )
            suggestion = clean_label_string(sp)
            model_genus = gn
            model_family = fa
            source_type = f"🤖 Model sp={c_sp * 100:.0f}% gn={c_gn * 100:.0f}% fa={c_fa * 100:.0f}%"

        display_frame = cv2.resize(cropped, (400, 400))
        cv2.putText(
            display_frame,
            f"Frame: {int(row['frame'])} | Track ID: {int(row['id'])}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Active Classification Workspace", display_frame)
        cv2.waitKey(10)

        print(f"\n[Index: {idx}] Frame: {os.path.basename(img_path)}")
        if suggestion:
            print(f"  Recommendation: {suggestion}  | Source: {source_type}")

        user_input = input("Enter species tag: ").strip().lower()

        if user_input == "exit":
            break

        if user_input == "" and suggestion:
            final_species = clean_label_string(suggestion)
        elif user_input != "":
            final_species = clean_label_string(user_input)
        else:
            print(" ❌ Input required — no valid recommendation exists yet.")
            continue

        _, final_genus, final_family = resolver.resolve(final_species)

        if final_family == "Unknown_Family" and model_family:
            final_family = model_family
        if final_genus == "Unknown" and model_genus:
            final_genus = model_genus

        track_mask = (df["id"] == row["id"]) & (df["assigned_species"].isna())
        n_matched = int(track_mask.sum())
        df.loc[track_mask, "assigned_species"] = final_species
        df.loc[track_mask, "assigned_genus"] = final_genus
        df.loc[track_mask, "assigned_family"] = final_family
        print(
            f" 💾 Applied labels to {n_matched} frames for Track ID {int(row['id'])}."
        )

        species_dir = os.path.join(OUTPUT_CROP_DIR, final_species)
        os.makedirs(species_dir, exist_ok=True)
        crop_path = os.path.join(species_dir, f"crop_idx{idx}_id{int(row['id'])}.jpg")
        cv2.imwrite(crop_path, cropped)

        action_counter += 1

        if action_counter % RETRAIN_INTERVAL == 0:
            print(f"\n🔄 Milestone ({action_counter} actions). Retraining...")
            result = retrain(model, resolver)
            if result[0] is not None:
                sp_to_idx, gn_to_idx, fa_to_idx = result
                sp_inv = {i: s for s, i in sp_to_idx.items()}
                gn_inv = {i: g for g, i in gn_to_idx.items()}
                fa_inv = {i: f for f, i in fa_to_idx.items()}
                is_trained = True
                save_checkpoint(model, sp_to_idx, gn_to_idx, fa_to_idx)
            df.to_csv(out_csv_path, index=False)
            print(f" 💾 CSV flushed to {out_csv_path}. Resuming...\n")

    if action_counter > 0:
        print("\n🏁 Session ending. Running final retrain...")
        result = retrain(model, resolver)
        if result[0] is not None:
            sp_to_idx, gn_to_idx, fa_to_idx = result
            save_checkpoint(model, sp_to_idx, gn_to_idx, fa_to_idx)
        else:
            save_checkpoint(model, sp_to_idx, gn_to_idx, fa_to_idx)

    df.to_csv(out_csv_path, index=False)
    cv2.destroyAllWindows()
    print(f"🏁 Session closed. Progress saved to: {out_csv_path}")


if __name__ == "__main__":
    execute_pipeline()
