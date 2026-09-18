"""
train.py

Pretrain a ResNet18 backbone on reference imagery (GBIF / iNaturalist
folders from download_images.py, or local copies of FishWIO / WildFish /
Fish4Knowledge arranged as one folder per class).

Only the convolutional weights from the output .pth are consumed by the
multi-head model in model.py — the final `fc` layer is discarded at load
time, so the exact class count here does not need to match downstream.

Usage:
    python train.py --data-dir ./reference_images \
        --epochs 15 --batch-size 32 --out pretrained_backbone.pth [--amp]
"""

import argparse
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, use_amp=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_seen = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_seen += inputs.size(0)

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a fish classification backbone."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/fish-classifier-0.pth")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    args = parser.parse_args()

    full_train_ds = datasets.ImageFolder(args.data_dir, transform=train_transform)
    full_eval_ds = datasets.ImageFolder(args.data_dir, transform=eval_transform)

    num_classes = len(full_train_ds.classes)
    if num_classes < 2:
        raise SystemExit(f"Need ≥2 species folders; found {num_classes}.")

    val_size = max(1, int(len(full_train_ds) * args.val_fraction))
    train_size = len(full_train_ds) - val_size
    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(
        range(len(full_train_ds)), [train_size, val_size], generator=gen
    )
    train_subset = torch.utils.data.Subset(full_train_ds, train_idx.indices)
    val_subset = torch.utils.data.Subset(full_eval_ds, val_idx.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Classes ({num_classes}): {full_train_ds.classes}")
    print(f"Train: {train_size} | Val: {val_size} | Device: {device}")

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  💾 Saved best backbone ({args.out}), val acc {val_acc:.3f}")

    classes_path = args.out.replace(".pth", "_classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_train_ds.classes))
    print(f"\nDone. Best val acc: {best_val_acc:.3f}. Class order: {classes_path}")


if __name__ == "__main__":
    main()
