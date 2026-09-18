"""
scan_bad_images.py

One-time scan of an ImageFolder-style directory (one subfolder per class)
to find files that PIL can't open — truncated downloads, HTML error pages
saved with an image extension, zero-byte files, etc. This is common after
bulk scraping from GBIF/iNaturalist.

Bad files are MOVED to a sibling `_quarantine/<class>/` folder (not
deleted), so you can inspect or restore them, and so ImageFolder no longer
sees them at all — this is more reliable than try/except in the training
loop because it also drops the count of that class up front.

Usage:
    python scan_bad_images.py --data-dir ./reference_images
    python scan_bad_images.py --data-dir ./reference_images --dry-run
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def is_bad(path: Path) -> str | None:
    """Return a short reason string if the file is bad, else None."""
    try:
        if path.stat().st_size == 0:
            return "zero-byte file"
        with Image.open(path) as img:
            img.verify()  # cheap structural check, doesn't decode pixels
        # verify() closes the file handle internally; reopen to fully decode
        # one more time, since verify() misses some truncation cases
        with Image.open(path) as img:
            img.load()
    except UnidentifiedImageError:
        return "not a recognizable image (likely HTML/text saved as image)"
    except OSError as e:
        return f"truncated or unreadable ({e})"
    except Exception as e:  # noqa: BLE001 - catch-all so the scan never crashes
        return f"unexpected error ({e})"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report bad files, don't move them.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    quarantine_dir = data_dir.parent / f"{data_dir.name}_quarantine"

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    all_files = [
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    ]

    print(f"Scanning {len(all_files)} files under {data_dir} ...")

    bad = []
    for i, path in enumerate(all_files, 1):
        reason = is_bad(path)
        if reason:
            bad.append((path, reason))
        if i % 500 == 0:
            print(f"  ...{i}/{len(all_files)} checked, {len(bad)} bad so far")

    print(f"\nFound {len(bad)} bad file(s) out of {len(all_files)}.")

    # tally by class (immediate parent folder)
    per_class = {}
    for path, _ in bad:
        per_class[path.parent.name] = per_class.get(path.parent.name, 0) + 1
    for cls, count in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    if args.dry_run:
        print("\n--dry-run set: nothing moved. Bad files:")
        for path, reason in bad:
            print(f"  {path}  ({reason})")
        return

    for path, reason in bad:
        rel = path.relative_to(data_dir)
        dest = quarantine_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))

    print(f"\nMoved {len(bad)} bad file(s) to {quarantine_dir}")
    print("Re-run train.py once this scan reports 0 bad files.")


if __name__ == "__main__":
    main()
