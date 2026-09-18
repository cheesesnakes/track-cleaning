"""
download_images.py

Pull reference imagery per species from GBIF and iNaturalist, OR ingest a
pre-downloaded dataset (FishWIO / WildFish / Fish4Knowledge) from a local
directory.

Input CSV must have a 'species' column (scientific name).

Output:
    reference_images/<species_underscored>/<source>_<id>.jpg
    reference_images/provenance_log.csv

Ingest mode:
    --ingest-dir PATH  --source-name fishwio
    Scans PATH for class-named subdirectories (one per species), copies
    images into reference_images/, and logs provenance.

Cropping (on by default): every image, whether fetched from GBIF/iNat or
ingested from a local dataset, is cropped to the highest-confidence fish
detection (+ padding) using DETECTOR, a global YOLO fish-presence model
loaded once at startup from DETECTOR_WEIGHTS_PATH below (or --detector-weights).
GBIF/iNat images especially are rarely tightly framed on the fish — lots of
background reef, divers, boats, hands, signage — which will otherwise leak
into a species classifier.

If the detector fails to load (missing weights, bad path, missing
ultralytics/opencv), DETECTOR stays None and the script falls back to
saving images uncropped rather than failing the whole run.

Images with no detection are logged as crop_status=no_detection and, by
default, NOT saved, since an undetected "fish" photo is usually not
actually a clean fish image (habitat shot, diagram, fish too small/occluded).
Pass --keep-undetected to save them uncropped anyway.

Usage:
    python download_images.py --species-csv species_list.csv \
        --out-dir ./reference_images --max-per-species 40 \
        --detector-weights ./weights/fish_static_best.pt
    python download_images.py --species-csv species_list.csv \
        --ingest-dir /path/to/FishWIO --source-name fishwio
"""

import argparse
import csv
import shutil
import time
from pathlib import Path

import requests

GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
INAT_OBS_URL = "https://api.inaturalist.org/v1/observations"
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 1.0

PROVENANCE_FIELDS = [
    "species",
    "source",
    "image_id",
    "url",
    "license",
    "download_status",
    "crop_status",
    "crop_bbox",
    "crop_conf",
]

# --------------------------------------------------------------------------
# Detector (global, loaded once)
# --------------------------------------------------------------------------

DETECTOR_WEIGHTS_PATH = "./weights/fish_static_best.pt"
CROP_IMGSZ = 1280

# Global holding the loaded detector + its crop settings, or None if no
# detector is available. Every cropping call guards on this being None.
DETECTOR = None


def load_detector(weights_path=DETECTOR_WEIGHTS_PATH, conf=0.25, pad_frac=0.15):
    """
    Loads a YOLO fish-presence detector into the global DETECTOR.
    On any failure (missing file, missing dependency, bad weights),
    leaves DETECTOR as None and prints a warning instead of raising,
    so the rest of the script can still run uncropped.
    """
    global DETECTOR

    try:
        from ultralytics import YOLO

        model = YOLO(weights_path)
        DETECTOR = {"model": model, "conf": conf, "pad_frac": pad_frac}
        print(f"Loaded fish detector for cropping: {weights_path}")
    except Exception as exc:  # noqa: BLE001 - any load failure should degrade gracefully
        print(f"⚠ Could not load fish detector from {weights_path}: {exc}")
        print("  Continuing without cropping — images will be saved uncropped.")
        DETECTOR = None


def clean_folder_name(name):
    return name.strip().replace(" ", "_").lower()


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------


def crop_image(image_path):
    """
    Returns (crop_array_or_None, status, bbox_str, best_conf).
    status is one of: "ok", "no_detection", "read_error", "no_detector".
    crop_array is a numpy BGR array (OpenCV-style) ready to write.

    Guard: if DETECTOR is None, returns immediately with status
    "no_detector" and does no image I/O or inference.
    """
    if DETECTOR is None:
        return None, "no_detector", "", 0.0

    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None, "read_error", "", 0.0

    h, w = img.shape[:2]
    results = DETECTOR["model"].predict(
        img, imgsz=CROP_IMGSZ, conf=DETECTOR["conf"], verbose=False
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, "no_detection", "", 0.0

    # Take the single highest-confidence detection.
    confs = boxes.conf.tolist()
    best_idx = max(range(len(confs)), key=lambda i: confs[i])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    best_conf = confs[best_idx]

    # Pad the box, then clamp to image bounds.
    pad_frac = DETECTOR["pad_frac"]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None, "no_detection", "", 0.0

    crop = img[y1:y2, x1:x2]
    bbox_str = f"{x1},{y1},{x2},{y2}"
    return crop, "ok", bbox_str, round(best_conf, 3)


def crop_and_save(src_path, dest_path, keep_undetected=False):
    """
    Crops src_path and writes the result to dest_path.
    Returns (status, bbox_str, conf).

    Guard: if DETECTOR is None, always falls back to copying the
    original uncropped (equivalent to cropping being unavailable, not
    "no fish found") and returns status "no_detector".
    """
    if DETECTOR is None:
        shutil.copy2(src_path, dest_path)
        return "no_detector", "", 0.0

    import cv2

    crop, status, bbox_str, conf = crop_image(src_path)
    if status == "ok":
        cv2.imwrite(str(dest_path), crop)
        return status, bbox_str, conf

    if keep_undetected:
        shutil.copy2(src_path, dest_path)
        return status + "_kept_uncropped", bbox_str, conf

    return status + "_skipped", bbox_str, conf


# --------------------------------------------------------------------------
# Remote fetch
# --------------------------------------------------------------------------


def fetch_gbif_images(species_name, max_images, session):
    results = []
    params = {
        "scientificName": species_name,
        "mediaType": "StillImage",
        "limit": min(max_images, 300),
    }
    try:
        resp = session.get(GBIF_OCCURRENCE_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  ⚠ GBIF query failed for {species_name}: {exc}")
        return results

    for record in data.get("results", []):
        for media in record.get("media", []):
            if media.get("type") != "StillImage" or not media.get("identifier"):
                continue
            results.append(
                {
                    "source": "gbif",
                    "image_id": record.get("key"),
                    "url": media["identifier"],
                    "license": record.get("license", ""),
                }
            )
            if len(results) >= max_images:
                return results
    return results


def fetch_inat_images(species_name, max_images, session):
    results = []
    params = {
        "taxon_name": species_name,
        "quality_grade": "research",
        "photos": "true",
        "per_page": min(max_images, 200),
    }
    try:
        resp = session.get(INAT_OBS_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  ⚠ iNaturalist query failed for {species_name}: {exc}")
        return results

    for obs in data.get("results", []):
        for photo in obs.get("photos", []):
            url = photo.get("url", "").replace("square", "medium")
            if not url:
                continue
            results.append(
                {
                    "source": "inaturalist",
                    "image_id": photo.get("id"),
                    "url": url,
                    "license": photo.get("license_code", ""),
                }
            )
            if len(results) >= max_images:
                return results
    return results


def download_image(url, dest_path, session):
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return True
    except requests.RequestException as exc:
        print(f"    ⚠ Download failed ({url}): {exc}")
        return False


def ingest_local_dataset(ingest_dir, out_dir, source_name, log_writer, keep_undetected):
    """
    Copy a class-organized local dataset into reference_images/.
    Expects <ingest_dir>/<class_name>/<image>.jpg (ImageFolder layout).
    Every image is passed through crop_and_save, which itself guards on
    DETECTOR being None.
    """
    ingest_dir = Path(ingest_dir)
    if not ingest_dir.exists():
        print(f" ⚠ Ingest dir not found: {ingest_dir}")
        return

    copied = 0
    for species_dir in sorted(ingest_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        dest_species_dir = Path(out_dir) / clean_folder_name(species)
        dest_species_dir.mkdir(parents=True, exist_ok=True)

        for img_path in species_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            dest = dest_species_dir / f"{source_name}_{img_path.name}"

            crop_status, bbox_str, conf = crop_and_save(
                img_path, dest, keep_undetected=keep_undetected
            )

            log_writer.writerow(
                {
                    "species": species.replace("_", " "),
                    "source": source_name,
                    "image_id": img_path.stem,
                    "url": str(img_path),
                    "license": "see source dataset",
                    "download_status": "ok",
                    "crop_status": crop_status,
                    "crop_bbox": bbox_str,
                    "crop_conf": conf,
                }
            )
            if crop_status.endswith("_skipped"):
                continue
            copied += 1

    print(f"  ✔ Ingested {copied} images from {ingest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fetch / ingest reference images.")
    parser.add_argument("--species-csv", required=True)
    parser.add_argument("--out-dir", default="./output/reference_images")
    parser.add_argument("--max-per-species", type=int, default=40)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["gbif", "inaturalist"],
        default=["gbif", "inaturalist"],
    )
    parser.add_argument(
        "--ingest-dir",
        help="Optional: copy a local ImageFolder dataset into reference_images/.",
    )
    parser.add_argument(
        "--source-name",
        default="local",
        help="Tag applied to ingested images (e.g. 'fishwio', 'wildfish').",
    )
    parser.add_argument(
        "--detector-weights",
        default=DETECTOR_WEIGHTS_PATH,
        help=f"Path to YOLO fish-presence detector weights used for cropping "
        f"(default: {DETECTOR_WEIGHTS_PATH}). If loading fails, the script "
        "continues without cropping.",
    )
    parser.add_argument(
        "--crop-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for the crop detector (default 0.25).",
    )
    parser.add_argument(
        "--crop-pad-frac",
        type=float,
        default=0.15,
        help="Fractional padding added around the detected box (default 0.15).",
    )
    parser.add_argument(
        "--keep-undetected",
        action="store_true",
        help="If no fish is detected in an image, save it uncropped instead "
        "of skipping it.",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="Skip the GBIF/iNaturalist fetch loop entirely (e.g. for "
        "ingest-only runs against a local dataset).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "provenance_log.csv"

    with open(args.species_csv, newline="", encoding="utf-8") as f:
        species_rows = list(csv.DictReader(f))
    if not species_rows or "species" not in species_rows[0]:
        raise SystemExit("Input CSV must have a 'species' column.")

    # Cropping is on by default; load_detector guards internally and leaves
    # DETECTOR as None (cropping disabled, falls back to uncropped saves)
    # if the weights can't be loaded.
    load_detector(
        args.detector_weights, conf=args.crop_conf, pad_frac=args.crop_pad_frac
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "andaman-fish-classifier-pretraining/1.0"})

    write_header = not provenance_path.exists()
    with open(provenance_path, "a", newline="", encoding="utf-8") as log_f:
        log_writer = csv.DictWriter(log_f, fieldnames=PROVENANCE_FIELDS)
        if write_header:
            log_writer.writeheader()

        if args.ingest_dir:
            print(f"\n📥 Ingesting local dataset: {args.ingest_dir}")
            ingest_local_dataset(
                args.ingest_dir,
                out_dir,
                args.source_name,
                log_writer,
                args.keep_undetected,
            )

        if args.skip_remote:
            print("\n⏭  --skip-remote set, not querying GBIF/iNaturalist.")

        for row in [] if args.skip_remote else species_rows:
            species_name = row["species"].strip()
            if not species_name:
                continue

            species_dir = out_dir / clean_folder_name(species_name)
            species_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🔎 {species_name}")
            candidates = []
            per_source_cap = max(1, args.max_per_species // len(args.sources))

            if "gbif" in args.sources:
                candidates += fetch_gbif_images(species_name, per_source_cap, session)
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            if "inaturalist" in args.sources:
                candidates += fetch_inat_images(species_name, per_source_cap, session)
                time.sleep(SLEEP_BETWEEN_REQUESTS)

            if not candidates:
                print("  ✖ No images found from any source.")
                continue

            downloaded = 0
            cropped_ok = 0
            skipped_no_detection = 0
            for cand in candidates[: args.max_per_species]:
                fname = f"{cand['source']}_{cand['image_id']}.jpg"
                final_dest = species_dir / fname

                # Download to a temp path first, then crop (or copy, if
                # DETECTOR is None) into the final destination.
                tmp_dest = species_dir / f".tmp_{fname}"
                ok = download_image(cand["url"], tmp_dest, session)
                crop_status, bbox_str, conf = "not_attempted", "", 0.0
                if ok:
                    crop_status, bbox_str, conf = crop_and_save(
                        tmp_dest, final_dest, keep_undetected=args.keep_undetected
                    )
                    tmp_dest.unlink(missing_ok=True)
                    if crop_status == "ok":
                        cropped_ok += 1
                        downloaded += 1
                    elif crop_status == "no_detector":
                        downloaded += 1
                    elif crop_status.endswith("_kept_uncropped"):
                        downloaded += 1
                        skipped_no_detection += 1
                    else:
                        skipped_no_detection += 1

                log_writer.writerow(
                    {
                        "species": species_name,
                        "source": cand["source"],
                        "image_id": cand["image_id"],
                        "url": cand["url"],
                        "license": cand["license"],
                        "download_status": "ok" if ok else "failed",
                        "crop_status": crop_status,
                        "crop_bbox": bbox_str,
                        "crop_conf": conf,
                    }
                )
                time.sleep(0.2)

            msg = (
                f"  ✔ Downloaded {downloaded}/{len(candidates[: args.max_per_species])}"
            )
            if DETECTOR is not None:
                msg += f" (cropped: {cropped_ok}, no detection: {skipped_no_detection})"
            print(msg)

    print(f"\nDone. Provenance log: {provenance_path}")


if __name__ == "__main__":
    main()
