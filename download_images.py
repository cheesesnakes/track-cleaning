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

Usage:
    python download_images.py --species-csv species_list.csv \
        --out-dir ./reference_images --max-per-species 40
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


def clean_folder_name(name):
    return name.strip().replace(" ", "_").lower()


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


def ingest_local_dataset(ingest_dir, out_dir, source_name, log_writer):
    """
    Copy a class-organized local dataset into reference_images/.
    Expects <ingest_dir>/<class_name>/<image>.jpg (ImageFolder layout).
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
            try:
                shutil.copy2(img_path, dest)
                log_writer.writerow(
                    {
                        "species": species.replace("_", " "),
                        "source": source_name,
                        "image_id": img_path.stem,
                        "url": str(img_path),
                        "license": "see source dataset",
                        "download_status": "ok",
                    }
                )
                copied += 1
            except OSError as exc:
                print(f"    ⚠ Copy failed ({img_path}): {exc}")

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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "provenance_log.csv"

    with open(args.species_csv, newline="", encoding="utf-8") as f:
        species_rows = list(csv.DictReader(f))
    if not species_rows or "species" not in species_rows[0]:
        raise SystemExit("Input CSV must have a 'species' column.")

    session = requests.Session()
    session.headers.update({"User-Agent": "andaman-fish-classifier-pretraining/1.0"})

    write_header = not provenance_path.exists()
    with open(provenance_path, "a", newline="", encoding="utf-8") as log_f:
        log_writer = csv.DictWriter(
            log_f,
            fieldnames=[
                "species",
                "source",
                "image_id",
                "url",
                "license",
                "download_status",
            ],
        )
        if write_header:
            log_writer.writeheader()

        if args.ingest_dir:
            print(f"\n📥 Ingesting local dataset: {args.ingest_dir}")
            ingest_local_dataset(args.ingest_dir, out_dir, args.source_name, log_writer)

        for row in species_rows:
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
            for cand in candidates[: args.max_per_species]:
                fname = f"{cand['source']}_{cand['image_id']}.jpg"
                dest_path = species_dir / fname
                ok = download_image(cand["url"], dest_path, session)
                if ok:
                    downloaded += 1
                log_writer.writerow(
                    {
                        "species": species_name,
                        "source": cand["source"],
                        "image_id": cand["image_id"],
                        "url": cand["url"],
                        "license": cand["license"],
                        "download_status": "ok" if ok else "failed",
                    }
                )
                time.sleep(0.2)

            print(
                f"  ✔ Downloaded {downloaded}/{len(candidates[: args.max_per_species])}"
            )

    print(f"\nDone. Provenance log: {provenance_path}")


if __name__ == "__main__":
    main()
