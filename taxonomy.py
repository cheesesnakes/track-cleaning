"""
taxonomy.py

Loads an Andaman & Nicobar (or any regional) species checklist and provides
genus / family resolution for arbitrary species labels.

Expected CSV columns: species (required), genus (optional), family (optional).

Used by track_cleaning.py, validate.py, and FishCropDataset.
"""

from pathlib import Path

import pandas as pd


class TaxonomyResolver:
    def __init__(self, checklist_path):
        path = Path(checklist_path)
        if not path.exists():
            raise SystemExit(f"Checklist not found: {path}")

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "species" not in df.columns:
            raise SystemExit("Checklist CSV must contain a 'species' column.")

        df["species_clean"] = (
            df["species"].astype(str).str.strip().str.lower().str.replace("_", " ")
        )

        self.species_to_genus = {}
        self.species_to_family = {}
        self.genus_to_family = {}

        if "genus" in df.columns:
            self.species_to_genus = dict(
                zip(df["species_clean"], df["genus"].astype(str).str.strip())
            )
            self.valid_genera = set(df["genus"].astype(str).str.strip().str.lower())
        else:
            self.valid_genera = set()

        if "family" in df.columns:
            self.species_to_family = dict(
                zip(df["species_clean"], df["family"].astype(str).str.strip())
            )
            self.valid_families = set(df["family"].astype(str).str.strip().str.lower())
            # genus -> family mapping (first observed wins)
            for g, f in zip(df.get("genus", []), df["family"]):
                g_lc = str(g).strip().lower()
                if g_lc and g_lc not in self.genus_to_family:
                    self.genus_to_family[g_lc] = str(f).strip()
        else:
            self.valid_families = set()

        self.valid_species = set(self.species_to_genus.keys())

    def resolve(self, species_label):
        """
        Return (canonical_species, genus, family).

        canonical_species: lowercased, underscore-free
        Falls back to first-token-as-genus if the species is not in the checklist,
        then to genus_to_family lookup for family.
        """
        sp = str(species_label).strip().lower().replace("_", " ")

        genus = self.species_to_genus.get(sp)
        family = self.species_to_family.get(sp)

        if genus is None:
            parts = sp.split()
            genus = parts[0].capitalize() if parts else "Unknown"

        if family is None:
            family = self.genus_to_family.get(genus.lower(), "Unknown_Family")

        return sp, genus, family

    def is_valid_species(self, label):
        sp = str(label).strip().lower().replace("_", " ")
        return bool(self.valid_species) and sp in self.valid_species
