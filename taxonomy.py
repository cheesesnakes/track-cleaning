"""
taxonomy.py

Loads an Andaman & Nicobar (or any regional) species checklist and provides
genus / family resolution for arbitrary species labels.

Expected CSV columns: species (required), genus (optional), family (optional).

Canonical species keys are lowercased, underscore-separated, and stripped of
punctuation.  This is the same form produced by canonical_species() and used
throughout the pipeline — crop directory names, checkpoint label maps, and
the output CSV all use one spelling:

    'Lutjanus decussatus'    →  'lutjanus_decussatus'
    'lutjanus_decussatus'    →  'lutjanus_decussatus'
    '  LUTJANUS DECUSSATUS  ' → 'lutjanus_decussatus'

Resolution is layered.  For a species not covered by the checklist itself,
the resolver tries, in order:

  1. A secondary taxonomy CSV (optional, `secondary_csv=`) containing
     broader genus→family or species→family mappings.
  2. A JSON cache on disk (`cache_path=`) of genera already resolved via
     GBIF in a previous run.
  3. The GBIF species/match API (`allow_gbif=True`, the default), queried
     once per genus and cached.  No API key required.

Pass `allow_gbif=False` for a fully offline, deterministic resolver.

Used by track_cleaning.py, validate.py, and FishCropDataset.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
DEFAULT_USER_AGENT = "andaman-fish-active-learning/1.0"


def canonical_species(text) -> str:
    """
    Filesystem- and CSV-safe species key: lowercase, spaces → underscores,
    punctuation stripped.  Single source of truth for species spelling.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9_]", "", text.strip().replace(" ", "_").lower())


class TaxonomyResolver:
    def __init__(
        self,
        checklist_path,
        secondary_csv=None,
        cache_path="./taxonomy/.gbif_cache.json",
        allow_gbif=True,
        gbif_timeout=10.0,
        gbif_min_interval=0.1,
        verbose=True,
    ):
        self.verbose = verbose
        self.allow_gbif = allow_gbif
        self.gbif_timeout = gbif_timeout
        self.gbif_min_interval = gbif_min_interval
        self.cache_path = Path(cache_path) if cache_path else None

        # Counters for finalize() reporting.
        self._stats = {
            "checklist": 0,
            "secondary": 0,
            "cache": 0,
            "gbif_hits": 0,
            "gbif_misses": 0,
            "still_unknown": 0,
        }
        self._lookup_attempted = set()
        self._last_gbif_call = 0.0

        # ---------------------------------------------------------------
        # Layer 1: primary checklist
        # ---------------------------------------------------------------
        path = Path(checklist_path)
        if not path.exists():
            raise SystemExit(f"Checklist not found: {path}")

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "species" not in df.columns:
            raise SystemExit("Checklist CSV must contain a 'species' column.")

        df["species_key"] = df["species"].astype(str).map(canonical_species)
        df = df[df["species_key"] != ""]

        dupes = df["species_key"][df["species_key"].duplicated()].unique()
        if len(dupes) and verbose:
            preview = ", ".join(dupes[:5])
            more = "" if len(dupes) <= 5 else f" (+{len(dupes) - 5} more)"
            print(
                f" ⚠ Checklist has {len(dupes)} duplicate species rows "
                f"(last wins): {preview}{more}"
            )

        self.species_to_genus = {}
        self.species_to_family = {}
        self.genus_to_family = {}
        self.valid_species = set(df["species_key"])

        genus_series = None
        if "genus" in df.columns:
            genus_series = df["genus"].astype(str).str.strip().str.title()
            self.species_to_genus = dict(zip(df["species_key"], genus_series))
            self.valid_genera = {g.lower() for g in genus_series if g}
        else:
            self.valid_genera = set()

        if "family" in df.columns:
            family_series = df["family"].astype(str).str.strip().str.title()
            self.species_to_family = dict(zip(df["species_key"], family_series))
            self.valid_families = {f.lower() for f in family_series if f}
            if genus_series is not None:
                for g, f in zip(genus_series, family_series):
                    g_lc = g.strip().lower()
                    if g_lc and g_lc not in self.genus_to_family:
                        self.genus_to_family[g_lc] = f
        else:
            self.valid_families = set()

        # ---------------------------------------------------------------
        # Layer 2: secondary taxonomy CSV (optional)
        # ---------------------------------------------------------------
        if secondary_csv:
            self._load_secondary(secondary_csv)

        # ---------------------------------------------------------------
        # Layer 3: JSON cache of previous GBIF lookups
        # ---------------------------------------------------------------
        self._load_cache()

    # ------------------------------------------------------------------
    # Layer loaders
    # ------------------------------------------------------------------
    def _load_secondary(self, path):
        p = Path(path)
        if not p.exists():
            if self.verbose:
                print(f" ⚠ Secondary taxonomy CSV not found: {p}")
            return

        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        if "family" not in df.columns:
            if self.verbose:
                print(f" ⚠ {p} has no 'family' column — ignored.")
            return

        n_g = n_s = 0
        if "genus" in df.columns:
            for g, f in zip(df["genus"], df["family"]):
                g_lc = str(g).strip().lower()
                f_s = str(f).strip().title()
                if g_lc and f_s and g_lc not in self.genus_to_family:
                    self.genus_to_family[g_lc] = f_s
                    n_g += 1

        if "species" in df.columns:
            for s, f in zip(df["species"], df["family"]):
                key = canonical_species(str(s))
                f_s = str(f).strip().title()
                if key and f_s and key not in self.species_to_family:
                    self.species_to_family[key] = f_s
                    n_s += 1

        if self.verbose:
            print(f" ➔ Secondary taxonomy: {n_g} genera, {n_s} species.")

    def _load_cache(self):
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            if self.verbose:
                print(f" ⚠ Could not read {self.cache_path}: {exc}")
            return

        cached = data.get("genus_to_family", {})
        added = 0
        for g, f in cached.items():
            if not f:
                continue
            g_lc = g.lower()
            if g_lc not in self.genus_to_family:
                self.genus_to_family[g_lc] = f
                added += 1
        if self.verbose and added:
            print(f" ➔ Taxonomy cache: {added} new genera from {self.cache_path}.")

    def _save_cache(self):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"genus_to_family": dict(sorted(self.genus_to_family.items()))},
                f,
                indent=2,
            )
        tmp.replace(self.cache_path)

    # ------------------------------------------------------------------
    # Layer 4: GBIF
    # ------------------------------------------------------------------
    def _gbif_lookup(self, genus):
        if not genus:
            return None
        g = genus.strip()
        if not g or g.lower() in ("unknown", "unidentified", "sp", "spp"):
            return None

        dt = time.time() - self._last_gbif_call
        if dt < self.gbif_min_interval:
            time.sleep(self.gbif_min_interval - dt)

        params = urllib.parse.urlencode({"name": g, "rank": "GENUS"})
        url = f"{GBIF_MATCH_URL}?{params}"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": DEFAULT_USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=self.gbif_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"   ⚠ GBIF lookup failed for {g}: {exc}")
            return None
        finally:
            self._last_gbif_call = time.time()

        if payload.get("matchType") in (None, "NONE"):
            return None
        fam = payload.get("family")
        return str(fam).strip().title() if fam else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(self, species_label):
        """
        Return (canonical_species, genus, family).

        canonical_species is the underscored, filesystem-safe key.  genus /
        family come from the checklist, the secondary CSV, the cache, or
        GBIF (in that order), falling back to first-token genus and
        'Unknown_Family'.
        """
        sp_key = canonical_species(species_label)
        sp_space = sp_key.replace("_", " ")

        # Checklist hit for both genus and family — common case, fast path.
        genus = self.species_to_genus.get(sp_key)
        family = self.species_to_family.get(sp_key)
        if genus is not None and family is not None:
            self._stats["checklist"] += 1
            return sp_key, genus, family

        # Derive a genus if the checklist didn't give one.
        if genus is None:
            parts = sp_space.split()
            genus = parts[0].title() if parts else "Unknown"

        # Family still missing: try genus_to_family, then GBIF.
        if family is None:
            g_lc = genus.lower()
            family = self.genus_to_family.get(g_lc)
            if family:
                self._stats["secondary"] += 1
            elif self.allow_gbif and g_lc and g_lc not in self._lookup_attempted:
                self._lookup_attempted.add(g_lc)
                family = self._gbif_lookup(genus)
                if family:
                    self.genus_to_family[g_lc] = family
                    self._stats["gbif_hits"] += 1
                    if self.verbose:
                        print(f"   ⤷ GBIF: {genus} → {family}")
                else:
                    self._stats["gbif_misses"] += 1

        if family is None:
            self._stats["still_unknown"] += 1
            family = "Unknown_Family"

        return sp_key, genus, family

    def is_valid_species(self, label):
        """True iff the canonical form of `label` appears in the checklist."""
        return canonical_species(label) in self.valid_species

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------
    def prewarm(self, genera, batch_log_every=25):
        """
        Resolve a list of genera up front via GBIF, so a later image scan
        doesn't interleave network I/O.  At most one query per unique genus
        per process; already-cached genera are skipped.
        """
        if not self.allow_gbif:
            return
        unique = sorted({g for g in genera if g})
        pending = [g for g in unique if g.lower() not in self.genus_to_family]
        if not pending:
            if self.verbose:
                print(" ➔ All reference genera already resolved.")
            return

        if self.verbose:
            print(
                f" ➔ Querying GBIF for {len(pending)} unknown genera "
                f"(cached after this)…"
            )
        for i, g in enumerate(pending, 1):
            self._lookup_attempted.add(g.lower())
            fam = self._gbif_lookup(g)
            if fam:
                self.genus_to_family[g.lower()] = fam
                self._stats["gbif_hits"] += 1
            else:
                self._stats["gbif_misses"] += 1
            if self.verbose and i % batch_log_every == 0:
                print(f"   prewarm {i}/{len(pending)} …")
        self._save_cache()

    def finalize(self):
        """Persist the cache and print a one-line summary."""
        self._save_cache()
        if self.verbose:
            s = self._stats
            print(
                f" ➔ Family resolution: "
                f"{s['checklist']} checklist, "
                f"{s['secondary']} secondary/cache, "
                f"{s['gbif_hits']} GBIF, "
                f"{s['gbif_misses']} GBIF misses, "
                f"{s['still_unknown']} still unknown."
            )
