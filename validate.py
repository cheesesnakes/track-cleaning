"""
validate.py

Sanity-check the classifier's assigned_species / assigned_genus /
assigned_family columns against a published Andaman & Nicobar checklist.
Flags rows whose assigned labels are missing from the checklist — a
likely sign of pretraining-source bias.

Additionally emits `suggested_species/genus/family` for flagged rows
using the resolver's fallback logic, so you get a proposed correction.

Usage:
    python validate.py \
        --master-csv tracking_master.csv \
        --checklist-csv andaman_checklist.csv \
        --report-out validation_report.csv
"""

import argparse

import pandas as pd

from taxonomy import TaxonomyResolver


def main():
    parser = argparse.ArgumentParser(description="Validate assigned labels.")
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--checklist-csv", required=True)
    parser.add_argument("--report-out", default="./output/validation_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.master_csv)
    required_cols = {"assigned_species", "assigned_genus", "assigned_family"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise SystemExit(f"Master CSV missing columns: {missing_cols}")

    resolver = TaxonomyResolver(args.checklist_csv)

    labeled = df.dropna(subset=["assigned_species"]).copy()
    labeled["species_lc"] = (
        labeled["assigned_species"].str.replace("_", " ").str.strip().str.lower()
    )
    labeled["genus_lc"] = labeled["assigned_genus"].astype(str).str.strip().str.lower()
    labeled["family_lc"] = (
        labeled["assigned_family"].astype(str).str.strip().str.lower()
    )

    # Presence flags (empty checklist sets skip that level silently)
    labeled["species_flagged"] = labeled["species_lc"].apply(
        lambda s: bool(resolver.valid_species) and s not in resolver.valid_species
    )
    labeled["genus_flagged"] = labeled["genus_lc"].apply(
        lambda g: bool(resolver.valid_genera) and g not in resolver.valid_genera
    )
    labeled["family_flagged"] = labeled["family_lc"].apply(
        lambda f: bool(resolver.valid_families) and f not in resolver.valid_families
    )

    # Suggested corrections via resolver
    suggestions = labeled["species_lc"].apply(
        lambda s: pd.Series(
            resolver.resolve(s), index=["sug_species", "sug_genus", "sug_family"]
        )
    )
    labeled = pd.concat([labeled, suggestions], axis=1)

    flagged_mask = (
        labeled["species_flagged"]
        | labeled["genus_flagged"]
        | labeled["family_flagged"]
    )
    flagged = labeled[flagged_mask]

    print("=== Validation summary ===")
    print(f"Total labeled rows checked:      {len(labeled)}")
    print(f"Species not in checklist:        {int(labeled['species_flagged'].sum())}")
    print(f"Genus not in checklist:          {int(labeled['genus_flagged'].sum())}")
    print(f"Family not in checklist:         {int(labeled['family_flagged'].sum())}")

    if not flagged.empty:
        summary = (
            flagged.groupby(["assigned_family", "assigned_genus", "assigned_species"])
            .size()
            .reset_index(name="row_count")
            .sort_values("row_count", ascending=False)
        )
        print("\nTop flagged family/genus/species combinations:")
        print(summary.head(20).to_string(index=False))

        flagged.to_csv(args.report_out, index=False)
        print(f"\nFull flagged report written to {args.report_out}")
    else:
        print("\nNo flagged rows — all assigned labels match the checklist.")


if __name__ == "__main__":
    main()
