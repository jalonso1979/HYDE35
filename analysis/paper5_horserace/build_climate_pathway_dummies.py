# analysis/paper5_horserace/build_climate_pathway_dummies.py
"""Append HYDE-features-based pathway dummy columns to the master panel.

The master panel (deep_determinants_horserace.parquet) already carries
climate-only clustering dummies in pathway_0..4.  This script attaches
HYDE-trajectory-based clustering dummies from paper1_clustered_features.parquet
as hyde_pathway_0..4, so Exercise 3 can use them as an alternative mediator.

HYDE-feature clustering was constructed on:
  peak_ag_expansion_year, peak_pop_growth_year, max_density, density_1750,
  crop_share_1750, rice_share_1750, irrigation_1750, urban_1750,
  pop_growth_0_1000, pop_growth_1000_1750, ag_intensity_change.

These are downstream HYDE outcomes, making them an orthogonal robustness
check against the climate-primitive clustering used in Exercise 2.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
PANEL = ROOT / "analysis/data/deep_determinants_horserace.parquet"
HYDE_LABELS = ROOT / "analysis/data/paper1_clustered_features.parquet"


def main() -> None:
    df = pd.read_parquet(PANEL)
    hyde = pd.read_parquet(HYDE_LABELS)

    # Keep only valid ISO3 (3-letter alpha codes); drop numeric legacy codes
    hyde = hyde[hyde["iso3"].str.match(r"^[A-Z]{3}$", na=False)].copy()
    hyde = hyde[["iso3", "cluster"]].dropna(subset=["cluster"]).copy()
    hyde["cluster"] = hyde["cluster"].astype(int)

    # Drop singleton cluster (only 1 country) to avoid collinearity in regression
    counts = hyde["cluster"].value_counts()
    singleton_clusters = counts[counts <= 1].index.tolist()
    if singleton_clusters:
        print(f"Dropping singleton HYDE clusters (<=1 country): {singleton_clusters}")
        print(f"  Countries dropped: {hyde[hyde['cluster'].isin(singleton_clusters)]['iso3'].tolist()}")
        hyde = hyde[~hyde["cluster"].isin(singleton_clusters)]

    # Build one-hot dummies with prefix hyde_pathway_
    dummies = pd.get_dummies(hyde["cluster"], prefix="hyde_pathway").astype(float)
    hyde_d = pd.concat([hyde[["iso3"]], dummies], axis=1)

    # Drop any existing hyde_pathway_* columns before merge
    old_cols = [c for c in df.columns if c.startswith("hyde_pathway_")]
    if old_cols:
        df = df.drop(columns=old_cols)

    df = df.merge(hyde_d, on="iso3", how="left")

    n_dummies = len([c for c in df.columns if c.startswith("hyde_pathway_")])
    n_matched = df["hyde_pathway_0"].notna().sum() if "hyde_pathway_0" in df.columns else 0
    print(f"HYDE pathway dummies: {n_dummies} columns")
    print(f"Countries matched: {n_matched} / {len(df)}")
    print(f"Dummy columns: {[c for c in df.columns if c.startswith('hyde_pathway_')]}")

    df.to_parquet(PANEL, index=False)
    print(f"Updated {PANEL}")


if __name__ == "__main__":
    main()
